"""Lightning training of the DINO-DETR head on features encoded on the fly by the frozen encoder.

Spec: spec/2026-10-26-detector-training-pipeline (FR5). The recipe is detrex's DINO-ViTDet 50ep
(AdamW lr 1e-4, wd 1e-4, no weight decay on norm layers, lr ×0.1 from epoch 40), with the
deviations E1-E6 of plan.md; the values come from the Hydra config (projects/detection/conf).

- ``self.encoder`` is the frozen F4 ``DinoV2RegEncoder`` (SDPA attention by default). It is a
  registered submodule, so Lightning moves it to the GPU, but checkpoints never carry its weights:
  ``on_save_checkpoint`` strips them, ``on_load_checkpoint`` puts the live ones back.
- Only ``self.detector`` (F7 ``DINODetector``) is trained. Training needs CUDA (detrex).
- Validation scores the whole val set once per validation with pycocotools
  (``irwm.detection.coco_val``) on rank 0, writes ``<metrics_dir>/epoch_<NNN>.json`` and logs ``val/*``.

Imported explicitly (``irwm.detection.lightning``), not from ``irwm.detection``, so the package stays light.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import torch
from detectron2.solver.build import get_default_optimizer_params
from torch import Tensor, nn

from irwm.detection.coco_val import coco_map, to_flir_detections
from irwm.detection.detector import DINODetector
from irwm.features.preprocess import POLICY_SIZE
from irwm.utils.tracking import git_info

BASE_LOSSES = ("loss_class", "loss_bbox", "loss_giou", "loss_class_dn", "loss_bbox_dn", "loss_giou_dn")
MAX_NONFINITE_RUN = 100  # consecutive non-finite losses before training is aborted


class DetectorModule(L.LightningModule):
    """Frozen encoder + trainable ``DINODetector``; batches from ``irwm.data.detection.collate_detection``."""

    def __init__(self, *, detector: Mapping[str, Any], encoder: Mapping[str, Any], optim: Mapping[str, Any],
                 val_gt: dict | None = None, encoder_module: nn.Module | None = None, perf_every: int = 50) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["val_gt", "encoder_module"])
        if encoder_module is None:
            from irwm.features.encoder import DinoV2RegEncoder, set_attention

            encoder_module = DinoV2RegEncoder.from_assets(precision=encoder["precision"],
                                                          hub_version=encoder["hub_version"])
            set_attention(encoder_module.backbone, encoder.get("attention", "sdpa"))
        self.encoder = encoder_module  # frozen: F4 forces eval() and requires_grad_(False)
        self.detector = DINODetector(**detector)
        self.image_size: tuple[int, int] = POLICY_SIZE[encoder["policy"]]
        self.val_gt = val_gt
        self.metrics_dir: Path | None = None  # set by the entry point to <hydra run dir>/val_metrics
        self.perf_every = perf_every
        # A buffer, so the total survives SLURM requeues through the checkpoint.
        self.register_buffer("_nonfinite_steps", torch.zeros((), dtype=torch.long))
        self._nonfinite_run = 0
        self._val_dets: dict[int, list[dict]] = {}
        self._timed = False
        self._t_start = self._t_encoded = 0.0
        self._epoch_t0 = 0.0
        self._epoch_images = 0
        self._step_ms: list[float] = []  # every timed step, for the overfit summary (FR8)
        self._epoch_perf: list[tuple[float, float]] = []  # (encode_ms, step_ms) of this epoch's timed steps

    # ------------------------------------------------------------------ data
    def transfer_batch_to_device(self, batch: dict, device: torch.device, dataloader_idx: int) -> dict:
        """Move images and targets only; geometries are frozen dataclasses Lightning cannot copy."""
        out = dict(batch)
        out["images"] = batch["images"].to(device, non_blocking=True)
        out["targets"] = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch["targets"]]
        return out

    def encode(self, images: Tensor) -> Tensor:
        """``[B, 3, H, W]`` normalised images → ``[B, 1152, H', W']`` PCA features (no grad)."""
        return self.encoder.encode_pca(images).permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------ train
    def _count_nonfinite(self, loss: Tensor) -> None:
        """Count non-finite losses; ``MAX_NONFINITE_RUN`` in a row raise ``RuntimeError``."""
        if torch.isfinite(loss).item():
            self._nonfinite_run = 0
            return
        self._nonfinite_steps += 1
        self._nonfinite_run += 1
        if self._nonfinite_run >= MAX_NONFINITE_RUN:
            raise RuntimeError(f"{self._nonfinite_run} consecutive non-finite losses "
                               f"(last {float(loss)}); stopping (plan.md 'Precision')")

    def _sync(self) -> float:
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def on_train_epoch_start(self) -> None:
        self._epoch_t0 = time.perf_counter()
        self._epoch_images = 0
        self._epoch_perf = []

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:
        self._timed = self.device.type == "cuda" and batch_idx % self.perf_every == 0
        if self._timed:
            self._t_start = self._sync()

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        feats = self.encode(batch["images"])
        if self._timed:
            self._t_encoded = self._sync()
        losses = self.detector(feats, batch["targets"], image_size=self.image_size)
        loss = sum(losses.values())
        self._count_nonfinite(loss)
        bs = len(batch["targets"])
        self._epoch_images += bs
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=bs)
        for name in BASE_LOSSES:
            self.log(f"train/{name}", losses[name], on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/nonfinite_steps", self._nonfinite_steps.float(), on_step=True, on_epoch=False, batch_size=bs)
        return loss

    def on_train_batch_end(self, outputs: Any, batch: dict, batch_idx: int) -> None:
        if not self._timed:
            return
        t_end = self._sync()
        encode_ms = 1e3 * (self._t_encoded - self._t_start)
        step_ms = 1e3 * (t_end - self._t_start)
        self._step_ms.append(step_ms)
        self._epoch_perf.append((encode_ms, step_ms))

    def on_train_epoch_end(self) -> None:
        epoch_s = time.perf_counter() - self._epoch_t0
        self.log("perf/epoch_s", epoch_s, on_step=False, on_epoch=True)
        self.log("perf/train_images_per_s", self._epoch_images / epoch_s, on_step=False, on_epoch=True)
        # Once per epoch (amendment 2026-10-27): Lightning writes on_step values only every
        # log_every_n_steps steps, so per-step timings on other steps would be dropped.
        if self._epoch_perf:
            encode_ms = statistics.median(e for e, _ in self._epoch_perf)
            step_ms = statistics.median(s for _, s in self._epoch_perf)
            self.log("perf/encode_ms", encode_ms, on_step=False, on_epoch=True)
            self.log("perf/step_ms", step_ms, on_step=False, on_epoch=True)
            self.log("perf/encode_share", statistics.median(e / s for e, s in self._epoch_perf),
                     on_step=False, on_epoch=True)
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            self.log("train/grad_scale", float(scaler.get_scale()), on_step=False, on_epoch=True)

    # ------------------------------------------------------------------ val
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        preds = self.detector(self.encode(batch["images"]), image_size=self.image_size)
        for image_id, geom, pred in zip(batch["image_ids"], batch["geometries"], preds):
            self._val_dets[image_id] = to_flir_detections([pred], [image_id], [geom])

    def on_validation_epoch_end(self) -> None:
        dets, self._val_dets = self._val_dets, {}
        if self.trainer.sanity_checking:  # amendment 2026-10-26: the sanity batches are not scored
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[dict[int, list[dict]] | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, dets)
            dets = {}
            for part in gathered:  # dedupe by image_id: the DDP sampler pads with repeated frames
                for image_id, d in (part or {}).items():
                    dets.setdefault(image_id, d)

        metrics: dict[str, float] | None = None
        if self.trainer.is_global_zero:
            metrics = coco_map(self.val_gt, [d for image_id in sorted(dets) for d in dets[image_id]])
            if self.metrics_dir is not None:
                self.metrics_dir.mkdir(parents=True, exist_ok=True)
                record = {"metrics": metrics, "epoch": self.current_epoch, "global_step": self.global_step,
                          "git": git_info()}
                path = self.metrics_dir / f"epoch_{self.current_epoch:03d}.json"
                path.write_text(json.dumps(record, indent=2))
        metrics = self.trainer.strategy.broadcast(metrics, src=0)
        for key, value in metrics.items():
            self.log(f"val/{key}", value, sync_dist=False)

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self) -> dict:
        o = self.hparams.optim
        factor = float(o["backbone_lr_factor"])
        params = get_default_optimizer_params(self.detector, base_lr=o["lr"], weight_decay_norm=0.0,
                                              lr_factor_func=lambda name: factor if "backbone" in name else 1.0)
        opt = torch.optim.AdamW(params, lr=o["lr"], betas=tuple(o["betas"]), weight_decay=o["weight_decay"])
        if o["lr_drop_epoch"] is None:  # overfit mode: constant lr (amendment 2026-10-26)
            return {"optimizer": opt}
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[o["lr_drop_epoch"]], gamma=0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    # ------------------------------------------------------------------ checkpoints
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("encoder.")}
        checkpoint["irwm"] = {"metrics_dir": str(self.metrics_dir) if self.metrics_dir else None}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"].update({f"encoder.{k}": v for k, v in self.encoder.state_dict().items()})
