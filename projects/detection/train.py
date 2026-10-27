"""Train one DINO-DETR head (``det_rgb`` / ``det_ir``) on FLIR features encoded on the fly.

FLIR frames → ``native`` preprocessing → flip → frozen DINOv2-Reg + PCA (SDPA, fp16) → F7 head,
with detrex's DINO-ViTDet 50ep recipe in ``16-mixed``. Needs CUDA and detrex's CUDA ops:

    IRWM_STAGE_DATA=1 bash slurm/submit.sh ctcomp3 projects/detection/train.py experiment=det_rgb \\
        cluster.timeout_min=1200 cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32
    # pipeline check on the first 256 non-empty train frames (train = val, no flip):
    IRWM_STAGE_DATA=1 bash slurm/submit.sh ctcomp3 projects/detection/train.py experiment=det_ir \\
        overfit.enabled=true cluster.timeout_min=120 cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32

Writes ``<run_dir>/val_metrics/epoch_<NNN>.json`` per validation, then ``train_summary.json``
(normal mode) or ``overfit.json`` (overfit mode). Checkpoints in ``${paths.ckpt_root}/detection/<run_name>/``.
Spec: spec/2026-10-26-detector-training-pipeline (FR8).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from irwm.detection import cuda_ops_available
from irwm.utils.paths import load_env
from irwm.utils.slurm import cluster_plugins, on_slurm, resume_ckpt_path
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()


class OverfitGate(Callback):
    """Records every validation and stops at the first one where ``AP50_<c> ≥ bar`` for every class."""

    def __init__(self, classes: list[str], bar: float) -> None:
        self.classes = list(classes)
        self.bar = bar
        self.validations: list[dict] = []
        self.passed_step: int | None = None

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:  # amendment 2026-10-26: the sanity check is not scored
            return
        metrics = {k.removeprefix("val/"): float(v) for k, v in trainer.callback_metrics.items()
                   if k.startswith("val/")}
        self.validations.append({"global_step": trainer.global_step, "metrics": metrics})
        if all(metrics.get(f"AP50_{c}", 0.0) >= self.bar for c in self.classes):
            self.passed_step = trainer.global_step
            trainer.should_stop = True


def _container(node: DictConfig) -> dict:
    return OmegaConf.to_container(node, resolve=True)  # type: ignore[return-value]


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    if not cuda_ops_available():
        sys.exit("projects/detection/train.py: cuda_ops_available() is False. DINO training needs CUDA and "
                 "detrex's compiled ops (build them with slurm/setup_env.sh on a node with nvcc).")
    # Heavy imports after the check, so a misconfigured machine fails in seconds.
    from irwm.data.detection import FlirDetectionDataModule, first_nonempty_ids
    from irwm.data.flir import FlirSplit, flir_root
    from irwm.detection.lightning import DetectorModule

    L.seed_everything(cfg.seed, workers=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    # Stable across SLURM requeues (run_name = slurm-<job id>), unlike the timestamped run dir.
    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.project / cfg.run_name
    overfit = cfg.overfit.enabled

    dm_kwargs = dict(modality=cfg.modality, policy=cfg.encoder.policy, batch_size=cfg.data.batch_size,
                     num_workers=cfg.data.num_workers, flip_p=cfg.data.flip_p)
    if overfit:
        ids = first_nonempty_ids(FlirSplit.load(cfg.modality, "train"), cfg.overfit.n_frames, cfg.encoder.policy)
        dm_kwargs.update(flip_p=0.0, train_image_ids=ids, val_split="train", val_image_ids=ids)
    dm = FlirDetectionDataModule(**dm_kwargs)
    dm.setup("fit")
    optim = _container(cfg.optim)
    if overfit:
        optim["lr_drop_epoch"] = None  # constant lr (amendment 2026-10-26)
    model = DetectorModule(detector=_container(cfg.detector), encoder=_container(cfg.encoder), optim=optim,
                           val_gt=dm.val_gt)
    model.metrics_dir = run_dir / "val_metrics"

    logger = make_wandb_logger(cfg, save_dir=run_dir)
    counts = {"n_train_frames": len(dm.train_set), "n_dropped_empty": dm.train_set.n_dropped_empty,
              "n_val_frames": len(dm.val_set)}
    logger.log_metrics({f"data/{k}": v for k, v in counts.items()}, step=0)

    trainer_kwargs = _container(cfg.trainer)
    if overfit:
        gate = OverfitGate(cfg.overfit.classes, cfg.overfit.ap50_bar)
        callbacks = [ModelCheckpoint(dirpath=ckpt_dir, save_top_k=0, save_last=True), gate]
        trainer_kwargs.update(max_steps=cfg.overfit.max_steps, max_epochs=-1, val_check_interval=cfg.overfit.eval_every,
                              check_val_every_n_epoch=None)
    else:
        best = ModelCheckpoint(dirpath=ckpt_dir, monitor="val/AP", mode="max", save_top_k=1, save_last=True,
                               filename="best-e{epoch:03d}", auto_insert_metric_name=False)
        callbacks = [best]
    trainer = L.Trainer(
        accelerator=cfg.cluster.accelerator,
        devices=cfg.cluster.devices,
        num_nodes=cfg.cluster.num_nodes,
        logger=logger,
        plugins=cluster_plugins(),
        callbacks=callbacks,
        # SLURM auto-requeue writes hpc_ckpt_*.ckpt here and restores from it on restart.
        default_root_dir=ckpt_dir,
        enable_progress_bar=not on_slurm(),
        **trainer_kwargs,
    )
    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt_path(ckpt_dir))
    fit_s = time.perf_counter() - t0
    if not trainer.is_global_zero:
        return

    common = {**counts, "fit_wall_s": fit_s, "data_root": str(flir_root()), "git": git_info()}
    if overfit:
        step_ms = model._step_ms
        result = {"passed": gate.passed_step is not None, "passed_step": gate.passed_step,
                  "ap50_bar": cfg.overfit.ap50_bar, "classes": list(cfg.overfit.classes),
                  "batch_size": cfg.data.batch_size, "validations": gate.validations,
                  "median_step_ms": statistics.median(step_ms) if step_ms else None,
                  "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, **common}
        (run_dir / "overfit.json").write_text(json.dumps(result, indent=2))
        print(f"[overfit] passed={result['passed']} step={gate.passed_step} -> {run_dir / 'overfit.json'}", flush=True)
    else:
        m = re.search(r"best-e(\d+)\.ckpt$", best.best_model_path or "")
        result = {"best_ckpt": best.best_model_path or None,
                  "best_val_AP": float(best.best_model_score) if best.best_model_score is not None else None,
                  "best_epoch": int(m.group(1)) if m else None, **common}
        (run_dir / "train_summary.json").write_text(json.dumps(result, indent=2))
        print(f"[train] best val/AP {result['best_val_AP']} at epoch {result['best_epoch']} "
              f"-> {run_dir / 'train_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
