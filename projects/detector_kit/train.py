"""Train a DINO-DETR head on FLIR (rgb or thermal) with the F8 recipe, with or without compiled CUDA ops.

FLIR frames → ``native`` preprocessing (448×560) → flip → frozen DINOv2-Reg + PCA (SDPA, fp16)
→ DINO-DETR head (``irwm.detection.lightning.DetectorModule``). Needs a CUDA GPU (detrex's
denoising hard-codes ``.cuda()``), but not detrex's compiled ops: ``msda=auto`` falls back to
detrex's PyTorch deformable attention. Normally launched through ``train.sh``::

    bash projects/detector_kit/train.sh --modality thermal --batch-size 8

Writes ``<run_dir>/val_metrics/epoch_<NNN>.json``, ``<run_dir>/csv/.../metrics.csv`` and
``<run_dir>/train_summary.json``; checkpoints (``best-eNNN.ckpt``, ``last.ckpt``) in
``${paths.ckpt_root}/detector_kit/<run_name>/``. A rerun with the same ``run_name`` resumes.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from irwm.utils.paths import load_env
from irwm.utils.slurm import cluster_plugins, on_slurm, resume_ckpt_path
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()


def _container(node: DictConfig) -> dict:
    return OmegaConf.to_container(node, resolve=True)  # type: ignore[return-value]


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.modality not in ("rgb", "thermal"):
        sys.exit(f"[train] modality must be 'rgb' or 'thermal', got {cfg.modality!r}")
    if not torch.cuda.is_available():
        sys.exit("[train] no CUDA GPU visible. DINO training needs one (detrex's denoising code calls .cuda()).")
    # Heavy imports after the checks, so a misconfigured machine fails in seconds.
    from irwm.data.detection import FlirDetectionDataModule
    from irwm.data.flir import flir_root
    from irwm.detection.detrex_adapter import select_msda
    from irwm.detection.lightning import DetectorModule

    msda = select_msda(cfg.msda)
    print(f"[train] GPU {torch.cuda.get_device_name()} | deformable attention: {msda}"
          + ("" if msda == "cuda" else " (PyTorch fallback, no compiled detrex ops: slower, same results)"),
          flush=True)

    L.seed_everything(cfg.seed, workers=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.project / cfg.run_name

    dm = FlirDetectionDataModule(modality=cfg.modality, policy=cfg.encoder.policy, batch_size=cfg.data.batch_size,
                                 num_workers=cfg.data.num_workers, flip_p=cfg.data.flip_p)
    dm.setup("fit")
    optim = _container(cfg.optim)
    optim["lr_drop_epoch"] = round(optim.pop("lr_drop_fraction") * cfg.trainer.max_epochs)
    model = DetectorModule(detector=_container(cfg.detector), encoder=_container(cfg.encoder), optim=optim,
                           val_gt=dm.val_gt)
    model.hparams["modality"] = cfg.modality  # read back by inference.py from the checkpoint
    model.metrics_dir = run_dir / "val_metrics"

    loggers = [make_wandb_logger(cfg, save_dir=run_dir), CSVLogger(run_dir, name="csv")]
    counts = {"n_train_frames": len(dm.train_set), "n_dropped_empty": dm.train_set.n_dropped_empty,
              "n_val_frames": len(dm.val_set)}
    for logger in loggers:
        logger.log_metrics({f"data/{k}": v for k, v in counts.items()}, step=0)

    best = ModelCheckpoint(dirpath=ckpt_dir, monitor="val/AP", mode="max", save_top_k=1, save_last=True,
                           filename="best-e{epoch:03d}", auto_insert_metric_name=False)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.cluster.devices,
        num_nodes=cfg.cluster.num_nodes,
        logger=loggers,
        plugins=cluster_plugins(),
        callbacks=[best],
        default_root_dir=ckpt_dir,
        enable_progress_bar=not on_slurm(),
        **_container(cfg.trainer),
    )
    resume = resume_ckpt_path(ckpt_dir)
    print(f"[train] {cfg.modality}: {counts['n_train_frames']} train / {counts['n_val_frames']} val frames | "
          f"checkpoints -> {ckpt_dir}" + (f" | resuming from {resume}" if resume else ""), flush=True)
    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm, ckpt_path=resume)
    if not trainer.is_global_zero:
        return

    m = re.search(r"best-e(\d+)\.ckpt$", best.best_model_path or "")
    result = {"modality": cfg.modality, "run_name": cfg.run_name, "msda": msda,
              "best_ckpt": best.best_model_path or None,
              "best_val_AP": float(best.best_model_score) if best.best_model_score is not None else None,
              "best_epoch": int(m.group(1)) if m else None, "last_ckpt": str(ckpt_dir / "last.ckpt"),
              **counts, "fit_wall_s": time.perf_counter() - t0, "device": torch.cuda.get_device_name(),
              "data_root": str(flir_root()), "git": git_info()}
    (run_dir / "train_summary.json").write_text(json.dumps(result, indent=2))
    print(f"[train] best val/AP {result['best_val_AP']} at epoch {result['best_epoch']}\n"
          f"[train] best checkpoint: {result['best_ckpt']}\n"
          f"[train] summary: {run_dir / 'train_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
