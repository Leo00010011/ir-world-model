"""Train one aligned FLIR v1 detector (F18): ``aligned_<dinopca|dino3072|mae>_<rgb|ir>``.

The DINO heads (``dinopca``, ``dino3072``): aligned frames → ``native`` 448×560 → flip → frozen
DINOv2-Reg encoder (SDPA, fp16; ``encoder.kind``) → F7 head (2 classes). The ``mae`` heads: frames →
``native640`` 512×640 → flip → detrex's stock DINO-ViTDet, MAE-initialised and trained end to end.
Recipe: ``conf/train_aligned.yaml`` (20 epochs, lr 1e-4, ×0.1 from epoch 14, validation every epoch).

Every head is its own SLURM job; ``slurm/train_aligned.sh`` submits them (all six, or a subset)::

    bash slurm/train_aligned.sh ctcomp3 [--smoke] [--heads aligned_dinopca_rgb,aligned_mae_ir]
    # one head by hand (what the launcher runs):
    IRWM_STAGE_DATA=1 IRWM_STAGE_DATASETS="aligned/align" bash slurm/submit.sh ctcomp3 \\
        projects/detection/train_aligned.py aligned=aligned_dinopca_rgb cluster.timeout_min=360 ...

Writes ``<run_dir>/val_metrics/epoch_<NNN>.json`` per validation, then ``train_summary.json``
(or ``smoke.json`` with ``smoke.enabled=true``), both with the GPU the run used and its GPU-hours.
Checkpoints (``best-eNNN.ckpt`` on ``val/AP``, ``last.ckpt``) in ``${paths.ckpt_root}/detection/<run_name>/``.

After a complete (non-smoke) fit, rank 0 exports the best checkpoint as the frozen head
``$IRWM_CKPT_ROOT/detection/frozen/<name>/`` (``export.enabled``, write-once unless ``export.overwrite``),
so ``slurm/eval_aligned.sh`` can run with ``--dependency=afterok:<train job ids>``. A failed export
is recorded in ``train_summary.json`` and fails the job. Smoke runs never export.
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

from irwm.utils.paths import load_env
from irwm.utils.slurm import cluster_plugins, on_slurm, resume_ckpt_path
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()


class ValRecorder(Callback):
    """Keeps every scored validation (smoke mode summary)."""

    def __init__(self) -> None:
        self.validations: list[dict] = []

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = {k.removeprefix("val/"): float(v) for k, v in trainer.callback_metrics.items()
                   if k.startswith("val/")}
        self.validations.append({"global_step": trainer.global_step, "metrics": metrics})


def _container(node: DictConfig) -> dict:
    return OmegaConf.to_container(node, resolve=True)  # type: ignore[return-value]


def check_config(cfg: DictConfig) -> None:
    """Fail fast on an inconsistent aligned config (before any data or weights are loaded)."""
    from irwm.data.aligned import ALIGNED_NUM_CLASSES
    from irwm.detection.lightning import VITDET_ARCH, detector_arch
    from irwm.features.preprocess import POLICY_SIZE

    if cfg.dataset != "aligned_v1":
        raise ValueError(f"train_aligned.py trains aligned_v1 heads, got dataset={cfg.dataset!r}")
    if not str(cfg.name).startswith("aligned_"):
        raise ValueError(f"aligned head names start with 'aligned_', got {cfg.name!r}")
    if cfg.detector.num_classes != ALIGNED_NUM_CLASSES:
        raise ValueError(f"detector.num_classes {cfg.detector.num_classes} != {ALIGNED_NUM_CLASSES} (person, car)")
    arch = detector_arch(cfg.detector)
    policy = cfg.encoder.policy
    expected = "native640" if arch == VITDET_ARCH else "native"
    if policy != expected:
        raise ValueError(f"{cfg.name}: encoder.policy {policy!r}, expected {expected!r} for arch {arch!r}")
    if arch == VITDET_ARCH and cfg.detector.get("image_hw") is not None:
        if list(cfg.detector.image_hw) != list(POLICY_SIZE[policy]):
            raise ValueError(f"detector.image_hw {list(cfg.detector.image_hw)} != {list(POLICY_SIZE[policy])}")


def fit_complete(trainer: L.Trainer) -> bool:
    """True iff ``trainer.fit`` ran to ``max_epochs`` (not interrupted, not a requeue's partial run)."""
    from lightning.pytorch.trainer.states import TrainerStatus

    max_epochs = trainer.max_epochs if trainer.max_epochs is not None else -1
    return (trainer.state.status == TrainerStatus.FINISHED and not trainer.interrupted
            and max_epochs > 0 and trainer.current_epoch >= max_epochs)


def export_best(name: str, best_ckpt: str | Path, *, overwrite: bool = False) -> Path:
    """Export ``best_ckpt`` as frozen head ``name`` (``aligned_mae_*`` → full ViTDet, else ``DINODetector``)."""
    from irwm.detection.frozen import export_frozen
    from irwm.detection.frozen_vitdet import MAE_NAMES, export_frozen_vitdet

    if not str(name).startswith("aligned_"):
        raise ValueError(f"refusing to export {name!r}: not an aligned head name")
    if name in MAE_NAMES:
        return export_frozen_vitdet(name, Path(best_ckpt), overwrite=overwrite)
    return export_frozen(name, Path(best_ckpt), overwrite=overwrite, dataset="aligned_v1")


@hydra.main(version_base="1.3", config_path="conf", config_name="train_aligned")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    if not torch.cuda.is_available():
        sys.exit("projects/detection/train_aligned.py: CUDA is not available; DINO training needs a GPU "
                 "(detrex hard-codes .cuda() in denoising).")
    # Heavy imports after the checks, so a misconfigured machine fails in seconds.
    from irwm.data.aligned import ALIGNED_CLASS_NAMES, ALIGNED_LABEL_TO_CATEGORY_ID, AlignedSplit, aligned_root
    from irwm.data.detection import AlignedDetectionDataModule, first_nonempty_ids
    from irwm.detection.detrex_adapter import select_msda
    from irwm.detection.lightning import DetectorModule

    msda = select_msda(cfg.msda)
    L.seed_everything(cfg.seed, workers=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.project / cfg.run_name  # stable across SLURM requeues
    smoke = cfg.smoke.enabled
    policy = cfg.encoder.policy

    dm_kwargs = dict(modality=cfg.modality, policy=policy, batch_size=cfg.data.batch_size,
                     num_workers=cfg.data.num_workers, flip_p=cfg.data.flip_p)
    if smoke:
        ids = first_nonempty_ids(AlignedSplit.load(cfg.modality, "train"), cfg.smoke.n_frames, policy)
        dm_kwargs.update(flip_p=0.0, train_image_ids=ids, val_split="train", val_image_ids=ids)
    dm = AlignedDetectionDataModule(**dm_kwargs)
    dm.setup("fit")
    optim = _container(cfg.optim)
    if smoke:
        optim["lr_drop_epoch"] = None
    model = DetectorModule(detector=_container(cfg.detector), encoder=_container(cfg.encoder), optim=optim,
                           val_gt=dm.val_gt, class_names=list(ALIGNED_CLASS_NAMES),
                           label_to_category_id=dict(ALIGNED_LABEL_TO_CATEGORY_ID))
    model.metrics_dir = run_dir / "val_metrics"

    logger = make_wandb_logger(cfg, save_dir=run_dir)
    counts = {"n_train_frames": len(dm.train_set), "n_dropped_empty": dm.train_set.n_dropped_empty,
              "n_val_frames": len(dm.val_set)}
    logger.log_metrics({f"data/{k}": v for k, v in counts.items()}, step=0)

    trainer_kwargs = _container(cfg.trainer)
    recorder = ValRecorder()
    if smoke:
        callbacks = [ModelCheckpoint(dirpath=ckpt_dir, save_top_k=0, save_last=True), recorder]
        trainer_kwargs.update(max_steps=cfg.smoke.max_steps, max_epochs=-1, val_check_interval=cfg.smoke.eval_every,
                              check_val_every_n_epoch=None)
    else:
        best = ModelCheckpoint(dirpath=ckpt_dir, monitor="val/AP", mode="max", save_top_k=1, save_last=True,
                               filename="best-e{epoch:03d}", auto_insert_metric_name=False)
        callbacks = [best, recorder]
    trainer = L.Trainer(
        accelerator=cfg.cluster.accelerator,
        devices=cfg.cluster.devices,
        num_nodes=cfg.cluster.num_nodes,
        logger=logger,
        plugins=cluster_plugins(),
        callbacks=callbacks,
        default_root_dir=ckpt_dir,  # SLURM auto-requeue writes hpc_ckpt_*.ckpt here
        enable_progress_bar=not on_slurm(),
        **trainer_kwargs,
    )
    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt_path(ckpt_dir))
    fit_s = time.perf_counter() - t0
    if not trainer.is_global_zero:
        return

    gpu = model.gpu_info or {}
    common = {"name": cfg.name, "dataset": cfg.dataset, "modality": cfg.modality, "arch": model.arch,
              "encoder_kind": cfg.encoder.get("kind"), "policy": policy, "image_size": list(model.image_size),
              "msda": msda, **counts, "fit_wall_s": fit_s,
              "gpu": gpu, "gpu_hours": fit_s / 3600 * max(1, int(gpu.get("world_size", 1))),
              "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
              "data_root": str(aligned_root()), "git": git_info()}
    step_ms = model._step_ms
    common["median_step_ms"] = statistics.median(step_ms) if step_ms else None
    if smoke:
        result = {"smoke": True, "batch_size": cfg.data.batch_size, "max_steps": cfg.smoke.max_steps,
                  "global_step": trainer.global_step, "validations": recorder.validations,
                  "nonfinite_steps": int(model._nonfinite_steps), **common}
        out = run_dir / "smoke.json"
    else:
        m = re.search(r"best-e(\d+)\.ckpt$", best.best_model_path or "")
        complete = fit_complete(trainer)
        result = {"best_ckpt": best.best_model_path or None,
                  "best_val_AP": float(best.best_model_score) if best.best_model_score is not None else None,
                  "best_epoch": int(m.group(1)) if m else None, "fit_complete": complete,
                  "validations": recorder.validations, **common}
        export_error: Exception | None = None
        result["export"] = {"enabled": bool(cfg.export.enabled), "path": None, "error": None}
        if cfg.export.enabled and complete and best.best_model_path:
            try:
                result["export"]["path"] = str(export_best(cfg.name, best.best_model_path,
                                                           overwrite=cfg.export.overwrite))
            except Exception as e:  # recorded, then re-raised so afterok dependants do not run
                export_error = e
                result["export"]["error"] = f"{type(e).__name__}: {e}"
        elif cfg.export.enabled:
            result["export"]["error"] = ("skipped: no best checkpoint" if complete
                                         else "skipped: fit not complete (interrupted / requeued)")
        out = run_dir / "train_summary.json"
    out.write_text(json.dumps(result, indent=2))
    logger.log_metrics({"perf/gpu_hours": result["gpu_hours"]}, step=trainer.global_step)
    print(f"[train_aligned] {cfg.name} on {gpu.get('name')}: {out}", flush=True)
    if not smoke:
        print(f"[train_aligned] export: {result['export']}", flush=True)
        if export_error is not None:
            raise export_error
        # A requeue's partial run (fit not complete) exits cleanly: the requeued job exports later.
        if cfg.export.enabled and complete and result["export"]["path"] is None:
            sys.exit(f"[train_aligned] {cfg.name}: no frozen export ({result['export']['error']})")


if __name__ == "__main__":
    main()
