"""Train one KAIST-Paired detector (2026-11-27): ``kaist_dinopca_rgb`` (visible) or ``kaist_dinopca_ir`` (lwir).

KAIST frames (640×512) → ``native`` 448×560 → flip → frozen DINOv2-Reg + PCA (SDPA, fp16) → F7 head
(person, cyclist). Each head trains on its own camera's KAIST-Paired boxes; ``people``, ``person?``
and ``unpaired`` are ignore regions. Every epoch uses all positive frames plus a fresh random draw of
empty frames (``data.empty_ratio`` × positives, default 0.25). Splits (``irwm.data.kaist``): train = set00, set01, set03, set04,
val = set02 (day) + set05 (night), ``train-all-04`` frames. Recipe: ``conf/train_kaist.yaml``
(20 epochs, lr 1e-4, ×0.1 from epoch 15, validation every 2 epochs).

Each validation scores the val sets (``val/*``) and a fixed train subset (``train_eval/*``): AP, AP50,
per-class AP/AP50 and AP/AP50 on occluded pedestrians (``*_occ_{partial,heavy,any}``), then splits each
set by frame: ``*_nonempty/*`` (AP on frames with a box) and ``*_empty/*`` (FPPI and the fraction of
frames with a false positive at ``kaist.fp_thresholds``, on frames with no box at all)::

    bash slurm/train_kaist.sh ctcomp3 [--smoke] [--heads kaist_dinopca_ir]
    # one head by hand (what the launcher runs):
    IRWM_STAGE_DATA=1 IRWM_STAGE_DATASETS="kaist-cvpr15 kaist-paired" bash slurm/submit.sh ctcomp3 \\
        projects/detection/train_kaist.py kaist=kaist_dinopca_ir cluster.timeout_min=300 ...
    # locally:
    uv run python projects/detection/train_kaist.py kaist=kaist_dinopca_ir smoke.enabled=true

Writes ``<run_dir>/val_metrics/epoch_<NNN>.json`` per validation (``metrics`` = val, ``eval`` = every set),
then ``train_summary.json`` (or ``smoke.json``), with the GPU and GPU-hours. Checkpoints
(``best-eNNN.ckpt`` on ``val/AP``, ``last.ckpt``) in ``${paths.ckpt_root}/detection/<run_name>/``. After a
complete (non-smoke) fit, rank 0 exports the best checkpoint as ``$IRWM_CKPT_ROOT/detection/frozen/<name>/``.
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
    """Keeps every scored validation (all prefixes) for the run summary."""

    def __init__(self) -> None:
        self.validations: list[dict] = []

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                   if k.startswith(("val", "train_eval"))}
        self.validations.append({"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                                 "metrics": metrics})


def _container(node: DictConfig) -> dict:
    return OmegaConf.to_container(node, resolve=True)  # type: ignore[return-value]


def check_config(cfg: DictConfig) -> None:
    """Fail fast on an inconsistent KAIST config (before any data or weights are loaded)."""
    from irwm.data.kaist import KAIST_NUM_CLASSES
    from irwm.detection.frozen import FROZEN_DATASET, FROZEN_MODALITY
    from irwm.detection.lightning import detector_arch

    if cfg.dataset != "kaist":
        raise ValueError(f"train_kaist.py trains KAIST heads, got dataset={cfg.dataset!r}")
    if FROZEN_DATASET.get(cfg.name) != "kaist":
        raise ValueError(f"{cfg.name!r} is not a KAIST head name (kaist_<backbone>_<rgb|ir>)")
    if FROZEN_MODALITY[cfg.name] != cfg.modality:
        raise ValueError(f"{cfg.name}: modality {cfg.modality!r}, expected {FROZEN_MODALITY[cfg.name]!r}")
    if cfg.detector.num_classes != KAIST_NUM_CLASSES:
        raise ValueError(f"detector.num_classes {cfg.detector.num_classes} != {KAIST_NUM_CLASSES} (person, cyclist)")
    if detector_arch(cfg.detector) != "dino" or cfg.encoder.policy != "native":
        raise ValueError(f"{cfg.name}: KAIST heads use the frozen-encoder DINO head at policy 'native'")
    if not (cfg.data.empty_ratio == "all" or (isinstance(cfg.data.empty_ratio, (int, float))
                                               and cfg.data.empty_ratio >= 0)):
        raise ValueError(f"data.empty_ratio must be a number >= 0 or 'all', got {cfg.data.empty_ratio!r}")
    if int(cfg.kaist.val_stride) < 1:
        raise ValueError(f"kaist.val_stride must be >= 1, got {cfg.kaist.val_stride}")


def fit_complete(trainer: L.Trainer) -> bool:
    """True iff ``trainer.fit`` ran to ``max_epochs`` (not interrupted, not a requeue's partial run)."""
    from lightning.pytorch.trainer.states import TrainerStatus

    max_epochs = trainer.max_epochs if trainer.max_epochs is not None else -1
    return (trainer.state.status == TrainerStatus.FINISHED and not trainer.interrupted
            and max_epochs > 0 and trainer.current_epoch >= max_epochs)


@hydra.main(version_base="1.3", config_path="conf", config_name="train_kaist")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    if not torch.cuda.is_available():
        sys.exit("projects/detection/train_kaist.py: CUDA is not available; DINO training needs a GPU "
                 "(detrex hard-codes .cuda() in denoising).")
    # Heavy imports after the checks, so a misconfigured machine fails in seconds.
    from irwm.data.detection import KaistDetectionDataModule, first_nonempty_ids
    from irwm.data.kaist import KAIST_CLASS_NAMES, KAIST_LABEL_TO_CATEGORY_ID, KaistSplit, kaist_paired_root, kaist_root
    from irwm.detection.detrex_adapter import select_msda
    from irwm.detection.frozen import export_frozen
    from irwm.detection.lightning import DetectorModule

    msda = select_msda(cfg.msda)
    L.seed_everything(cfg.seed, workers=True)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_dir = Path(cfg.paths.ckpt_root) / cfg.project / cfg.run_name  # stable across SLURM requeues
    smoke = cfg.smoke.enabled
    policy = cfg.encoder.policy
    te_stride = cfg.kaist.train_eval_stride

    dm_kwargs = dict(modality=cfg.modality, policy=policy, batch_size=cfg.data.batch_size,
                     num_workers=cfg.data.num_workers, flip_p=cfg.data.flip_p,
                     val_stride=cfg.kaist.val_stride,
                     train_eval_stride=None if te_stride is None else int(te_stride),
                     empty_ratio=cfg.data.empty_ratio, seed=cfg.seed)
    if smoke:
        ids = first_nonempty_ids(KaistSplit.load(cfg.modality, "train"), cfg.smoke.n_frames, policy)
        dm_kwargs.update(flip_p=0.0, train_image_ids=ids, val_split="train", val_image_ids=ids)
    dm = KaistDetectionDataModule(**dm_kwargs)
    dm.setup("fit")
    optim = _container(cfg.optim)
    if smoke:
        optim["lr_drop_epoch"] = None
    model = DetectorModule(detector=_container(cfg.detector), encoder=_container(cfg.encoder), optim=optim,
                           val_gt=dm.val_gt, class_names=list(KAIST_CLASS_NAMES),
                           label_to_category_id=dict(KAIST_LABEL_TO_CATEGORY_ID),
                           val_variants=dm.val_variants, extra_eval=dm.extra_eval(),
                           val_subsets=dm.val_subsets, val_negatives=dm.val_negatives,
                           fp_thresholds=list(cfg.kaist.fp_thresholds))
    model.metrics_dir = run_dir / "val_metrics"

    logger = make_wandb_logger(cfg, save_dir=run_dir)
    counts = {"n_train_frames": dm.train_epoch_size, "n_train_positive": dm.n_positive,
              "n_train_empty_pool": dm.n_empty_pool, "n_train_empty_per_epoch": dm.n_empty_per_epoch,
              "n_train_ignore_only_dropped": dm.n_ignore_only, "n_val_frames": len(dm.val_set),
              "n_train_eval_frames": len(dm.train_eval_set) if dm.train_eval_set is not None else 0}
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
              "data_root": str(kaist_root()), "annotations_root": str(kaist_paired_root()), "git": git_info()}
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
                result["export"]["path"] = str(export_frozen(cfg.name, Path(best.best_model_path),
                                                             overwrite=cfg.export.overwrite, dataset="kaist"))
            except Exception as e:  # recorded, then re-raised so afterok dependants do not run
                export_error = e
                result["export"]["error"] = f"{type(e).__name__}: {e}"
        elif cfg.export.enabled:
            result["export"]["error"] = ("skipped: no best checkpoint" if complete
                                         else "skipped: fit not complete (interrupted / requeued)")
        out = run_dir / "train_summary.json"
    out.write_text(json.dumps(result, indent=2))
    logger.log_metrics({"perf/gpu_hours": result["gpu_hours"]}, step=trainer.global_step)
    print(f"[train_kaist] {cfg.name} on {gpu.get('name')}: {out}", flush=True)
    if not smoke:
        print(f"[train_kaist] export: {result['export']}", flush=True)
        if export_error is not None:
            raise export_error
        if cfg.export.enabled and complete and result["export"]["path"] is None:
            sys.exit(f"[train_kaist] {cfg.name}: no frozen export ({result['export']['error']})")


if __name__ == "__main__":
    main()
