"""F19: score the aligned FLIR v1 heads on both modalities of the aligned test split (Result Table B).

Each test split is decoded once per batch; the ``dinopca`` heads share one DINOv2+PCA encoding, the
``dino3072`` heads another, the fine-tuned MAE ViTDet heads read the 512×640 images
(``irwm.eval.aligned_eval``). Any subset of the six heads runs (``heads=[...]``). No cross-dataset cells
(Roadmap R12). Launch with ``slurm/eval_aligned.sh`` (stages ``aligned/align``):

    bash slurm/eval_aligned.sh ctcomp3
    bash slurm/eval_aligned.sh ctcomp3 'heads=[aligned_mae_rgb,aligned_mae_ir]'

``mode=merge`` rebuilds one ``eval_aligned.json`` + ``table_b.md`` from the saved detection files of
``heads`` (re-scored on CPU, no model loading), to combine subset runs; it runs locally or on a login node:

    uv run python projects/detection/eval_aligned.py mode=merge

Writes the detection files (``irwm.detection.inference.detections_path``, write-once),
``<run_dir>/eval_aligned.json`` (cells, transfer ratios, provenance, GPU, git, config) and
``<run_dir>/table_b.md`` (``irwm.eval.aligned_table.render_table_b_markdown``), and logs to W&B.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from irwm.utils.paths import load_env
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()

TABLE_KEYS = ("AP_person_car", "AP50_person_car", "AP_person", "AP50_person", "AP_car", "AP50_car",
              "AP", "AP50", "APs", "APm", "APl")


def _device_info(device: str) -> dict:
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    return {"name": torch.cuda.get_device_name() if cuda else "cpu",
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_node": os.environ.get("SLURMD_NODENAME")}


@hydra.main(version_base="1.3", config_path="conf", config_name="eval_aligned")
def main(cfg: DictConfig) -> None:
    if cfg.mode not in ("run", "merge"):
        sys.exit(f"projects/detection/eval_aligned.py: mode must be 'run' or 'merge', got {cfg.mode!r}")
    if cfg.mode == "merge":
        from irwm.eval.aligned_eval import merge_aligned

        result = merge_aligned(list(cfg.heads), test_modalities=list(cfg.test_modalities), split=cfg.split)
        _finish(cfg, result, msda=None)
        return

    from irwm.detection.detrex_adapter import select_msda

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("projects/detection/eval_aligned.py: device=cuda but CUDA is not available")
    try:
        msda = select_msda(cfg.msda)
    except RuntimeError as e:
        sys.exit(f"projects/detection/eval_aligned.py: {e}; cuda_ops_available() is False")
    # Heavy imports after the checks, so a misconfigured machine fails in seconds.
    from irwm.eval.aligned_eval import evaluate_aligned

    result = evaluate_aligned(list(cfg.heads), test_modalities=list(cfg.test_modalities), split=cfg.split,
                              batch_size=cfg.batch_size, num_workers=cfg.num_workers, device=cfg.device,
                              overwrite=cfg.overwrite, log_every=cfg.log_every, verify_rescore=cfg.verify_rescore,
                              max_images=cfg.max_images)
    _finish(cfg, result, msda=msda)


def _finish(cfg: DictConfig, result: dict, *, msda: str | None) -> None:
    """Provenance, ``eval_aligned.json``, ``table_b.md``, W&B; exit non-zero if re-scoring disagreed."""
    import wandb

    from irwm.data.aligned import aligned_root
    from irwm.eval.aligned_table import render_table_b_markdown

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    result.setdefault("mode", "run")
    result.update({
        "dataset": "aligned_v1",
        "msda": msda,
        "device": _device_info(cfg.device if cfg.mode == "run" else "cpu"),
        "data_root": str(aligned_root()),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git": git_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    out = run_dir / "eval_aligned.json"
    out.write_text(json.dumps(result, indent=2))
    table = render_table_b_markdown(result)
    (run_dir / "table_b.md").write_text(table)

    logger = make_wandb_logger(cfg, save_dir=run_dir)
    cells = result["cells"]
    scalars = {f"test/{h}/{m}/{k}": v for h in cells for m, c in cells[h].items() for k, v in c.items()}
    scalars.update({f"transfer/{h}/{m}/{k}": v for h, per in result["transfer_ratio"].items()
                    for m, r in per.items() for k, v in r.items()})
    logger.log_metrics(scalars, step=0)
    rows = [[result["heads"][h]["backbone"], result["heads"][h]["train_modality"], m,
             *(cells[h][m][k] for k in TABLE_KEYS)] for h in cells for m in cells[h]]
    logger.experiment.log({"eval_aligned/table_b": wandb.Table(
        columns=["backbone", "train_modality", "test_modality", *TABLE_KEYS], data=rows)})
    logger.experiment.summary["table_b_markdown"] = table
    wandb.finish()

    print(table)
    print(f"[eval_aligned] {result['mode']}: {result['device']['name']}, {result['timing']['wall_s']:.0f} s -> {out}",
          flush=True)
    bad = {h: m for h, per in (result.get("rescore_identical") or {}).items() for m, ok in per.items() if not ok}
    if result.get("missing"):
        print(f"[eval_aligned] merge: missing detection files: {result['missing']}", flush=True)
    if bad:
        sys.exit(f"[eval_aligned] re-scoring the saved detections did not reproduce the metrics: {bad}")


if __name__ == "__main__":
    main()
