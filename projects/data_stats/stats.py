"""F2 dataset statistics: counts, sizes, sequences, file integrity and the imbalance study.

CPU only, no SLURM needed. Run where the data is (ctdesk236):

    uv run python projects/data_stats/stats.py
    uv run python projects/data_stats/stats.py check_files=false   # metadata only, faster

Writes <run_dir>/flir_stats.json and <run_dir>/flir_stats.md, and logs the selected-6 and
per-class tables to W&B. Sections follow spec/2026-10-08-flir-readers-class-mapping FR8.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from irwm.data.classes import CLASS_NAMES
from irwm.data.flir import FlirSplit
from irwm.data.stats import compute_stats, render_markdown
from irwm.utils.paths import load_env
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()


@hydra.main(version_base="1.3", config_path="conf", config_name="stats")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    root = Path(cfg.flir.root)
    splits = [FlirSplit.load(m, s, root) for m in cfg.modalities for s in cfg.splits]
    stats = compute_stats(splits, check_files=cfg.check_files)
    stats["config"] = OmegaConf.to_container(cfg, resolve=True)
    stats["git"] = git_info()
    (run_dir / "flir_stats.json").write_text(json.dumps(stats, indent=2))
    (run_dir / "flir_stats.md").write_text(render_markdown(stats), encoding="utf-8")

    import wandb

    logger = make_wandb_logger(cfg, save_dir=run_dir)
    keys = list(stats["selected"])
    selected = wandb.Table(
        columns=["class", *keys],
        data=[[c, *(stats["selected"][k]["kept"][c] for k in keys)] for c in CLASS_NAMES],
    )
    per_class = wandb.Table(
        columns=["class", "category_id", "modality", "split", "boxes", "images", "share_pct",
                 "selected", "coco80", "underrepresented", "absent_val", "absent_test"],
        data=[[c["name"], c["category_id"], c["modality"], c["split"], c["boxes"], c["images"],
               c["share_pct"], c["selected"], c["coco80"], c["underrepresented"],
               c["absent_val"], c["absent_test"]] for c in stats["classes"]],
    )
    logger.experiment.log({"selected_6": selected, "per_class": per_class})
    logger.experiment.finish()
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
