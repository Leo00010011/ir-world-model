"""F3 video clip index: build the clip set over the FLIR test split and report its counts.

CPU only, no SLURM needed. Run where the data is (ctdesk236):

    uv run python projects/clip_index/build.py install=true   # also writes lib/irwm/data/clip_sets/
    uv run python projects/clip_index/build.py                # run dir only, e.g. to cmp with the committed file

Writes <run_dir>/<clip_set>.json, <run_dir>/clip_index_report.json and
<run_dir>/clip_index_report.md, and logs the per-pair table to W&B. With install=true the
clip set also goes to irwm.data.clips.clip_set_path(), which refuses to overwrite a file
with different content. Spec: spec/2026-10-14-video-clip-index (FR8).
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from irwm.data.clips import (
    build_clip_index,
    clip_set_path,
    file_sha256,
    render_markdown,
    summarise,
    write_clip_index,
)
from irwm.data.flir import FlirSplit, load_rgb_thermal_map, split_dir
from irwm.utils.paths import load_env
from irwm.utils.tracking import git_info, make_wandb_logger

# Must run before @hydra.main resolves hydra.run.dir, which reads ${oc.env:IRWM_OUTPUT_ROOT}.
load_env()

PAIR_COLUMNS = ["rgb_video_id", "thermal_video_id", "n_frames", "segments", "n_clips",
                "n_evaluable", "clip_id_first", "clip_id_last"]


@hydra.main(version_base="1.3", config_path="conf", config_name="build")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    root = Path(cfg.flir.root)
    rgb = FlirSplit.load("rgb", "test", root)
    thermal = FlirSplit.load("thermal", "test", root)
    rgb_to_thermal = load_rgb_thermal_map(root)
    inputs = {
        "rgb_coco_sha256": file_sha256(split_dir("rgb", "test", root) / "coco.json"),
        "thermal_coco_sha256": file_sha256(split_dir("thermal", "test", root) / "coco.json"),
        "rgb_to_thermal_map_sha256": file_sha256(root / "rgb_to_thermal_vid_map.json"),
    }
    doc = build_clip_index(
        rgb, thermal, rgb_to_thermal, frame_stride=cfg.frame_stride, n_context=cfg.n_context,
        n_future=cfg.n_future, clip_stride=cfg.clip_stride, inputs=inputs,
    )
    clip_set = doc["clip_set"]
    run_copy = run_dir / f"{clip_set}.json"
    write_clip_index(doc, run_copy)
    sha = file_sha256(run_copy)

    summary = summarise(doc)
    report = {
        **summary,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git": git_info(),
        "inputs": inputs,
        "clip_set_sha256": sha,
    }
    (run_dir / "clip_index_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "clip_index_report.md").write_text(render_markdown(summary), encoding="utf-8")

    import wandb

    logger = make_wandb_logger(cfg, save_dir=run_dir)
    pairs = wandb.Table(
        columns=PAIR_COLUMNS,
        data=[[json.dumps(p[c]) if c == "segments" else p[c] for c in PAIR_COLUMNS]
              for p in summary["pairs"]],
    )
    logger.experiment.log({"pairs": pairs, "n_clips": summary["n_clips"],
                           "n_evaluable": summary["n_evaluable"]})
    logger.experiment.finish()

    if cfg.install:
        path = clip_set_path(clip_set)
        written = write_clip_index(doc, path)
        print(f"installed: {path} ({'written' if written else 'unchanged, identical file exists'})")
    print(f"run dir: {run_dir}")
    print(f"clip set: {run_copy}")
    print(f"sha256: {sha}")


if __name__ == "__main__":
    main()
