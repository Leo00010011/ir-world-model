"""Export the best checkpoint of a finished training run as a frozen head (``det_rgb`` / ``det_ir``).

CPU only (the cluster login node is fine)::

    uv run python projects/detection/export.py experiment=det_rgb source_run=slurm-<job id>

Takes the single ``best-e*.ckpt`` in ``${paths.ckpt_root}/detection/<source_run>/`` and writes
``$IRWM_CKPT_ROOT/detection/frozen/<name>/{head.pt, meta.json}`` (``irwm.detection.frozen``).
Spec: spec/2026-10-26-detector-training-pipeline (FR9).
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from irwm.detection.frozen import export_frozen, frozen_meta
from irwm.utils.paths import load_env

load_env()


def pick_best_ckpt(run_ckpt_dir: Path) -> Path:
    """The one ``best-e*.ckpt`` in ``run_ckpt_dir``: none → ``FileNotFoundError``, several → ``ValueError``."""
    found = sorted(Path(run_ckpt_dir).glob("best-e*.ckpt"))
    if not found:
        raise FileNotFoundError(f"no best-e*.ckpt in {run_ckpt_dir}")
    if len(found) > 1:
        raise ValueError(f"several best-e*.ckpt in {run_ckpt_dir}: {[p.name for p in found]}")
    return found[0]


@hydra.main(version_base="1.3", config_path="conf", config_name="export")
def main(cfg: DictConfig) -> None:
    ckpt = pick_best_ckpt(Path(cfg.paths.ckpt_root) / cfg.project / cfg.source_run)
    out = export_frozen(cfg.name, ckpt, overwrite=cfg.overwrite)
    print(f"[export] {cfg.name}: {ckpt} -> {out} | head_sha256 {frozen_meta(cfg.name)['head_sha256']}", flush=True)


if __name__ == "__main__":
    main()
