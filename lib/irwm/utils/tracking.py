"""Run tracking: git provenance and the W&B logger.

Every result must trace back to a Hydra config, a git commit and a W&B run
(Mission → Desired behaviours).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from irwm.utils.paths import repo_root


def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
    )
    return out.stdout.strip()


def git_info(repo: str | Path | None = None) -> dict[str, str | bool]:
    """Return ``{"commit": <sha>, "dirty": <bool>}`` for ``repo`` (default: the repo root).

    Never raises. Outside a git repo, or without git, it returns
    ``{"commit": "unknown", "dirty": False}``.
    """
    if repo is None:
        try:
            repo = repo_root()
        except FileNotFoundError:
            repo = Path.cwd()
    repo = Path(repo)
    try:
        # --show-toplevel guards against picking up a parent repo's HEAD.
        top = Path(_git(["rev-parse", "--show-toplevel"], repo)).resolve()
        if top != repo.resolve():
            return {"commit": "unknown", "dirty": False}
        commit = _git(["rev-parse", "HEAD"], repo)
        dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], repo))
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": False}
    return {"commit": commit, "dirty": dirty}


def make_wandb_logger(cfg: DictConfig, save_dir: str | Path) -> WandbLogger:
    """Build a ``WandbLogger`` from ``cfg.wandb`` and log the resolved config plus git info.

    Expects ``cfg.wandb.{project,entity,group,mode,tags,log_model}`` and ``cfg.run_name``.
    """
    w = cfg.wandb
    logger = WandbLogger(
        project=w.project,
        entity=w.entity,
        group=w.group,
        name=cfg.run_name,
        tags=list(w.tags),
        mode=w.mode,
        save_dir=str(save_dir),
        log_model=w.log_model,
    )
    hparams: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    hparams["git"] = git_info()
    logger.log_hyperparams(hparams)
    return logger
