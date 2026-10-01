"""SLURM helpers: detection and Lightning cluster plugins for preemption-safe jobs."""

from __future__ import annotations

import os
from pathlib import Path

from lightning.pytorch.plugins.environments import SLURMEnvironment


def on_slurm() -> bool:
    """True iff running inside a SLURM allocation (``SLURM_JOB_ID`` is set)."""
    return bool(os.environ.get("SLURM_JOB_ID"))


def cluster_plugins() -> list:
    """Lightning plugins for the current environment.

    On SLURM: ``SLURMEnvironment(auto_requeue=True)``, so that on SIGUSR1 the job
    saves a checkpoint and requeues itself (paired with ``--signal=SIGUSR1@90`` in
    ``slurm/job.sbatch``). Elsewhere: no plugins.
    """
    return [SLURMEnvironment(auto_requeue=True)] if on_slurm() else []


def resume_ckpt_path(ckpt_dir: str | Path) -> str | None:
    """Checkpoint to resume from in ``ckpt_dir``, or ``None`` for a fresh start.

    Prefers the newest ``hpc_ckpt_<n>.ckpt``, which Lightning writes on SIGUSR1 just
    before requeue and so is newer than ``last.ckpt``. Falls back to ``last.ckpt``.
    Requires the trainer's ``default_root_dir`` to be ``ckpt_dir`` (the place
    Lightning writes hpc checkpoints).
    """
    ckpt_dir = Path(ckpt_dir)
    hpc = []
    for p in ckpt_dir.glob("hpc_ckpt_*.ckpt"):
        suffix = p.stem.removeprefix("hpc_ckpt_")
        if suffix.isdigit():
            hpc.append((int(suffix), p))
    if hpc:
        return str(max(hpc)[1])
    last = ckpt_dir / "last.ckpt"
    return str(last) if last.is_file() else None
