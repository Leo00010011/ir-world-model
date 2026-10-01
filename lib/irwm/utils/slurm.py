"""SLURM helpers: detection, Lightning cluster plugins, resume, and sbatch flags from Hydra.

CLI (used by ``slurm/submit.sh``): print the sbatch flags for a cluster site config::

    python -m irwm.utils.slurm <site> [cluster.timeout_min=600 ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import DictConfig


def on_slurm() -> bool:
    """True iff running inside a SLURM allocation (``SLURM_JOB_ID`` is set)."""
    return bool(os.environ.get("SLURM_JOB_ID"))


def cluster_plugins() -> list:
    """Lightning plugins for the current environment.

    On SLURM: ``SLURMEnvironment(auto_requeue=True)``, so that on SIGUSR1 the job
    saves a checkpoint and requeues itself (paired with ``--signal=SIGUSR1@90`` in
    ``slurm/job.sbatch``). Elsewhere: no plugins.
    """
    if not on_slurm():
        return []
    from lightning.pytorch.plugins.environments import SLURMEnvironment  # heavy import

    return [SLURMEnvironment(auto_requeue=True)]


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


def sbatch_args(cluster: DictConfig) -> list[str]:
    """Translate a ``cluster`` config node into sbatch flags. Null/empty fields are omitted.

    This keeps the Hydra cluster config the single source of truth for resources:
    ``slurm/submit.sh`` passes these flags to ``sbatch slurm/job.sbatch``.
    """
    flags = {
        "partition": cluster.get("partition"),
        "account": cluster.get("account"),
        "qos": cluster.get("qos"),
        "constraint": cluster.get("constraint"),
        "gres": cluster.get("gres"),
        "nodes": cluster.get("num_nodes"),
        "ntasks-per-node": cluster.get("devices"),
        "cpus-per-task": cluster.get("cpus_per_task"),
        "mem": f"{cluster.mem_gb}G" if cluster.get("mem_gb") else None,
        "time": cluster.get("timeout_min"),
    }
    out = [f"--{k}={v}" for k, v in flags.items() if v not in (None, "")]
    out += [str(a) for a in cluster.get("extra_sbatch_args") or []]
    return out


def compose_cluster(site: str, overrides: list[str] = ()) -> DictConfig:
    """Compose ``irwm_base`` with ``cluster=<site>`` (plus ``cluster.*`` overrides); return ``cfg.cluster``."""
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="irwm.conf", version_base="1.3"):
        cfg = compose("irwm_base", overrides=[f"cluster={site}", *overrides])
    return cfg.cluster


def _main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print("usage: python -m irwm.utils.slurm <site> [cluster.key=value ...]", file=sys.stderr)
        return 2
    site, rest = argv[0], argv[1:]
    # Only cluster.* overrides affect resources; anything else is for the entry point itself.
    cluster_overrides = [a for a in rest if a.lstrip("+~").startswith("cluster.")]
    cluster = compose_cluster(site, cluster_overrides)
    if "TODO" in " ".join(str(v) for v in cluster.values()):
        print(f"cluster/{site}.yaml still has TODO values", file=sys.stderr)
        return 1
    print(" ".join(sbatch_args(cluster)))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
