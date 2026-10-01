from __future__ import annotations

from pathlib import Path

import pytest

from irwm.utils.paths import ROOT_ENV_VARS


@pytest.fixture(autouse=True)
def _no_wandb_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never talk to W&B."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture
def irwm_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point every ``IRWM_*`` root at a fresh tmp dir. Returns ``{logical name: path}``."""
    roots = {}
    for name, var in ROOT_ENV_VARS.items():
        roots[name] = tmp_path / name
        monkeypatch.setenv(var, str(roots[name]))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    return roots
