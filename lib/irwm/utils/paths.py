"""Filesystem path resolution.

Every path the code touches comes from one of the ``IRWM_*`` environment
variables (TechStack §6). Nothing is hard-coded. On the laptop the variables
come from a git-ignored ``.env`` at the repo root (see ``.env.example``). On the
cluster they come from the shell or the sbatch script, and those always win over
``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_ENV_VARS: dict[str, str] = {
    "data": "IRWM_DATA_ROOT",
    "cache": "IRWM_CACHE_ROOT",
    "ckpt": "IRWM_CKPT_ROOT",
    "output": "IRWM_OUTPUT_ROOT",
}


class MissingEnvVarError(KeyError):
    """A required ``IRWM_*`` environment variable is unset or empty."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else ""


def repo_root() -> Path:
    """Return the repository root, i.e. the nearest ancestor containing ``uv.lock``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "uv.lock").is_file():
            return parent
    raise FileNotFoundError(f"No uv.lock found above {Path(__file__).resolve()}")


def load_env(dotenv_path: str | Path | None = None) -> bool:
    """Load ``.env`` (default: ``<repo_root>/.env``) without overriding existing variables.

    Returns ``True`` if a file was loaded and ``False`` if it does not exist.
    """
    if dotenv_path is None:
        try:
            path = repo_root() / ".env"
        except FileNotFoundError:
            return False
    else:
        path = Path(dotenv_path)
    if not path.is_file():
        return False
    load_dotenv(path, override=False)
    return True


def get_root(name: str) -> Path:
    """Resolve a root directory by logical name (``data``, ``cache``, ``ckpt``, ``output``).

    The returned path is absolute and ``~``-expanded. It is not required to exist.
    Raises ``ValueError`` for an unknown name, and ``MissingEnvVarError`` if the
    variable is unset or blank.
    """
    if name not in ROOT_ENV_VARS:
        raise ValueError(f"Unknown root {name!r}; expected one of {sorted(ROOT_ENV_VARS)}")
    var = ROOT_ENV_VARS[name]
    value = os.environ.get(var, "").strip()
    if not value:
        raise MissingEnvVarError(
            f"{var} is not set. Define it in the shell or in <repo>/.env (see .env.example)."
        )
    return Path(value).expanduser().resolve()


def data_root() -> Path:
    """``$IRWM_DATA_ROOT``: raw datasets (read-only)."""
    return get_root("data")


def cache_root() -> Path:
    """``$IRWM_CACHE_ROOT``: feature and prediction caches."""
    return get_root("cache")


def ckpt_root() -> Path:
    """``$IRWM_CKPT_ROOT``: model checkpoints and downloaded weights."""
    return get_root("ckpt")


def output_root() -> Path:
    """``$IRWM_OUTPUT_ROOT``: Hydra run dirs, eval JSONs, figures."""
    return get_root("output")
