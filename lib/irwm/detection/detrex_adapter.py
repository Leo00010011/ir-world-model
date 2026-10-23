"""Access to detrex's DINO code and configs from the pinned submodule ``third_party/detrex``.

Spec: spec/2026-10-23-dino-detr-head-port (FR3). detrex @ ``DETREX_COMMIT`` with its nested
detectron2 @ ``DETECTRON2_COMMIT``, both unmodified and installed editable into the uv env.

detrex's DINO classes and configs live in ``projects/dino/...``, which its ``setup.py`` does not
package (``projects/`` has no ``__init__.py``), so the submodule root goes on ``sys.path``. It is
**appended**, not inserted, so the installed ``detrex``/``detectron2`` packages always win over any
same-named directory, and it stays there: the namespace package ``projects`` recomputes its path
when ``sys.path`` changes, and later ``LazyConfig.load`` calls need ``projects.dino.modeling``.
The import goes through ``importlib``/``LazyConfig``, never an ``import projects`` statement, so
the "lib never imports from projects" rule (TechStack §2) keeps meaning our own ``projects/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from detectron2.config import LazyConfig
from omegaconf import DictConfig

from irwm.utils.paths import repo_root

DETREX_COMMIT = "e244e6c3da3e84566728c52c21fb061d23ce0e2f"
DETECTRON2_COMMIT = "70f454304e1a38378200459dd2dbca0f0f4a5ab4"

DINO_VITDET_CONFIG = "projects/dino/configs/models/dino_vitdet.py"


def detrex_root() -> Path:
    """``<repo>/third_party/detrex``; raises if the submodule is not checked out."""
    root = repo_root() / "third_party" / "detrex"
    if not (root / "projects" / "dino" / "modeling" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"detrex submodule not found at {root}; run `git submodule update --init --recursive`")
    return root


def ensure_detrex_projects() -> None:
    """Append the detrex root to ``sys.path`` once, so ``projects.dino.modeling`` is importable."""
    root = str(detrex_root())
    if root not in sys.path:
        sys.path.append(root)


def load_dino_vitdet_config() -> DictConfig:
    """detrex's own ``dino_vitdet.py`` (which imports ``dino_r50.py``), loaded with ``LazyConfig``."""
    ensure_detrex_projects()
    return LazyConfig.load(str(detrex_root() / DINO_VITDET_CONFIG))


def cuda_ops_available() -> bool:
    """True iff CUDA is available and detrex's compiled ``MultiScaleDeformableAttention`` runs on it."""
    if not torch.cuda.is_available():
        return False
    try:
        from detrex.layers import MultiScaleDeformableAttention

        attn = MultiScaleDeformableAttention(embed_dim=256, num_heads=8, num_levels=1, num_points=4,
                                             batch_first=True).cuda()
        with torch.no_grad():
            attn(query=torch.randn(1, 4, 256, device="cuda"),
                 value=torch.randn(1, 16, 256, device="cuda"),
                 reference_points=torch.rand(1, 4, 1, 2, device="cuda"),
                 spatial_shapes=torch.tensor([[4, 4]], device="cuda"),
                 level_start_index=torch.tensor([0], device="cuda"))
        torch.cuda.synchronize()
        return True
    except Exception:
        return False
