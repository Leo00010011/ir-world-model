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

Without detrex's compiled CUDA ops (no ``nvcc`` at build time, or no ``_C`` at all), the model
still builds and runs: ``select_msda("auto")`` routes every ``MultiScaleDeformableAttention`` to
detrex's pure-PyTorch fallback (same outputs, F7: 1.5-2.8× slower, ≈ 1.6× the memory).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import torch
from detectron2.config import LazyConfig
from omegaconf import DictConfig

from irwm.utils.paths import repo_root

DETREX_COMMIT = "e244e6c3da3e84566728c52c21fb061d23ce0e2f"
DETECTRON2_COMMIT = "70f454304e1a38378200459dd2dbca0f0f4a5ab4"

DINO_VITDET_CONFIG = "projects/dino/configs/models/dino_vitdet.py"
MSDA_MODES = ("auto", "cuda", "pytorch")


def _register_detrex_ops() -> None:
    """Make sure ``detrex._C`` imports before detrex does, with an empty stand-in if it cannot load.

    When ``from detrex import _C`` fails, detrex replaces ``MultiScaleDeformableAttention`` by a
    dummy that raises on construction (``detrex/layers/multi_scale_deform_attn.py:419-427``), so no
    model can be built. An empty module keeps the real class; its CUDA call then fails, which
    ``cuda_ops_available`` reports as ``False`` and ``use_pytorch_msda`` routes around.
    """
    if "detrex" in sys.modules or "detrex._C" in sys.modules:
        return
    spec = importlib.util.find_spec("detrex")  # top-level lookup only, detrex/__init__ does not run
    if spec is None or not spec.submodule_search_locations:
        return
    for directory in spec.submodule_search_locations:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = Path(directory) / f"_C{suffix}"
            if not path.is_file():
                continue
            try:
                ops_spec = importlib.util.spec_from_file_location("detrex._C", path)
                module = importlib.util.module_from_spec(ops_spec)
                ops_spec.loader.exec_module(module)
            except (ImportError, OSError):
                continue
            sys.modules["detrex._C"] = module
            return
    sys.modules["detrex._C"] = types.ModuleType("detrex._C")


_register_detrex_ops()


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
        import detrex  # noqa: F401  (runs detrex's own `from detrex import _C`)
        from detrex import _C

        # The compiled op itself, not the layer: after use_pytorch_msda the layer runs on CUDA anyway.
        # value [B, keys, heads, dim], sampling_locations [B, queries, heads, levels, points, 2].
        cuda = dict(device="cuda")
        _C.ms_deform_attn_forward(torch.randn(1, 16, 8, 32, **cuda),
                                  torch.tensor([[4, 4]], **cuda), torch.tensor([0], **cuda),
                                  torch.rand(1, 4, 8, 1, 4, 2, **cuda), torch.rand(1, 4, 8, 1, 4, **cuda), 64)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def use_pytorch_msda() -> None:
    """Route detrex's ``MultiScaleDeformableAttention`` to its pure-PyTorch path on CUDA too.

    Its forward looks ``MultiScaleDeformableAttnFunction`` up as a module global at call time
    (``detrex/layers/multi_scale_deform_attn.py``), so replacing that name switches every layer
    without touching the submodule. The CPU tests already pin both paths to the same output.
    """
    msda = importlib.import_module("detrex.layers.multi_scale_deform_attn")
    fallback = msda.multi_scale_deformable_attn_pytorch

    class _PyTorchMSDA:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return fallback(value, spatial_shapes, sampling_locations, attention_weights)

    msda.MultiScaleDeformableAttnFunction = _PyTorchMSDA


def select_msda(mode: str = "auto") -> str:
    """Pick the deformable-attention implementation; returns the one in use (``cuda`` or ``pytorch``).

    ``auto``: the compiled op if ``cuda_ops_available()``, else the PyTorch fallback. ``cuda``
    raises ``RuntimeError`` when the compiled op cannot run. Call it before the first forward.
    """
    if mode not in MSDA_MODES:
        raise ValueError(f"msda must be one of {list(MSDA_MODES)}, got {mode!r}")
    if mode == "pytorch" or (mode == "auto" and not cuda_ops_available()):
        use_pytorch_msda()
        return "pytorch"
    if mode == "cuda" and not cuda_ops_available():
        raise RuntimeError("msda='cuda' but detrex's compiled CUDA ops are not available "
                           "(rebuild detrex with nvcc on PATH, or use msda='auto')")
    return "cuda"
