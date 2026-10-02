"""DINO-Foresight's PCA as a frozen, pure-torch module.

Reproduces ``Dino_f.pca_transform`` / ``pca_inverse_transform``
(third_party/DINO-Foresight/src/dino_f.py:387-398) exactly:

    transform:  z = ((x - mean) / std - pca_mean) @ components.T
    inverse:    x = (z @ components + pca_mean) * std + mean

``mean``/``std`` are the per-dimension standardisation stats fitted on Cityscapes train
tokens, and ``pca_mean``/``components`` are sklearn's ``PCA.mean_``/``components_`` (no
whitening). The file is produced from the released sklearn pickle by
``projects/dinof_audit/pca_export.py``, so nothing here needs scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

_KEYS = ("mean", "std", "pca_mean", "components")


class PCABasis(nn.Module):
    """Frozen standardise+PCA map, 3072-d DINOv2 features ↔ 1152-d coefficients."""

    def __init__(self, mean: Tensor, std: Tensor, pca_mean: Tensor, components: Tensor,
                 sha256: str = "") -> None:
        super().__init__()
        k, d = components.shape
        for name, t in (("mean", mean), ("std", std), ("pca_mean", pca_mean)):
            if t.shape != (d,):
                raise ValueError(f"{name} must have shape [{d}], got {list(t.shape)}")
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("std", std.float().clone())
        self.register_buffer("pca_mean", pca_mean.float().clone())
        self.register_buffer("components", components.float().clone())
        self.sha256 = sha256

    @classmethod
    def load(cls, path: str | Path) -> "PCABasis":
        """Load a ``pca_1152.pt`` export (dict with ``mean, std, pca_mean, components``)."""
        d = torch.load(path, map_location="cpu", weights_only=True)
        missing = [k for k in _KEYS if k not in d]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}")
        return cls(*(d[k] for k in _KEYS), sha256=str(d.get("source_sha256", "")))

    @property
    def n_components(self) -> int:
        return self.components.shape[0]

    @property
    def dim(self) -> int:
        return self.components.shape[1]

    def _check(self, t: Tensor, expected: int, what: str) -> None:
        if t.shape[-1] != expected:
            raise ValueError(f"{what}: expected last dim {expected}, got {t.shape[-1]}")

    def transform(self, x: Tensor) -> Tensor:
        """``[..., 3072]`` features → ``[..., 1152]`` PCA coefficients."""
        self._check(x, self.dim, "transform")
        return ((x - self.mean) / self.std - self.pca_mean) @ self.components.T

    def inverse(self, z: Tensor) -> Tensor:
        """``[..., 1152]`` PCA coefficients → ``[..., 3072]`` reconstructed features."""
        self._check(z, self.n_components, "inverse")
        return (z @ self.components + self.pca_mean) * self.std + self.mean

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_components={self.n_components}, sha256={self.sha256[:12]}"
