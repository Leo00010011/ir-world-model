"""Export DINO-Foresight's pickled sklearn PCA to plain tensors (``pca_1152.pt``).

Unpickling needs scikit-learn (pinned 1.5.2, as in their requirements.txt). Everything
downstream then uses ``irwm.features.pca.PCABasis`` without sklearn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from irwm.worldmodel.assets import sha256_file


def export_pca(pth_path: str | Path, out_path: str | Path) -> dict:
    """Write ``{mean, std, pca_mean, components}`` (fp32) plus metadata. Returns a summary.

    Raises ``ValueError`` if the PCA was fit with ``whiten=True``: the reference transform
    (``Dino_f.pca_transform``) ignores whitening, so it would silently disagree with sklearn.
    """
    d = torch.load(pth_path, map_location="cpu", weights_only=False)
    pca = d["pca_model"]
    if getattr(pca, "whiten", False):
        raise ValueError(f"{pth_path}: PCA has whiten=True, which Dino_f.pca_transform does not apply")
    arrays = {
        "mean": np.asarray(d["mean"]),
        "std": np.asarray(d["std"]),
        "pca_mean": np.asarray(pca.mean_),
        "components": np.asarray(pca.components_),
    }
    payload = {k: torch.from_numpy(v.astype(np.float32)) for k, v in arrays.items()}
    payload.update(
        whiten=bool(getattr(pca, "whiten", False)),
        n_components=int(pca.n_components_),
        explained_variance_ratio=torch.from_numpy(np.asarray(pca.explained_variance_ratio_, dtype=np.float32)),
        original_dtypes={k: str(v.dtype) for k, v in arrays.items()},
        sklearn_version=str(getattr(pca, "_sklearn_version", "unknown")),
        source_file=Path(pth_path).name,
        source_sha256=sha256_file(pth_path),
    )
    torch.save(payload, out_path)
    return {
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "original_dtypes": payload["original_dtypes"],
        "whiten": payload["whiten"],
        "explained_variance_total": float(payload["explained_variance_ratio"].sum()),
        "std_min": float(arrays["std"].min()),
        "pca_mean_abs_max": float(np.abs(arrays["pca_mean"]).max()),
        "sklearn_version": payload["sklearn_version"],
    }
