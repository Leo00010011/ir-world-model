"""Pinned external artefacts for DINO-Foresight and its DINOv2 backbone.

Everything is downloaded once on a login node (``projects/dinof_audit/download.py``)
into ``$IRWM_CKPT_ROOT`` and verified against SHA-256 manifests, so GPU jobs never
touch the network. Layout::

    $IRWM_CKPT_ROOT/dino_foresight/<hf_rev[:8]>/    HF files, pca_1152.pt, manifest.json
    $IRWM_CKPT_ROOT/torch_hub/                      torch.hub dir (torch.hub.set_dir)
        dinov2@<sha[:8]>/                           DINOv2 hub code at a pinned commit
        facebookresearch_dinov2_main/               copy of the pinned commit; the cache
                                                    dir torch.hub.load('facebookresearch/dinov2')
                                                    resolves to, so the reference code runs offline
        checkpoints/dinov2_vitb14_reg4_pretrain.pth
        manifest.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from irwm.utils.paths import ckpt_root

DINOF_HF_REPO = "Sta8is/DINO-Foresight"
DINOF_HF_REVISION = "9174a31e86c0ccd4df68c8c51f92ccce3313657a"
DINOF_HIGHRES = "dinof_highres.ckpt"
DINOF_LOWRES = "dinof_lowres.ckpt"
DINOF_PCA = "dinov2_pca_448_l[2_5_8_11]_1152.pth"
DINOF_HEADS = ("head_segm_pca1152.ckpt", "head_depth_pca1152.ckpt", "head_normals_pca1152.ckpt")
DINOF_FILES: tuple[str, ...] = (DINOF_HIGHRES, DINOF_LOWRES, DINOF_PCA, *DINOF_HEADS)
DINOF_META_FILES: tuple[str, ...] = ("README.md", "config.json")
PCA_EXPORT = "pca_1152.pt"

DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_HUB_COMMITS: dict[str, str] = {
    # Parent of e1277af ("[bug] Fix interpolation of positional embeddings (#378)", 2024-02-22).
    "pre378": "2302b6bf46953431b969155307b9bed152754069",
    # Last commit before DINO-Foresight's public release (2024-12-17).
    "dinof_release": "85a24602099d397264d5b30461ad7f3bfd726ca1",
    # main as of the audit (2026-06-03).
    "main_2026": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
}
# Selected by the F1 audit (FR7.5): the hub code under which the reference model predicts best.
DINOV2_HUB_PINNED = "dinof_release"
DINOV2_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
DINOV2_HUB_CACHE_NAME = "facebookresearch_dinov2_main"

MANIFEST = "manifest.json"
_CHUNK = 8 * 1024 * 1024


def dinof_dir() -> Path:
    """``$IRWM_CKPT_ROOT/dino_foresight/<hf revision[:8]>``."""
    return ckpt_root() / "dino_foresight" / DINOF_HF_REVISION[:8]


def dinov2_hub_dir() -> Path:
    """torch.hub directory holding the pinned DINOv2 code and weights."""
    return ckpt_root() / "torch_hub"


def dinov2_code_dir(name: str = DINOV2_HUB_PINNED) -> Path:
    """Extracted DINOv2 hub code for ``DINOV2_HUB_COMMITS[name]`` (contains ``hubconf.py``)."""
    if name not in DINOV2_HUB_COMMITS:
        raise KeyError(f"Unknown DINOv2 hub version {name!r}; expected one of {sorted(DINOV2_HUB_COMMITS)}")
    return dinov2_hub_dir() / f"dinov2@{DINOV2_HUB_COMMITS[name][:8]}"


def dinov2_weights_path() -> Path:
    """Where torch.hub caches the DINOv2-Reg ViT-B/14 weights."""
    return dinov2_hub_dir() / "checkpoints" / Path(DINOV2_WEIGHTS_URL).name


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory: str | Path, entries: list[dict]) -> Path:
    """Write ``manifest.json`` for files in ``directory``.

    Each entry needs ``file`` (path relative to ``directory``) and may carry ``source`` and
    ``revision``. ``bytes`` and ``sha256`` are computed here.
    """
    directory = Path(directory)
    rows = []
    for e in entries:
        p = directory / e["file"]
        rows.append({**e, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    out = directory / MANIFEST
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), "files": rows}
    out.write_text(json.dumps(payload, indent=2))
    return out


def read_manifest(directory: str | Path) -> dict[str, dict]:
    """``{file: entry}`` from ``directory/manifest.json`` (empty if there is none)."""
    path = Path(directory) / MANIFEST
    if not path.is_file():
        return {}
    return {row["file"]: row for row in json.loads(path.read_text())["files"]}


def verify_manifest(directory: str | Path) -> list[str]:
    """Files in the manifest that are missing or whose size/sha256 differ. Empty means OK.

    A missing manifest is reported as ``[MANIFEST]``.
    """
    directory = Path(directory)
    rows = read_manifest(directory)
    if not rows:
        return [MANIFEST]
    bad = []
    for name, row in rows.items():
        p = directory / name
        if not p.is_file() or p.stat().st_size != row["bytes"] or sha256_file(p) != row["sha256"]:
            bad.append(name)
    return bad
