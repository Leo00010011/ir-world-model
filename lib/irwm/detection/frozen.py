"""Frozen DINO-DETR heads ``det_rgb`` / ``det_ir``: export from a Lightning checkpoint, and loader.

Spec: spec/2026-10-26-detector-training-pipeline (FR6). Layout::

    $IRWM_CKPT_ROOT/detection/frozen/<name>/
    ├── head.pt     # DINODetector state_dict (keys "detr.*"), fp32 master weights of 16-mixed training
    └── meta.json   # modality, precision "16-mixed", source checkpoint, val metrics, hyperparameters, SHA-256

``head.pt`` stays fp32: casting it to fp16 would export a head that differs from the validated one.
Every consumer (F9, F12) runs it under CUDA fp16 autocast and encodes with the ``meta["encoder"]``
settings (SDPA attention). A head is trained once and then never changed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from irwm.detection.detector import DINODetector
from irwm.features.encoder import encoder_tag
from irwm.features.preprocess import POLICY_SIZE
from irwm.utils.paths import ckpt_root
from irwm.utils.tracking import git_info

FROZEN_NAMES: tuple[str, ...] = ("det_rgb", "det_ir")
FROZEN_MODALITY: dict[str, str] = {"det_rgb": "rgb", "det_ir": "thermal"}
PRECISION = "16-mixed"
_PREFIX = "detector."


def frozen_dir(name: str) -> Path:
    """``$IRWM_CKPT_ROOT/detection/frozen/<name>``; ``name`` must be one of ``FROZEN_NAMES``."""
    if name not in FROZEN_NAMES:
        raise ValueError(f"Unknown frozen detector {name!r}; expected one of {list(FROZEN_NAMES)}")
    return ckpt_root() / "detection" / "frozen" / name


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _val_metrics(ckpt: dict) -> dict | None:
    """The ``val_metrics/epoch_<NNN>.json`` written in the validation that saved ``ckpt``, if found."""
    metrics_dir = (ckpt.get("irwm") or {}).get("metrics_dir")
    if not metrics_dir:
        return None
    path = Path(metrics_dir) / f"epoch_{int(ckpt['epoch']):03d}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def export_frozen(name: str, ckpt_path: Path, *, overwrite: bool = False) -> Path:
    """Write ``head.pt`` + ``meta.json`` of ``name`` from a ``DetectorModule`` checkpoint; returns the dir.

    Same ``head.pt`` bytes already there → no-op. Different bytes → ``FileExistsError`` unless ``overwrite``.
    """
    out = frozen_dir(name)
    ckpt_path = Path(ckpt_path).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = {k[len(_PREFIX):]: v for k, v in ckpt["state_dict"].items() if k.startswith(_PREFIX)}
    if not state:
        raise ValueError(f"{ckpt_path} has no {_PREFIX!r} keys; is it a DetectorModule checkpoint?")

    out.mkdir(parents=True, exist_ok=True)
    head, tmp = out / "head.pt", out / "head.pt.tmp"
    torch.save(state, tmp)
    sha = _sha256(tmp)
    if head.is_file():
        if _sha256(head) == sha:
            tmp.unlink()
            return out
        if not overwrite:
            tmp.unlink()
            raise FileExistsError(f"{head} exists with a different SHA-256; pass overwrite=True to replace it")
    tmp.replace(head)

    hp = ckpt["hyper_parameters"]
    encoder = dict(hp["encoder"])
    encoder["encoder_tag"] = encoder_tag(encoder["policy"])
    meta = {
        "name": name,
        "modality": FROZEN_MODALITY[name],
        "precision": PRECISION,
        "source_ckpt": ckpt_path.relative_to(ckpt_root()).as_posix(),
        "epoch": int(ckpt["epoch"]),
        "global_step": int(ckpt["global_step"]),
        "val_metrics": _val_metrics(ckpt),
        "detector": dict(hp["detector"]),
        "encoder": encoder,
        "image_size": list(POLICY_SIZE[encoder["policy"]]),
        "head_sha256": sha,
        "git": git_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def frozen_meta(name: str) -> dict:
    """``meta.json`` of a frozen detector."""
    return json.loads((frozen_dir(name) / "meta.json").read_text())


def load_frozen_detector(name: str, *, device: str | torch.device = "cpu") -> DINODetector:
    """The frozen ``DINODetector`` of ``name``: SHA-256 checked, strict load, ``eval()``, no grad."""
    meta = frozen_meta(name)
    head = frozen_dir(name) / "head.pt"
    sha = _sha256(head)
    if sha != meta["head_sha256"]:
        raise ValueError(f"{head}: SHA-256 {sha} != meta.json head_sha256 {meta['head_sha256']}")
    det = DINODetector(**meta["detector"])
    det.load_state_dict(torch.load(head, map_location="cpu"), strict=True)
    det.eval().requires_grad_(False)
    return det.to(device)
