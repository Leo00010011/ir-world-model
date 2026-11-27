"""Frozen DINO-DETR heads: export from a Lightning checkpoint, and loader.

Spec: spec/2026-10-26-detector-training-pipeline (FR6). Names: ``det_rgb`` / ``det_ir`` (FLIR v2)
and, for the aligned FLIR v1 backbone ablation (F18, Roadmap Phase 2b),
``aligned_{dinopca,dino3072,mae}_{rgb,ir}`` (``ir`` → modality ``thermal``), and for KAIST-Paired
(2026-11-27) ``kaist_dinopca_{rgb,ir}``. Layout::

    $IRWM_CKPT_ROOT/detection/frozen/<name>/
    ├── head.pt     # DINODetector state_dict (keys "detr.*"), fp32 master weights of 16-mixed training
    └── meta.json   # modality, dataset, precision "16-mixed", source checkpoint, val metrics, hyperparameters, SHA-256

``meta["dataset"]`` is ``"flir_v2"``, ``"aligned_v1"`` or ``"kaist"``; metas written before it existed are
FLIR v2 (``frozen_meta`` fills the default). ``aligned_mae_*`` heads (detrex's stock ViTDet,
``meta["detector"]["arch"] == "vitdet_mae"``) are not ``DINODetector``s and have their own loader.

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
from irwm.features.encoder import ENCODER_CHANNELS, encoder_tag
from irwm.features.preprocess import POLICY_SIZE
from irwm.utils.paths import ckpt_root
from irwm.utils.tracking import git_info

ALIGNED_BACKBONES: tuple[str, ...] = ("dinopca", "dino3072", "mae")
KAIST_BACKBONES: tuple[str, ...] = ("dinopca",)
_SUFFIX_MODALITY: dict[str, str] = {"rgb": "rgb", "ir": "thermal"}
FROZEN_NAMES: tuple[str, ...] = ("det_rgb", "det_ir") + tuple(
    f"aligned_{b}_{m}" for b in ALIGNED_BACKBONES for m in _SUFFIX_MODALITY) + tuple(
    f"kaist_{b}_{m}" for b in KAIST_BACKBONES for m in _SUFFIX_MODALITY)
FROZEN_MODALITY: dict[str, str] = {n: _SUFFIX_MODALITY[n.rsplit("_", 1)[1]] for n in FROZEN_NAMES}
_PREFIX_DATASET: dict[str, str] = {"aligned_": "aligned_v1", "kaist_": "kaist"}
FROZEN_DATASET: dict[str, str] = {
    n: next((d for p, d in _PREFIX_DATASET.items() if n.startswith(p)), "flir_v2") for n in FROZEN_NAMES}
DATASETS: tuple[str, ...] = ("flir_v2", "aligned_v1", "kaist")
PRECISION = "16-mixed"
MAE_ARCH = "vitdet_mae"
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


def export_frozen(name: str, ckpt_path: Path, *, overwrite: bool = False, dataset: str | None = None) -> Path:
    """Write ``head.pt`` + ``meta.json`` of ``name`` from a ``DetectorModule`` checkpoint; returns the dir.

    Same ``head.pt`` bytes already there → no-op. Different bytes → ``FileExistsError`` unless ``overwrite``.
    ``dataset`` (``"flir_v2"`` | ``"aligned_v1"``) defaults to the one the name implies and must match it.
    """
    out = frozen_dir(name)
    expected = FROZEN_DATASET[name]
    dataset = expected if dataset is None else dataset
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {list(DATASETS)}")
    if dataset != expected:
        raise ValueError(f"head {name!r} belongs to dataset {expected!r}, got dataset={dataset!r}")
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
    kind = encoder.get("kind", "dinopca")
    if "encoder_tag" not in encoder or kind in ENCODER_CHANNELS:  # non-DINO backbones bring their own tag
        encoder["encoder_tag"] = encoder_tag(encoder["policy"], kind)
    meta = {
        "name": name,
        "modality": FROZEN_MODALITY[name],
        "dataset": dataset,
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
    """``meta.json`` of a frozen detector (``dataset`` defaults to ``"flir_v2"`` for older metas)."""
    meta = json.loads((frozen_dir(name) / "meta.json").read_text())
    meta.setdefault("dataset", "flir_v2")
    return meta


def load_frozen_detector(name: str, *, device: str | torch.device = "cpu") -> DINODetector:
    """The frozen ``DINODetector`` of ``name``: SHA-256 checked, strict load, ``eval()``, no grad."""
    meta = frozen_meta(name)
    arch = meta["detector"].get("arch")
    if arch == MAE_ARCH:
        raise ValueError(f"{name!r} is a {MAE_ARCH!r} head (detrex's stock ViTDet), not a DINODetector; "
                         "load_frozen_detector cannot build it, use the MAE loader")
    if arch is not None:
        raise ValueError(f"{name!r}: unknown detector arch {arch!r} in meta.json")
    head = frozen_dir(name) / "head.pt"
    sha = _sha256(head)
    if sha != meta["head_sha256"]:
        raise ValueError(f"{head}: SHA-256 {sha} != meta.json head_sha256 {meta['head_sha256']}")
    det = DINODetector(**meta["detector"])
    det.load_state_dict(torch.load(head, map_location="cpu"), strict=True)
    det.eval().requires_grad_(False)
    return det.to(device)
