"""Frozen end-to-end ViTDet heads (``aligned_mae_{rgb,ir}``, F18): export from a checkpoint, and loader.

Same layout and rules as ``irwm.detection.frozen`` (write-once, SHA-256 checked, ``eval()``, no grad)::

    $IRWM_CKPT_ROOT/detection/frozen/aligned_mae_<rgb|ir>/
    ├── head.pt     # the FULL ViTDetDINO state_dict (keys "detr.*": MAE-initialised ViT, pyramid, DINO head), fp32
    └── meta.json   # detector.arch "vitdet_mae", dataset, preprocess policy, image_size, GPU, val metrics, git, SHA-256

Unlike the frozen-encoder heads, the backbone was fine-tuned, so it is part of ``head.pt``
(TechStack §3.1b; F8's deviation E4 reversed for this variant only). Consumers feed it images
preprocessed with ``meta["encoder"]["policy"]`` (``native640``: 512×640, ImageNet norm, thermal 1→3)
and read its eval outputs exactly like a ``DINODetector``'s (xyxy px of the preprocessed input).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from irwm.detection.frozen import (
    FROZEN_DATASET,
    FROZEN_MODALITY,
    MAE_ARCH,
    PRECISION,
    _sha256,
    _val_metrics,
    frozen_dir,
    frozen_meta,
)
from irwm.features.preprocess import POLICY_SIZE
from irwm.utils.paths import ckpt_root
from irwm.utils.tracking import git_info

MAE_NAMES: tuple[str, ...] = ("aligned_mae_rgb", "aligned_mae_ir")
MAE_ENCODER_TAG = "mae_vitb16_512x640_vitdet-ft"  # default tag; a config's encoder.encoder_tag wins
_PREFIX = "detector."


def _check_name(name: str) -> None:
    if name not in MAE_NAMES:
        raise ValueError(f"{name!r} is not an end-to-end ViTDet head; expected one of {list(MAE_NAMES)}")


def export_frozen_vitdet(name: str, ckpt_path: Path, *, overwrite: bool = False,
                         dataset: str | None = None) -> Path:
    """Write ``head.pt`` (full model) + ``meta.json`` of ``name`` from a ViTDet ``DetectorModule`` checkpoint.

    Same ``head.pt`` bytes already there → no-op. Different bytes → ``FileExistsError`` unless ``overwrite``.
    """
    _check_name(name)
    expected = FROZEN_DATASET[name]
    dataset = expected if dataset is None else dataset
    if dataset != expected:
        raise ValueError(f"head {name!r} belongs to dataset {expected!r}, got dataset={dataset!r}")
    ckpt_path = Path(ckpt_path).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    detector = dict(hp["detector"])
    if detector.get("arch") != MAE_ARCH:
        raise ValueError(f"{ckpt_path}: detector.arch {detector.get('arch')!r} != {MAE_ARCH!r}; "
                         "export DINODetector heads with irwm.detection.frozen.export_frozen")
    state = {k[len(_PREFIX):]: v for k, v in ckpt["state_dict"].items() if k.startswith(_PREFIX)}
    if not state:
        raise ValueError(f"{ckpt_path} has no {_PREFIX!r} keys; is it a DetectorModule checkpoint?")

    out = frozen_dir(name)
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

    enc = dict(hp["encoder"])
    policy = enc["policy"]
    encoder = {"kind": "mae", "policy": policy, "encoder_tag": enc.get("encoder_tag") or MAE_ENCODER_TAG,
               "precision": enc.get("precision", "fp16")}
    val = _val_metrics(ckpt)
    irwm = ckpt.get("irwm") or {}
    meta = {
        "name": name,
        "modality": FROZEN_MODALITY[name],
        "dataset": dataset,
        "precision": PRECISION,
        "source_ckpt": ckpt_path.relative_to(ckpt_root()).as_posix(),
        "epoch": int(ckpt["epoch"]),
        "global_step": int(ckpt["global_step"]),
        "val_metrics": val,
        "detector": detector,
        "encoder": encoder,
        "preprocess_policy": policy,
        "image_size": list(POLICY_SIZE[policy]),
        "class_names": hp.get("class_names"),
        "label_to_category_id": hp.get("label_to_category_id"),
        "gpu": irwm.get("gpu") or (val or {}).get("gpu"),
        "head_sha256": sha,
        "n_params": int(sum(v.numel() for v in state.values() if torch.is_floating_point(v))),
        "git": git_info(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_frozen_vitdet(name: str, device: str | torch.device = "cpu") -> nn.Module:
    """The frozen ``ViTDetDINO`` of ``name``: SHA-256 checked, strict load, ``eval()``, no grad.

    Built with ``init: none`` (no MAE file needed): every weight comes from ``head.pt``.
    """
    _check_name(name)
    meta = frozen_meta(name)
    if meta["detector"].get("arch") != MAE_ARCH:
        raise ValueError(f"{name!r}: meta detector.arch {meta['detector'].get('arch')!r} != {MAE_ARCH!r}")
    head = frozen_dir(name) / "head.pt"
    sha = _sha256(head)
    if sha != meta["head_sha256"]:
        raise ValueError(f"{head}: SHA-256 {sha} != meta.json head_sha256 {meta['head_sha256']}")
    from irwm.detection.vitdet import vitdet_from_config

    cfg = {k: v for k, v in meta["detector"].items() if k != "arch"}
    cfg["init"] = "none"
    model = vitdet_from_config(OmegaConf.create(cfg))
    model.load_state_dict(torch.load(head, map_location="cpu"), strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)
