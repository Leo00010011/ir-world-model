"""Frozen DINO-DETR heads on a whole FLIR split: encode once, run every head, save the detections.

Spec: spec/2026-10-28-cross-modal-detector-transfer (FR1-FR4). The inference path equals F8's
validation step (``DetectorModule.validation_step``): ``Preprocess(policy, split.modality)`` →
frozen encoder ``encode_pca`` → head under CUDA fp16 autocast (``meta.json`` ``precision: 16-mixed``)
→ ``to_flir_detections`` (FLIR px, all ≤ 300 boxes per image, no score threshold). Each batch is
encoded once and shared by all heads, so the heads must agree on their encoder (``shared_encoder_meta``).

Detections file, write-once (TechStack §4.4)::

    $IRWM_OUTPUT_ROOT/detection/detections/<encoder_tag>/<head>/<modality>_<split>.json.gz

gzipped JSON (``mtime=0``, sorted keys: identical content → identical bytes) holding the COCO
results list, head/encoder provenance and per-image ``video_id``/``frame_index`` + the split's
``video_tags``. Consumers re-score it with ``coco_map`` instead of running inference again (F6, F9, F12).

Aligned FLIR v1 (F19, Roadmap Phase 2b): ``predict_heads`` generalises ``predict_split`` to several
head groups on one split. Each batch is decoded once, preprocessed once per resolution policy and
encoded once per frozen encoder; a group with ``encoder=None`` (the fine-tuned MAE ViTDet heads) gets the
preprocessed images instead of features. It works on any ``FlirSplit``-like split (``AlignedSplit``)
and takes the split's label map. Its detections files carry ``dataset`` and have no
``video_id``/``frame_index``/``video_tags`` when the split has none; FLIR v2 files are unchanged.
Heads are never scored on another dataset's split (Roadmap R12, ``check_same_dataset``).

Imported explicitly (``irwm.detection.inference``), not from ``irwm.detection``.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from irwm.data.detection import FlirDetectionDataset, collate_detection
from irwm.data.flir import FlirImageDataset, FlirSplit, Modality, Split
from irwm.detection.coco_val import to_flir_detections
from irwm.detection.detector import DINODetector
from irwm.detection.frozen import PRECISION, frozen_meta
from irwm.features.preprocess import Policy, Preprocess
from irwm.utils.paths import output_root
from irwm.utils.tracking import git_info

FORMAT = "irwm.detections/1"
CONFIG_KEYS = ("policy", "precision", "hub_version", "attention")  # Hydra encoder group vs meta.json


@dataclass(frozen=True)
class SplitPrediction:
    """Detections of every head on one split, plus timing (wall seconds, CUDA-synchronised)."""

    detections: dict[str, list[dict]]  # head name -> COCO results in FLIR px, image_id ascending
    image_ids: list[int]
    encode_s: float
    head_s: dict[str, float]


# --------------------------------------------------------------------------- FR1
def shared_encoder_meta(names: Sequence[str]) -> dict:
    """``{"encoder", "image_size", "precision"}`` common to the frozen heads ``names``.

    Raises ``ValueError`` on an empty ``names`` or on the first differing key.
    """
    names = list(names)
    if not names:
        raise ValueError("shared_encoder_meta needs at least one frozen head name")
    shared: dict | None = None
    for name in names:
        meta = frozen_meta(name)
        cur = {"encoder": dict(meta["encoder"]), "image_size": tuple(meta["image_size"]),
               "precision": meta["precision"]}
        if shared is None:
            shared, first = cur, name
            continue
        for key in sorted(set(shared["encoder"]) | set(cur["encoder"])):
            a, b = shared["encoder"].get(key), cur["encoder"].get(key)
            if a != b:
                raise ValueError(f"frozen heads disagree on encoder.{key}: {a!r} ({first}) != {b!r} ({name})")
        for key in ("image_size", "precision"):
            if shared[key] != cur[key]:
                raise ValueError(f"frozen heads disagree on {key}: {shared[key]!r} ({first}) != {cur[key]!r} ({name})")
    return shared


def check_encoder_config(cfg_encoder: Mapping[str, Any], shared: Mapping[str, Any]) -> None:
    """``ValueError`` if the Hydra ``encoder`` node differs from the heads' encoder on ``CONFIG_KEYS``."""
    for key in CONFIG_KEYS:
        a, b = cfg_encoder.get(key), shared["encoder"].get(key)
        if a != b:
            raise ValueError(f"config encoder.{key} = {a!r}, but the frozen heads were trained with {b!r}")


# --------------------------------------------------------------------------- FR2
def _sync(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def predict_split(detectors: Mapping[str, DINODetector], split: FlirSplit, encoder: nn.Module, *,
                  policy: Policy, image_size: tuple[int, int], batch_size: int = 16,
                  num_workers: int = 0, device: str | torch.device = "cpu",
                  log_every: int = 0) -> SplitPrediction:
    """Every frame of ``split`` (ascending ``image_id``) encoded once per batch, then every head on it.

    ``encoder.encode_pca``: ``[B, 3, H, W]`` → ``[B, H', W', 1152]``; heads get ``[B, 1152, H', W']``.
    Heads must be in eval mode. On CUDA the head forward runs under fp16 autocast, as F8 validated.
    ``log_every > 0`` prints progress every ``log_every`` batches and after the last one.
    """
    if not detectors:
        raise ValueError("predict_split needs at least one detector")
    training = [name for name, det in detectors.items() if det.training]
    if training:
        raise ValueError(f"detectors in training mode: {training}; call .eval() (frozen heads already are)")
    device = torch.device(device)
    dataset = FlirDetectionDataset(split, policy=policy, flip_p=0.0, filter_empty=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda", collate_fn=collate_detection)
    autocast = ((lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda"
                else contextlib.nullcontext)

    detections: dict[str, list[dict]] = {name: [] for name in detectors}
    head_s = {name: 0.0 for name in detectors}
    encode_s = 0.0
    image_ids: list[int] = []
    t_start = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            images = batch["images"].to(device, non_blocking=True)
            t0 = _sync(device)
            feats = encoder.encode_pca(images).permute(0, 3, 1, 2).contiguous()
            t1 = _sync(device)
            encode_s += t1 - t0
            for name, det in detectors.items():
                with autocast():
                    preds = det(feats, image_size=image_size)
                t2 = _sync(device)
                head_s[name] += t2 - t1
                detections[name].extend(to_flir_detections(preds, batch["image_ids"], batch["geometries"]))
                t1 = _sync(device)
            image_ids.extend(batch["image_ids"])
            if log_every > 0 and (i % log_every == 0 or i == len(loader)):
                done, elapsed = len(image_ids), time.perf_counter() - t_start
                eta = elapsed / done * (len(dataset) - done)
                print(f"[predict_split] {split.modality}/{split.split}: {done}/{len(dataset)} images, "
                      f"{elapsed:.0f} s elapsed, ~{eta:.0f} s left", flush=True)
    return SplitPrediction(detections=detections, image_ids=image_ids, encode_s=encode_s, head_s=head_s)


# --------------------------------------------------------------------------- F19: several head groups
@dataclass(frozen=True)
class HeadGroup:
    """Heads that read the same input: one frozen encoder's features, or (``encoder=None``) the images.

    ``encoder.encode``: ``[B, 3, H, W]`` → ``[B, C, H', W']`` (``irwm.features.encoder.FrozenEncoder``).
    Every head is called as ``head(x, image_size=image_size)`` and returns per-image
    ``{"boxes" [K, 4] xyxy input px, "scores" [K], "labels" [K]}`` (``DINODetector``, the MAE ViTDet).
    ``image_size`` = ``(H, W)`` of the preprocessed image (``POLICY_SIZE[policy]``).
    """

    heads: Mapping[str, nn.Module]
    policy: str
    image_size: tuple[int, int]
    encoder: nn.Module | None = None
    autocast: bool = True  # CUDA fp16 autocast over the heads (their meta precision is 16-mixed)


@dataclass(frozen=True)
class GroupedPrediction:
    """Detections of every head of every group on one split, plus timing (wall s, CUDA-synchronised)."""

    detections: dict[str, list[dict]]  # head name -> COCO results in source px, image_id ascending
    image_ids: list[int]
    encode_s: dict[str, float]  # group name -> encoding time (0 for image groups)
    head_s: dict[str, float]  # head name -> head time


class _MultiPolicyImages(torch.utils.data.Dataset):
    """Each frame of a split decoded once and preprocessed under every policy in ``policies``."""

    def __init__(self, split: Any, policies: Sequence[str]) -> None:
        self.split = split
        self.preprocess = {p: Preprocess(p, split.modality) for p in policies}  # type: ignore[arg-type]
        self._images = split.image_dataset() if hasattr(split, "image_dataset") else FlirImageDataset(split)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, i: int) -> dict:
        item = self._images[i]
        out = {p: pre(item["image"]) for p, pre in self.preprocess.items()}
        return {"images": {p: x for p, (x, _) in out.items()}, "geometries": {p: g for p, (_, g) in out.items()},
                "image_id": int(item["image_id"])}


def _collate_multi(items: list[dict]) -> dict:
    policies = list(items[0]["images"])
    return {"images": {p: torch.stack([it["images"][p] for it in items]) for p in policies},
            "geometries": {p: [it["geometries"][p] for it in items] for p in policies},
            "image_ids": [it["image_id"] for it in items]}


def _to_detections(preds: list[dict], image_ids: list[int], geometries: list,
                   label_to_category_id: Mapping[int, int] | None) -> list[dict]:
    if label_to_category_id is None:
        return to_flir_detections(preds, image_ids, geometries)
    return to_flir_detections(preds, image_ids, geometries, label_to_category_id=dict(label_to_category_id))


def predict_heads(groups: Mapping[str, HeadGroup], split: Any, *, batch_size: int = 16, num_workers: int = 0,
                  device: str | torch.device = "cpu", label_to_category_id: Mapping[int, int] | None = None,
                  log_every: int = 0) -> GroupedPrediction:
    """Every frame of ``split`` (its record order, ascending ``image_id``) through every head of every group.

    Per batch: decode once, preprocess once per distinct ``policy``, encode once per group with an
    encoder, run each head of the group on that input. ``label_to_category_id`` maps the heads' labels
    to the split's COCO category ids (``None``: the FLIR v2 6-class map). Head names must be unique
    across groups and heads in eval mode. With one dinopca group whose encoder ``encode`` equals
    ``encode_pca`` permuted, this gives exactly ``predict_split``'s detections.
    """
    if not groups or not any(g.heads for g in groups.values()):
        raise ValueError("predict_heads needs at least one head")
    names = [h for g in groups.values() for h in g.heads]
    if len(set(names)) != len(names):
        raise ValueError(f"head names repeat across groups: {names}")
    training = [h for g in groups.values() for h, m in g.heads.items() if m.training]
    if training:
        raise ValueError(f"heads in training mode: {training}; call .eval() (frozen heads already are)")
    device = torch.device(device)
    policies = sorted({g.policy for g in groups.values()})
    dataset = _MultiPolicyImages(split, policies)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda", collate_fn=_collate_multi)

    def autocast(enabled: bool):
        return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" \
            else contextlib.nullcontext()

    detections: dict[str, list[dict]] = {h: [] for h in names}
    head_s = {h: 0.0 for h in names}
    encode_s = {g: 0.0 for g in groups}
    image_ids: list[int] = []
    t_start = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            on_device = {p: x.to(device, non_blocking=True) for p, x in batch["images"].items()}
            for gname, group in groups.items():
                if not group.heads:
                    continue
                t0 = _sync(device)
                x = on_device[group.policy]
                if group.encoder is not None:
                    x = group.encoder.encode(x)
                t1 = _sync(device)
                encode_s[gname] += t1 - t0
                for h, head in group.heads.items():
                    with autocast(group.autocast):
                        preds = head(x, image_size=tuple(group.image_size))
                    t2 = _sync(device)
                    head_s[h] += t2 - t1
                    detections[h].extend(_to_detections(preds, batch["image_ids"], batch["geometries"][group.policy],
                                                        label_to_category_id))
                    t1 = _sync(device)
            image_ids.extend(batch["image_ids"])
            if log_every > 0 and (i % log_every == 0 or i == len(loader)):
                done, elapsed = len(image_ids), time.perf_counter() - t_start
                eta = elapsed / done * (len(dataset) - done)
                print(f"[predict_heads] {split.modality}/{split.split}: {done}/{len(dataset)} images, "
                      f"{elapsed:.0f} s elapsed, ~{eta:.0f} s left", flush=True)
    return GroupedPrediction(detections=detections, image_ids=image_ids, encode_s=encode_s, head_s=head_s)


# --------------------------------------------------------------------------- FR3
def with_person_car(metrics: Mapping[str, float]) -> dict[str, float]:
    """``coco_map`` output plus ``AP_person_car`` / ``AP50_person_car`` (mean of the two classes; Roadmap R10).

    Equal to ``COCOeval`` restricted to ``catIds = [1, 3]``; ``nan`` if either class is ``nan``.
    """
    out = dict(metrics)
    out["AP_person_car"] = (metrics["AP_person"] + metrics["AP_car"]) / 2
    out["AP50_person_car"] = (metrics["AP50_person"] + metrics["AP50_car"]) / 2
    return out


# --------------------------------------------------------------------------- FR4
def detections_path(name: str, modality: Modality, split: Split = "test") -> Path:
    """``$IRWM_OUTPUT_ROOT/detection/detections/<encoder_tag>/<name>/<modality>_<split>.json.gz``."""
    return output_root() / "detection" / "detections" / head_tag(name) / name / f"{modality}_{split}.json.gz"


def head_tag(name: str) -> str:
    """``meta["encoder"]["encoder_tag"]`` of a frozen head; ``<kind>_<policy>`` if the meta has no tag."""
    enc = frozen_meta(name)["encoder"]
    tag = enc.get("encoder_tag")
    return tag if tag else f"{enc.get('kind', 'dinopca')}_{enc['policy']}"


# --------------------------------------------------------------------------- R12
def split_dataset(split: Any) -> str:
    """``"aligned_v1"`` for an ``AlignedSplit``, else ``split.dataset`` if set, else ``"flir_v2"``."""
    from irwm.data.aligned import AlignedSplit

    if isinstance(split, AlignedSplit):
        return "aligned_v1"
    return str(getattr(split, "dataset", "flir_v2"))


def check_same_dataset(head: str, split_or_dataset: Any) -> None:
    """``ValueError`` unless head ``head`` was trained on the dataset of the split (Roadmap R12).

    ``split_or_dataset``: a split or a dataset name (``"flir_v2"`` | ``"aligned_v1"``). Metas without
    ``dataset`` are FLIR v2. Cross-dataset cells (aligned heads on v2 or the reverse) are never scored.
    """
    dataset = split_or_dataset if isinstance(split_or_dataset, str) else split_dataset(split_or_dataset)
    trained = frozen_meta(head).get("dataset", "flir_v2")
    if trained != dataset:
        raise ValueError(f"head {head!r} was trained on {trained!r} and cannot be scored on {dataset!r} "
                         "(no cross-dataset cells, Roadmap R12)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_name() -> str:
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def save_detections(path: Path, detections: list[dict], *, head: str, split: FlirSplit,
                    overwrite: bool = False) -> str:
    """Write the detections file of ``head`` on ``split`` (write-once); returns its SHA-256.

    ``FileExistsError`` if ``path`` exists and not ``overwrite``; ``ValueError`` on a detection whose
    ``image_id`` is not in ``split``. Written to ``<path>.tmp``, then renamed.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    check_same_dataset(head, split)
    known = {r.image_id for r in split.records}
    unknown = sorted({d["image_id"] for d in detections} - known)
    if unknown:
        raise ValueError(f"detections reference image_ids not in {split_dataset(split)} "
                         f"{split.modality}/{split.split}: {unknown[:10]}")

    meta = frozen_meta(head)
    video = hasattr(split, "video_tags")  # FLIR v2: per-image video ids and the split's tags
    images = []
    for r in split.records:
        im = {"image_id": r.image_id, "file_name": r.file_name, "width": r.width, "height": r.height}
        if video:
            im.update(video_id=r.video_id, frame_index=r.frame_index)
        images.append(im)
    obj = {
        "format": FORMAT,
        "head": head,
        "head_sha256": meta["head_sha256"],
        "head_modality": meta["modality"],
        "encoder": meta["encoder"],
        "image_size": meta["image_size"],
        "precision": PRECISION,
        "modality": split.modality,
        "split": split.split,
        "n_images": len(split.records),
        "n_detections": len(detections),
        "images": images,
        "detections": detections,
        "git": git_info(),
        "created_at": _now(),
        "device": _device_name(),
    }
    if video:
        obj["video_tags"] = split.video_tags() if split.split == "test" else {}
    if meta.get("dataset", "flir_v2") != "flir_v2":  # FLIR v2 files keep F15's exact keys
        obj["dataset"] = meta["dataset"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.GzipFile(tmp, "wb", mtime=0) as f:
        f.write(json.dumps(obj, sort_keys=True).encode("utf-8"))
    tmp.replace(path)
    return _sha256(path)


def load_detections(path: Path) -> dict:
    """The object written by ``save_detections``."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def check_outputs_free(paths: Iterable[Path], *, overwrite: bool) -> None:
    """``FileExistsError`` listing every existing path, unless ``overwrite``."""
    if overwrite:
        return
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise FileExistsError("detection files already exist (pass overwrite=true to replace them): "
                              + ", ".join(existing))
