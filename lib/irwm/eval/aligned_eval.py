"""Aligned FLIR v1 evaluation (F19): every aligned head on both modalities of the aligned test split.

``plan_groups`` sorts the requested heads by what they read: one group per frozen DINO encoder kind
(``dinopca`` heads share one encoding, ``dino3072`` heads another) and one image group for the
fine-tuned MAE ViTDet heads (``meta["detector"]["arch"] == "vitdet_mae"``, ``native640``). The heads
of a group must agree on their encoder settings (``shared_encoder_meta``). ``evaluate_aligned`` then,
per test modality, runs ``predict_heads`` once (each frame decoded once, each encoder run once), saves
every head's detections (write-once, ``detections_path``) and scores them with ``score_aligned``.

Roadmap R12 is enforced: only ``aligned_*`` heads whose ``meta["dataset"] == "aligned_v1"``, only on
``AlignedSplit``s.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from irwm.eval.aligned_table import ALIGNED_DATASET, parse_aligned_head, score_aligned, transfer_ratios

IMAGE_GROUP_PREFIX = "images_"


def head_arch(meta: Mapping[str, Any]) -> str:
    """``meta["detector"]["arch"]``, ``"dino_detr"`` for frozen-feature ``DINODetector`` heads."""
    return meta["detector"].get("arch") or "dino_detr"


def plan_groups(heads: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``{group: {"heads", "encoder" (meta encoder or None), "policy", "image_size", "precision"}}``.

    Groups are ordered dinopca, dino3072, then image groups. ``ValueError`` for a non-aligned name, a
    head whose meta dataset is not ``aligned_v1`` (R12), repeated heads, or heads of one group that
    disagree on their encoder, image size or precision.
    """
    from irwm.detection.frozen import MAE_ARCH, frozen_meta
    from irwm.detection.inference import check_same_dataset, shared_encoder_meta
    from irwm.features.preprocess import POLICY_SIZE

    heads = list(heads)
    if not heads:
        raise ValueError("no heads to evaluate")
    if len(set(heads)) != len(heads):
        raise ValueError(f"repeated heads: {heads}")
    members: dict[str, list[str]] = {}
    for h in heads:
        parse_aligned_head(h)
        check_same_dataset(h, ALIGNED_DATASET)
        meta = frozen_meta(h)
        arch = head_arch(meta)
        if arch == MAE_ARCH:
            key = IMAGE_GROUP_PREFIX + meta["encoder"]["policy"]
        elif arch == "dino_detr":
            key = meta["encoder"].get("kind", "dinopca")
        else:
            raise ValueError(f"{h}: unknown detector arch {arch!r}")
        members.setdefault(key, []).append(h)

    order = {"dinopca": 0, "dino3072": 1}
    plan: dict[str, dict[str, Any]] = {}
    for key in sorted(members, key=lambda k: (order.get(k, 2), k)):
        names = members[key]
        shared = shared_encoder_meta(names)
        image = key.startswith(IMAGE_GROUP_PREFIX)
        policy = shared["encoder"]["policy"]
        size = tuple(shared["image_size"]) if shared.get("image_size") else tuple(POLICY_SIZE[policy])
        plan[key] = {"heads": names, "encoder": None if image else shared["encoder"], "policy": policy,
                     "image_size": size, "precision": shared["precision"]}
    return plan


def load_aligned_head(name: str, device: str | torch.device) -> nn.Module:
    """Frozen aligned head in eval mode: ``load_frozen_vitdet`` for MAE heads, else ``load_frozen_detector``."""
    from irwm.detection.frozen import MAE_ARCH, frozen_meta, load_frozen_detector

    if head_arch(frozen_meta(name)) == MAE_ARCH:
        from irwm.detection.frozen_vitdet import load_frozen_vitdet

        return load_frozen_vitdet(name, device=device)
    return load_frozen_detector(name, device=device)


def build_encoder(encoder_meta: Mapping[str, Any], device: str | torch.device) -> nn.Module:
    """Frozen DINO encoder of a head group (``frozen_encoder_from_config`` on the heads' meta encoder)."""
    from irwm.features.encoder import frozen_encoder_from_config

    return frozen_encoder_from_config(encoder_meta, device=device)


def evaluate_aligned(heads: Sequence[str], *, test_modalities: Sequence[str] = ("rgb", "thermal"),
                     split: str = "test", batch_size: int = 16, num_workers: int = 0,
                     device: str | torch.device = "cuda", overwrite: bool = False, root: Path | None = None,
                     log_every: int = 0, verify_rescore: bool = True, max_images: int | None = None,
                     head_loader: Callable[[str, Any], nn.Module] = load_aligned_head,
                     encoder_builder: Callable[[Mapping[str, Any], Any], nn.Module] = build_encoder,
                     ) -> dict[str, Any]:
    """Score ``heads`` on ``test_modalities`` of the aligned ``split``; returns the result dict.

    Keys: ``cells[head][modality]`` (metrics), ``transfer_ratio``, ``groups``, ``heads`` (provenance),
    ``detections[head][modality]`` (path relative to ``$IRWM_OUTPUT_ROOT``, sha256, counts),
    ``n_images``, ``timing``, ``rescore_identical`` (``verify_rescore``: every saved file re-scored from
    disk gives the same metrics). ``max_images`` (smoke runs only) keeps the first N pairs and writes the
    detections under ``<modality>_<split>-first<N>.json.gz``, never the real file. Detection files are write-once: ``FileExistsError`` before any
    loading if one exists and not ``overwrite``.
    """
    from irwm.data.aligned import ALIGNED_LABEL_TO_CATEGORY_ID, AlignedSplit
    from irwm.detection.frozen import frozen_meta
    from irwm.detection.inference import (HeadGroup, check_outputs_free, check_same_dataset, detections_path,
                                          predict_heads, save_detections)
    from irwm.eval.aligned_table import rescore_detections
    from irwm.utils.paths import output_root

    t_start = time.perf_counter()
    heads = list(heads)
    plan = plan_groups(heads)
    modalities = list(test_modalities)
    file_split = split if max_images is None else f"{split}-first{int(max_images)}"
    paths = {h: {m: detections_path(h, m, file_split) for m in modalities} for h in heads}  # type: ignore[arg-type]
    check_outputs_free([p for per in paths.values() for p in per.values()], overwrite=overwrite)

    groups: dict[str, HeadGroup] = {}
    for g, p in plan.items():
        loaded = {h: head_loader(h, device) for h in p["heads"]}
        encoder = encoder_builder(p["encoder"], device) if p["encoder"] is not None else None
        groups[g] = HeadGroup(heads=loaded, policy=p["policy"], image_size=p["image_size"], encoder=encoder)

    cells: dict[str, dict[str, dict]] = {h: {} for h in heads}
    det_files: dict[str, dict[str, dict]] = {h: {} for h in heads}
    n_images: dict[str, int] = {}
    encode_s: dict[str, dict[str, float]] = {}
    head_s: dict[str, dict[str, float]] = {h: {} for h in heads}
    rescore_ok: dict[str, dict[str, bool]] = {h: {} for h in heads}
    for m in modalities:
        s = AlignedSplit.load(m, split, root)  # type: ignore[arg-type]
        if max_images is not None:
            s = s.subset([r.image_id for r in s.records[:int(max_images)]])
        for h in heads:
            check_same_dataset(h, s)
        pred = predict_heads(groups, s, batch_size=batch_size, num_workers=num_workers, device=device,
                             label_to_category_id=ALIGNED_LABEL_TO_CATEGORY_ID, log_every=log_every)
        n_images[m] = len(pred.image_ids)
        encode_s[m] = pred.encode_s
        gt = s.to_coco_gt("normalised")
        for h in heads:
            dets = pred.detections[h]
            sha = save_detections(paths[h][m], dets, head=h, split=s, overwrite=overwrite)
            det_files[h][m] = {"path": paths[h][m].relative_to(output_root()).as_posix(), "sha256": sha,
                               "n_images": len(s.records), "n_detections": len(dets)}
            cells[h][m] = score_aligned(gt, dets)
            head_s[h][m] = pred.head_s[h]
            if verify_rescore:
                again = rescore_detections(paths[h][m], root=root)
                rescore_ok[h][m] = _same_metrics(again, cells[h][m])
            print(f"[eval_aligned] {h} on {m}: AP p+c {cells[h][m]['AP_person_car']:.4f} "
                  f"AP50 p+c {cells[h][m]['AP50_person_car']:.4f}", flush=True)

    metas = {h: frozen_meta(h) for h in heads}
    return {
        "cells": cells,
        "transfer_ratio": transfer_ratios(cells),
        "groups": {g: {**p, "image_size": list(p["image_size"])} for g, p in plan.items()},
        "heads": {h: {"backbone": parse_aligned_head(h)[0], "train_modality": metas[h]["modality"],
                      "dataset": metas[h]["dataset"], "arch": head_arch(metas[h]),
                      "head_sha256": metas[h].get("head_sha256"), "source_ckpt": metas[h].get("source_ckpt"),
                      "epoch": metas[h].get("epoch"), "encoder": metas[h]["encoder"],
                      "val_metrics": metas[h].get("val_metrics")} for h in heads},
        "detections": det_files,
        "n_images": n_images,
        "split": split,
        "max_images": max_images,
        "rescore_identical": rescore_ok if verify_rescore else None,
        "timing": {"encode_s": encode_s, "head_s": head_s, "wall_s": time.perf_counter() - t_start},
    }


def _same_metrics(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    if set(a) != set(b):
        return False
    return all((x != x and y != y) or x == y for x, y in ((a[k], b[k]) for k in a))  # nan == nan


def merge_aligned(heads: Sequence[str], *, test_modalities: Sequence[str] = ("rgb", "thermal"), split: str = "test",
                  root: Path | None = None) -> dict[str, Any]:
    """Rebuild one result from the saved detection files of ``heads`` (no model, no GPU): re-scores each file.

    Combines subset runs into one Table B. Missing files are listed under ``missing``, not an error, but at
    least one file must exist. ``ValueError`` if a file belongs to another head, dataset or split, or its
    ``head_sha256`` differs from the head's current ``meta.json`` (a stale file of a re-exported head).
    """
    from irwm.detection.frozen import frozen_meta
    from irwm.detection.inference import check_same_dataset, detections_path, load_detections
    from irwm.eval.aligned_table import rescore_detections
    from irwm.utils.paths import output_root

    t_start = time.perf_counter()
    heads = list(heads)
    cells: dict[str, dict[str, dict]] = {}
    det_files: dict[str, dict[str, dict]] = {}
    n_images: dict[str, int] = {}
    missing: list[str] = []
    for h in heads:
        parse_aligned_head(h)
        check_same_dataset(h, ALIGNED_DATASET)
        meta = frozen_meta(h)
        for m in test_modalities:
            path = detections_path(h, m, split)  # type: ignore[arg-type]
            if not path.is_file():
                missing.append(path.relative_to(output_root()).as_posix())
                continue
            obj = load_detections(path)
            if (obj.get("head"), obj.get("modality"), obj.get("split")) != (h, m, split):
                raise ValueError(f"{path}: holds {obj.get('head')}/{obj.get('modality')}/{obj.get('split')}")
            if obj.get("head_sha256") != meta.get("head_sha256"):
                raise ValueError(f"{path}: head_sha256 {obj.get('head_sha256')} != meta.json "
                                 f"{meta.get('head_sha256')} (stale detections of a re-exported head)")
            cells.setdefault(h, {})[m] = rescore_detections(path, root=root)
            det_files.setdefault(h, {})[m] = {
                "path": path.relative_to(output_root()).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "n_images": obj["n_images"], "n_detections": obj["n_detections"],
                "created_at": obj.get("created_at"), "device": obj.get("device"), "git": obj.get("git")}
            n_images[m] = obj["n_images"]
    if not cells:
        raise FileNotFoundError(f"no saved aligned detections for {heads} on {list(test_modalities)}: {missing}")
    metas = {h: frozen_meta(h) for h in cells}
    return {
        "mode": "merge",
        "cells": cells,
        "transfer_ratio": transfer_ratios(cells),
        "heads": {h: {"backbone": parse_aligned_head(h)[0], "train_modality": metas[h]["modality"],
                      "dataset": metas[h]["dataset"], "arch": head_arch(metas[h]),
                      "head_sha256": metas[h].get("head_sha256"), "source_ckpt": metas[h].get("source_ckpt"),
                      "epoch": metas[h].get("epoch"), "encoder": metas[h]["encoder"],
                      "val_metrics": metas[h].get("val_metrics")} for h in cells},
        "detections": det_files,
        "missing": missing,
        "n_images": n_images,
        "split": split,
        "timing": {"wall_s": time.perf_counter() - t_start},
    }
