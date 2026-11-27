"""Off-the-shelf val mAP for the DINO-DETR heads: pycocotools on ``FlirSplit.to_coco_gt``.

Spec: spec/2026-10-26-detector-training-pipeline (FR4). Predictions come back from the encoder
input frame to FLIR pixels with ``ResizeGeometry.invert_boxes``, so they are scored against the
full-frame FLIR GT, ignore regions as per-category crowd boxes and 640×512-normalised areas (F2).
``COCOeval`` runs with its default parameters, as detrex's ``COCOEvaluator`` does. This is a
selection metric only; the rigorous evaluator (CIs, per horizon) is F6.

Other label spaces (aligned FLIR v1, F18): pass ``label_to_category_id`` to ``to_flir_detections``
and ``class_names`` to ``coco_map``; per-class category ids are then looked up by name in
``gt["categories"]``. The defaults are FLIR v2's 6-class space, unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
from collections.abc import Mapping, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor

from irwm.data.classes import CLASS_NAMES, LABEL_TO_CATEGORY_ID
from irwm.features.preprocess import ResizeGeometry

_STATS = ("AP", "AP50", "AP75", "APs", "APm", "APl")  # COCOeval.stats[0:6]


def to_flir_detections(preds: list[dict[str, Tensor]], image_ids: list[int],
                       geometries: list[ResizeGeometry], *,
                       label_to_category_id: Mapping[int, int] | None = None) -> list[dict]:
    """``DINODetector`` eval outputs (xyxy encoder-input px) → COCO result dicts in FLIR px.

    All boxes are kept (≤ 300 per image); ``COCOeval``'s ``maxDets`` keeps the top 100.
    ``label_to_category_id`` maps model labels to GT category ids (default: FLIR v2's map).
    """
    cat_of = LABEL_TO_CATEGORY_ID if label_to_category_id is None else label_to_category_id
    out = []
    for pred, image_id, geom in zip(preds, image_ids, geometries, strict=True):
        xyxy = pred["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
        xywh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], axis=1)
        boxes = geom.invert_boxes(xywh)
        scores = pred["scores"].detach().float().cpu().tolist()
        labels = pred["labels"].detach().cpu().tolist()
        for b, s, lab in zip(boxes, scores, labels):
            out.append({"image_id": int(image_id), "category_id": int(cat_of[int(lab)]),
                        "bbox": [float(v) for v in b], "score": float(s)})
    return out


def _category_ids(gt: dict, class_names: Sequence[str] | None) -> dict[str, int]:
    """``{class name: GT category id}``: FLIR v2's map by default, else looked up by name in ``gt``."""
    if class_names is None:
        return {c: LABEL_TO_CATEGORY_ID[k] for k, c in enumerate(CLASS_NAMES)}
    by_name = {c["name"]: int(c["id"]) for c in gt.get("categories", [])}
    missing = [c for c in class_names if c not in by_name]
    if missing:
        raise KeyError(f"classes {missing} not in gt['categories'] {sorted(by_name)}")
    return {c: by_name[c] for c in class_names}


def _classes_without_gt(gt: dict, cat_ids: Mapping[str, int]) -> set[str]:
    """Classes with no non-crowd GT box: ``COCOeval`` gives them no precision (``-1``)."""
    present = {a["category_id"] for a in gt["annotations"] if not a.get("iscrowd", 0)}
    return {c for c, cid in cat_ids.items() if cid not in present}


def _mean_valid(p: np.ndarray) -> float:
    valid = p[p > -1]
    return float(valid.mean()) if valid.size else math.nan


def coco_map(gt: dict, detections: list[dict], *,
             class_names: Sequence[str] | None = None) -> dict[str, float]:
    """``AP``/``AP50``/``AP75``/``APs``/``APm``/``APl`` plus ``AP_<class>``/``AP50_<class>``; stdout suppressed.

    ``class_names`` (default: FLIR v2's ``CLASS_NAMES``) picks the per-class keys; their category
    ids are read from ``gt["categories"]`` by name. A class with no GT box is ``nan``. Empty
    ``detections`` give ``0.0`` everywhere else.
    """
    cat_ids = _category_ids(gt, class_names)
    if not detections:  # pycocotools' loadRes raises on an empty list
        missing = _classes_without_gt(gt, cat_ids)
        metrics = {k: 0.0 for k in _STATS}
        for c in cat_ids:
            metrics[f"AP_{c}"] = metrics[f"AP50_{c}"] = math.nan if c in missing else 0.0
        return metrics

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = copy.deepcopy(gt)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(copy.deepcopy(detections))
        E = COCOeval(coco_gt, coco_dt, iouType="bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()

    metrics = {k: float(v) for k, v in zip(_STATS, E.stats[:6])}
    p = E.eval["precision"]  # [T=10 IoUs, R=101 recalls, K categories, A=4 areas, M=3 maxDets]
    for c, cid in cat_ids.items():
        k = E.params.catIds.index(cid)
        metrics[f"AP_{c}"] = _mean_valid(p[:, :, k, 0, -1])  # area "all", maxDets 100
        metrics[f"AP50_{c}"] = _mean_valid(p[0, :, k, 0, -1])
    return metrics


def fp_metrics(detections: list[dict], image_ids: Sequence[int], category_ids: Mapping[str, int], *,
               thresholds: Sequence[float] = (0.3, 0.5)) -> dict[str, float]:
    """False-positive metrics on frames with no GT box at all, where every detection is a false positive.

    For each score threshold ``t`` (key suffix ``@t``), overall and per class (``_<class>``):

    - ``FPPI@t``: detections with ``score >= t`` per image (the KAIST / Caltech miss-rate curve's x axis);
    - ``frac_fp@t``: fraction of the images with at least one such detection.

    Detections on other images are ignored. ``n_images`` is ``len(image_ids)``; with no image every
    rate is ``nan``. AP is undefined here (no positives), which is why these replace it.
    """
    ids = {int(i) for i in image_ids}
    n = len(ids)
    by_cat = {cid: c for c, cid in category_ids.items()}
    dets = [d for d in detections if int(d["image_id"]) in ids and int(d["category_id"]) in by_cat]
    out: dict[str, float] = {"n_images": float(n)}
    for t in thresholds:
        kept = [d for d in dets if float(d["score"]) >= t]
        groups = {"": kept, **{f"_{c}": [d for d in kept if by_cat[int(d["category_id"])] == c]
                               for c in category_ids}}
        for suffix, ds in groups.items():
            out[f"FPPI@{t:g}{suffix}"] = len(ds) / n if n else math.nan
            out[f"frac_fp@{t:g}{suffix}"] = len({int(d["image_id"]) for d in ds}) / n if n else math.nan
    return out
