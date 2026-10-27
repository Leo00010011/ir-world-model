"""Off-the-shelf val mAP for the DINO-DETR heads: pycocotools on ``FlirSplit.to_coco_gt``.

Spec: spec/2026-10-26-detector-training-pipeline (FR4). Predictions come back from the encoder
input frame to FLIR pixels with ``ResizeGeometry.invert_boxes``, so they are scored against the
full-frame FLIR GT, ignore regions as per-category crowd boxes and 640×512-normalised areas (F2).
``COCOeval`` runs with its default parameters, as detrex's ``COCOEvaluator`` does. This is a
selection metric only; the rigorous evaluator (CIs, per horizon) is F6.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor

from irwm.data.classes import CLASS_NAMES, LABEL_TO_CATEGORY_ID
from irwm.features.preprocess import ResizeGeometry

_STATS = ("AP", "AP50", "AP75", "APs", "APm", "APl")  # COCOeval.stats[0:6]


def to_flir_detections(preds: list[dict[str, Tensor]], image_ids: list[int],
                       geometries: list[ResizeGeometry]) -> list[dict]:
    """``DINODetector`` eval outputs (xyxy encoder-input px) → COCO result dicts in FLIR px.

    All boxes are kept (≤ 300 per image); ``COCOeval``'s ``maxDets`` keeps the top 100.
    """
    out = []
    for pred, image_id, geom in zip(preds, image_ids, geometries, strict=True):
        xyxy = pred["boxes"].detach().float().cpu().numpy().reshape(-1, 4)
        xywh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], axis=1)
        boxes = geom.invert_boxes(xywh)
        scores = pred["scores"].detach().float().cpu().tolist()
        labels = pred["labels"].detach().cpu().tolist()
        for b, s, lab in zip(boxes, scores, labels):
            out.append({"image_id": int(image_id), "category_id": LABEL_TO_CATEGORY_ID[int(lab)],
                        "bbox": [float(v) for v in b], "score": float(s)})
    return out


def _classes_without_gt(gt: dict) -> set[str]:
    """Classes with no non-crowd GT box: ``COCOeval`` gives them no precision (``-1``)."""
    present = {a["category_id"] for a in gt["annotations"] if not a.get("iscrowd", 0)}
    return {c for k, c in enumerate(CLASS_NAMES) if LABEL_TO_CATEGORY_ID[k] not in present}


def _mean_valid(p: np.ndarray) -> float:
    valid = p[p > -1]
    return float(valid.mean()) if valid.size else math.nan


def coco_map(gt: dict, detections: list[dict]) -> dict[str, float]:
    """``AP``/``AP50``/``AP75``/``APs``/``APm``/``APl`` plus ``AP_<class>``/``AP50_<class>``; stdout suppressed.

    A class with no GT box is ``nan``. Empty ``detections`` give ``0.0`` everywhere else.
    """
    if not detections:  # pycocotools' loadRes raises on an empty list
        missing = _classes_without_gt(gt)
        metrics = {k: 0.0 for k in _STATS}
        for c in CLASS_NAMES:
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
    for c in CLASS_NAMES:
        k = E.params.catIds.index(LABEL_TO_CATEGORY_ID[CLASS_NAMES.index(c)])
        metrics[f"AP_{c}"] = _mean_valid(p[:, :, k, 0, -1])  # area "all", maxDets 100
        metrics[f"AP50_{c}"] = _mean_valid(p[0, :, k, 0, -1])
    return metrics
