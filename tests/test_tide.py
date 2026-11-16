"""TIDE error decomposition (``irwm.eval.tide``): one hand-built case per error type, the oracle fixes,
and exact agreement of the base AP50 with pycocotools (``irwm.detection.coco_val.coco_map``). CPU only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irwm.detection.coco_val import coco_map
from irwm.eval.tide import ERROR_TYPES, box_iou, tide

PERSON, CAR = 1, 2
CATS = [{"id": PERSON, "name": "person"}, {"id": CAR, "name": "car"}]


def make_gt(boxes_per_image: dict[int, list[tuple[int, list[float]]]], crowd: set[int] = frozenset()) -> dict:
    """``{image_id: [(category_id, xywh), ...]}`` → COCO GT; ``crowd`` holds annotation ids."""
    images, anns, aid = [], [], 0
    for i, rows in boxes_per_image.items():
        images.append({"id": i, "width": 640, "height": 512, "file_name": f"{i}.jpg"})
        for c, b in rows:
            aid += 1
            anns.append({"id": aid, "image_id": i, "category_id": c, "bbox": list(map(float, b)),
                         "area": float(b[2] * b[3]), "iscrowd": int(aid in crowd)})
    return {"images": images, "annotations": anns, "categories": CATS}


def det(i, c, b, s):
    return {"image_id": i, "category_id": c, "bbox": list(map(float, b)), "score": float(s)}


BOX = [100, 100, 100, 100]
SHIFTED = [160, 100, 100, 100]  # IoU with BOX = 40·100 / (2·10^4 − 4000) = 0.25
FAR = [400, 300, 50, 50]


def only(result, kind):
    return {k: v for k, v in result.counts.items() if k in ERROR_TYPES and v} == {kind: 1}


def test_box_iou_matches_definition():
    assert box_iou(np.array([BOX]), np.array([SHIFTED]))[0, 0] == pytest.approx(0.25)
    assert box_iou(np.array([BOX]), np.array([BOX]))[0, 0] == pytest.approx(1.0)
    # crowd: intersection over the detection's area
    assert box_iou(np.array([[100, 100, 50, 50]]), np.array([BOX]), np.array([True]))[0, 0] == pytest.approx(1.0)


def test_perfect_detector_has_no_errors():
    gt = make_gt({1: [(CAR, BOX), (PERSON, FAR)]})
    r = tide(gt, [det(1, CAR, BOX, 0.9), det(1, PERSON, FAR, 0.8)])
    assert r.ap50 == pytest.approx(1.0)
    assert all(r.counts[t] == 0 for t in ERROR_TYPES)
    assert all(abs(v) < 1e-12 for v in r.delta.values())
    assert r.counts["TP"] == r.counts["GT"] == 2


@pytest.mark.parametrize("kind, dets", [
    ("Loc", [det(1, CAR, SHIFTED, 0.9)]),
    ("Cls", [det(1, PERSON, BOX, 0.9)]),
    ("Both", [det(1, PERSON, SHIFTED, 0.9)]),
    ("Bkg", [det(1, CAR, BOX, 0.5), det(1, CAR, FAR, 0.9)]),
    ("Dupe", [det(1, CAR, BOX, 0.5), det(1, CAR, BOX, 0.9), det(2, CAR, BOX, 0.3)]),  # dupe ranks above a TP
])
def test_each_false_positive_type(kind, dets):
    gt = make_gt({1: [(CAR, BOX)], 2: [(CAR, BOX)]} if kind == "Dupe" else {1: [(CAR, BOX)]})
    r = tide(gt, dets)
    assert only(r, kind) or (kind in ("Loc", "Cls", "Both") and r.counts[kind] == 1), r.counts
    if kind in ("Loc", "Cls"):  # the GT is the target of the error, so it is not also a Miss
        assert r.counts["Miss"] == 0
        # fixing the error turns it into a TP of the car class: perfect car AP
        assert r.ap50_per_class[CAR] + r.delta_per_class[kind][CAR] == pytest.approx(1.0)
    if kind == "Both":  # Both is suppressed, the car GT stays missed
        assert r.counts["Miss"] == 1
    if kind in ("Bkg", "Dupe"):  # a high-scoring FP above the TP: suppressing it gives AP 1
        assert r.ap50 < 1.0
        assert r.ap50 + r.delta[kind] == pytest.approx(1.0)
        assert r.delta["FP"] == pytest.approx(r.delta[kind])


def test_miss_and_fn_oracles():
    gt = make_gt({1: [(CAR, BOX), (CAR, FAR)]})
    r = tide(gt, [det(1, CAR, BOX, 0.9)])
    assert r.counts["Miss"] == 1 and r.counts["GT"] == 2
    assert r.ap50 == pytest.approx(0.5, abs=0.01)  # recall 0.5 at precision 1 (101-point grid)
    assert r.ap50 + r.delta["Miss"] == pytest.approx(1.0)
    assert r.delta["FN"] == pytest.approx(r.delta["Miss"])


def test_loc_fix_on_an_already_matched_gt_suppresses():
    # the TP takes the GT; the low-scoring shifted box is Loc on a used GT: fixing it only suppresses it
    gt = make_gt({1: [(CAR, BOX)]})
    r = tide(gt, [det(1, CAR, BOX, 0.4), det(1, CAR, SHIFTED, 0.9)])
    assert r.counts["Loc"] == 1
    assert r.ap50 + r.delta["Loc"] == pytest.approx(1.0)  # the FP above the TP is gone


def test_two_loc_errors_on_one_gt_fix_only_one():
    gt = make_gt({1: [(CAR, BOX)], 2: [(CAR, BOX)]})
    dets = [det(1, CAR, SHIFTED, 0.9), det(1, CAR, [40, 100, 100, 100], 0.8), det(2, CAR, BOX, 0.7)]
    r = tide(gt, dets)
    assert r.counts["Loc"] == 2
    # after the fix: TP(0.9), suppressed(0.8), TP(0.7) → AP 1; a double-count would add a duplicate FP
    assert r.ap50 + r.delta["Loc"] == pytest.approx(1.0)


def test_detection_on_crowd_region_is_ignored():
    gt = make_gt({1: [(CAR, BOX), (CAR, [300, 300, 200, 150])]}, crowd={2})
    r = tide(gt, [det(1, CAR, BOX, 0.5), det(1, CAR, [320, 320, 60, 60], 0.9)])
    assert r.counts["GT"] == 1
    assert all(r.counts[t] == 0 for t in ERROR_TYPES)
    assert r.ap50 == pytest.approx(1.0)


def random_case(seed, n_img=40, crowd_rate=0.05):
    rng = np.random.default_rng(seed)
    gt_rows, dets, crowd, aid = {}, [], set(), 0
    for i in range(1, n_img + 1):
        rows = []
        for _ in range(rng.integers(0, 8)):
            aid += 1
            b = [rng.uniform(0, 560), rng.uniform(0, 440), rng.uniform(4, 120), rng.uniform(4, 100)]
            rows.append((int(rng.choice([PERSON, CAR])), b))
            if rng.random() < crowd_rate:
                crowd.add(aid)
            # jittered, relabelled and duplicated detections around each GT
            for _ in range(rng.integers(0, 3)):
                j = np.array(b) + rng.normal(0, 0.25, 4) * np.array([b[2], b[3], b[2], b[3]])
                j[2:] = np.abs(j[2:]) + 1
                c = rows[-1][0] if rng.random() < 0.85 else (PERSON + CAR - rows[-1][0])
                dets.append(det(i, c, j, round(float(rng.random()), 2)))  # rounded: many score ties
        gt_rows[i] = rows
        for _ in range(rng.integers(0, 6)):  # background
            b = [rng.uniform(0, 560), rng.uniform(0, 440), rng.uniform(4, 80), rng.uniform(4, 80)]
            dets.append(det(i, int(rng.choice([PERSON, CAR])), b, round(float(rng.random()), 2)))
    return make_gt(gt_rows, crowd), dets


@pytest.mark.parametrize("seed", range(5))
def test_base_ap50_equals_pycocotools(seed):
    gt, dets = random_case(seed)
    r = tide(gt, dets)
    m = coco_map(gt, dets, class_names=["person", "car"])
    assert r.ap50 == pytest.approx(m["AP50"], abs=1e-12)
    assert r.ap50_per_class[PERSON] == pytest.approx(m["AP50_person"], abs=1e-12)
    assert r.ap50_per_class[CAR] == pytest.approx(m["AP50_car"], abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_oracles_are_consistent_on_random_data(seed):
    gt, dets = random_case(seed + 10, crowd_rate=0.0)
    r = tide(gt, dets)
    for t in ERROR_TYPES:
        assert r.delta[t] >= -1e-9, (t, r.delta)  # fixing errors never lowers AP
        assert r.ap50 + r.delta[t] <= 1 + 1e-9
    n_fp = sum(r.counts[t] for t in ERROR_TYPES if t != "Miss")
    assert n_fp + r.counts["TP"] == len(dets)  # every detection is a TP or exactly one FP type
    assert r.counts["TP"] + r.counts["Miss"] <= r.counts["GT"]
    # each FP oracle alone gains no more than suppressing all FPs plus fixing what it relabels or snaps
    assert r.delta["FP"] >= max(r.delta[t] for t in ("Dupe", "Bkg", "Both")) - 1e-9
    assert not math.isnan(r.ap50)
