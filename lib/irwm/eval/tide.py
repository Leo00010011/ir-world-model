"""TIDE error decomposition for box detection (Bolya et al., ECCV 2020), numpy only.

Every false positive at IoU ``pos_thr`` (default 0.5) gets one error type, checked in this order
(the order of the reference ``tidecv`` implementation; IoU is the detection's best IoU with a GT box
of the given kind in the same image, ``bg_thr`` defaults to 0.1):

* **Loc**  best same-class IoU in ``[bg_thr, pos_thr)``: the right object, a bad box.
* **Cls**  best other-class IoU ``>= pos_thr``: a good box, the wrong class.
* **Dupe** best same-class IoU ``>= pos_thr``: its GT was already taken by a higher-scoring detection.
* **Bkg**  best IoU over all GT ``< bg_thr``: fired on background.
* **Both** anything else (a bad box on a GT of another class).

**Miss** = GT boxes left unmatched that no Cls or Loc error points at.

The weight of each type is ``dAP50``: the AP50 gain when an oracle fixes every error of that type and
nothing else, then the whole evaluation is re-run. Fixes (as in the paper): Cls → relabel to the GT's
class, Loc → snap the box to the GT, and in both cases suppress the detection instead if its GT is
already matched (by a TP or by an earlier-fixed detection); Dupe / Bkg / Both → suppress; Miss → drop
those GT boxes. Two summary oracles: **FP** suppresses every false positive, **FN** drops every
unmatched GT. The dAPs overlap, so they do not sum to ``1 - AP50``.

Matching and AP reproduce pycocotools' ``COCOeval`` (bbox, area "all", ``maxDets`` per image and
class, 101-point interpolation, crowd GT as ignore regions), so ``ap50`` equals ``coco_map``'s AP50.
Detections matched to a crowd region are ignored: they are neither TP nor an error.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

ERROR_TYPES = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
SPECIAL_TYPES = ("FP", "FN")
_REC_THRS = np.linspace(0.0, 1.0, 101)


@dataclass
class TideResult:
    """``ap50`` and every ``delta`` are on 0–1 (multiply by 100 for TIDE's usual points)."""

    ap50: float  # mean over classes with GT (COCO AP50)
    ap50_per_class: dict[int, float]
    delta: dict[str, float]  # ERROR_TYPES + SPECIAL_TYPES -> dAP50 (mean over classes)
    delta_per_class: dict[str, dict[int, float]]
    counts: dict[str, int]  # ERROR_TYPES -> number of errors; also "TP", "GT"
    counts_per_class: dict[str, dict[int, int]] = field(default_factory=dict)


# --------------------------------------------------------------------------- geometry
def _xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float).reshape(-1, 4)
    return np.concatenate([b[:, :2], b[:, :2] + b[:, 2:]], axis=1)


def box_iou(dt: np.ndarray, gt: np.ndarray, crowd: np.ndarray | None = None) -> np.ndarray:
    """``[D, G]`` IoU of COCO ``xywh`` boxes, as ``pycocotools.mask.iou`` computes it.

    For a crowd GT the denominator is the detection's own area (intersection over detection).
    """
    dt, gt = np.asarray(dt, float).reshape(-1, 4), np.asarray(gt, float).reshape(-1, 4)
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    a, b = _xyxy(dt), _xyxy(gt)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    da = (dt[:, 2] * dt[:, 3])[:, None]
    ga = (gt[:, 2] * gt[:, 3])[None, :]
    union = da + ga - inter
    if crowd is not None:
        union = np.where(np.asarray(crowd, bool)[None, :], da, union)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


# --------------------------------------------------------------------------- COCO evaluation
def _match(dt_boxes, dt_scores, gt_boxes, gt_crowd, thr):
    """COCOeval.evaluateImg for one (image, class) and one IoU threshold.

    Detections must already be sorted by score (stable, descending) and truncated to maxDets.
    Returns (dt_gt [D] index of matched GT or -1, dt_ignored [D] bool).
    """
    D, G = len(dt_boxes), len(gt_boxes)
    dt_gt = np.full(D, -1, int)
    dt_ig = np.zeros(D, bool)
    if D == 0 or G == 0:
        return dt_gt, dt_ig
    # COCO sorts GT with ignored (crowd) last; a non-crowd match always wins over a crowd one.
    order = np.argsort(gt_crowd.astype(int), kind="mergesort")
    ious = box_iou(dt_boxes, gt_boxes[order], gt_crowd[order])
    crowd = gt_crowd[order]
    taken = np.zeros(G, bool)
    for d in range(D):
        best, m = min(thr, 1 - 1e-10), -1
        for g in range(G):
            if taken[g] and not crowd[g]:
                continue
            if m > -1 and not crowd[m] and crowd[g]:
                break
            if ious[d, g] < best:
                continue
            best, m = ious[d, g], g
        if m == -1:
            continue
        dt_ig[d] = crowd[m]
        taken[m] = True
        dt_gt[d] = order[m]
    return dt_gt, dt_ig


def _ap(scores: np.ndarray, tp: np.ndarray, ignored: np.ndarray, n_gt: int) -> float:
    """COCOeval.accumulate for one class (area all, maxDets fixed), 101-point AP; -1 if no GT."""
    if n_gt == 0:
        return -1.0
    if len(scores) == 0:
        return 0.0
    inds = np.argsort(-scores, kind="mergesort")
    tp, ig = tp[inds], ignored[inds]
    tps = np.cumsum(tp & ~ig).astype(float)
    fps = np.cumsum(~tp & ~ig).astype(float)
    rc = tps / n_gt
    pr = tps / (fps + tps + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]  # the same monotone envelope as COCO's backward loop
    q = np.zeros(len(_REC_THRS))
    idx = np.searchsorted(rc, _REC_THRS, side="left")
    ok = idx < len(pr)
    q[ok] = pr[idx[ok]]
    return float(q.mean())


class _Data:
    """GT and detections indexed by image, in arrays."""

    def __init__(self, gt: Mapping, detections: Iterable[Mapping], max_dets: int):
        self.cats = sorted(int(c["id"]) for c in gt["categories"])
        self.img_ids = sorted(int(i["id"]) for i in gt["images"])
        known = set(self.img_ids)
        g = defaultdict(list)
        for a in gt["annotations"]:
            g[int(a["image_id"])].append((int(a["category_id"]), a["bbox"], int(bool(a.get("iscrowd", 0)))))
        self.gt = {}
        for i in self.img_ids:
            rows = g.get(i, [])
            self.gt[i] = (np.array([r[0] for r in rows], int), np.array([r[1] for r in rows], float).reshape(-1, 4),
                          np.array([r[2] for r in rows], bool))
        d = defaultdict(list)
        for det in detections:
            i = int(det["image_id"])
            if i not in known:
                raise ValueError(f"detection on image_id {i}, which is not in the GT")
            d[(i, int(det["category_id"]))].append((float(det["score"]), det["bbox"]))
        # per (image, class): sorted by score (stable, as COCO) and truncated to max_dets
        self.dt = {}
        for key, rows in d.items():
            s = np.array([r[0] for r in rows], float)
            o = np.argsort(-s, kind="mergesort")[:max_dets]
            self.dt[key] = (s[o], np.array([rows[k][1] for k in o], float).reshape(-1, 4))


def _evaluate(gt: Mapping[int, tuple], dt: Mapping[tuple, tuple], img_ids, cats, thr):
    """Match everything; returns (per-class AP dict, per (image, class) match arrays)."""
    per_cls = {c: ([], [], []) for c in cats}
    n_gt = {c: 0 for c in cats}
    matches = {}
    for i in img_ids:
        g_lab, g_box, g_crowd = gt[i]
        for c in cats:
            gm = g_lab == c
            n_gt[c] += int((gm & ~g_crowd).sum())
            if (i, c) not in dt:
                continue
            s, b = dt[(i, c)]
            dt_gt, dt_ig = _match(b, s, g_box[gm], g_crowd[gm], thr)
            gidx = np.flatnonzero(gm)
            dt_gt_full = np.full(len(dt_gt), -1, int)
            dt_gt_full[dt_gt >= 0] = gidx[dt_gt[dt_gt >= 0]]
            matches[(i, c)] = (dt_gt_full, dt_ig)
            per_cls[c][0].append(s)
            per_cls[c][1].append(dt_gt >= 0)
            per_cls[c][2].append(dt_ig)
    ap = {}
    for c in cats:
        s, tp, ig = (np.concatenate(x) if x else np.zeros(0, t) for x, t in zip(per_cls[c], (float, bool, bool)))
        ap[c] = _ap(s, tp, ig, n_gt[c])
    return ap, matches


def _mean(ap: Mapping[int, float]) -> float:
    v = [x for x in ap.values() if x > -1]
    return float(np.mean(v)) if v else -1.0


# --------------------------------------------------------------------------- TIDE
def tide(gt: Mapping, detections: Sequence[Mapping], *, pos_thr: float = 0.5, bg_thr: float = 0.1,
         max_dets: int = 100) -> TideResult:
    """TIDE on a COCO GT dict and a COCO results list (``image_id``, ``category_id``, ``bbox`` xywh, ``score``)."""
    data = _Data(gt, detections, max_dets)
    cats, img_ids = data.cats, data.img_ids
    base_ap, matches = _evaluate(data.gt, data.dt, img_ids, cats, pos_thr)

    # ---- classify every false positive, and record the GT each Cls / Loc error points at
    errors: list[tuple] = []  # (type, image, class, det index in data.dt[(i, c)], score, target GT index or -1)
    gt_used = {i: np.zeros(len(data.gt[i][0]), bool) for i in img_ids}
    for (i, c), (dt_gt, _) in matches.items():
        gt_used[i][dt_gt[dt_gt >= 0]] = True
    counts_cls = {t: defaultdict(int) for t in ERROR_TYPES}
    targeted = {i: np.zeros(len(data.gt[i][0]), bool) for i in img_ids}
    for (i, c), (s, b) in data.dt.items():
        dt_gt, dt_ig = matches[(i, c)]
        g_lab, g_box, g_crowd = data.gt[i]
        real = ~g_crowd
        iou = box_iou(b, g_box)  # plain IoU, crowd regions are not objects
        same = (g_lab == c) & real
        other = (g_lab != c) & real
        for d in np.flatnonzero((dt_gt < 0) & ~dt_ig):
            row = iou[d]
            m_same = row[same].max() if same.any() else 0.0
            m_other = row[other].max() if other.any() else 0.0
            m_all = row[real].max() if real.any() else 0.0
            target = -1
            if bg_thr <= m_same < pos_thr:
                kind = "Loc"
                target = int(np.flatnonzero(same)[row[same].argmax()])
            elif m_other >= pos_thr:
                kind = "Cls"
                target = int(np.flatnonzero(other)[row[other].argmax()])
            elif m_same >= pos_thr:
                kind = "Dupe"
            elif m_all < bg_thr:
                kind = "Bkg"
            else:
                kind = "Both"
            if target >= 0:
                targeted[i][target] = True
            errors.append((kind, i, c, int(d), float(s[d]), target))
            counts_cls[kind][c] += 1
    missed = {i: (~gt_used[i]) & (~data.gt[i][2]) & (~targeted[i]) for i in img_ids}
    unmatched = {i: (~gt_used[i]) & (~data.gt[i][2]) for i in img_ids}
    for i in img_ids:
        for lab in data.gt[i][0][missed[i]]:
            counts_cls["Miss"][int(lab)] += 1

    # ---- oracles
    def run(fix_types: set[str], drop_gt: Mapping[int, np.ndarray] | None = None) -> dict[int, float]:
        suppress: set[tuple] = set()
        moved: dict[tuple, list] = defaultdict(list)  # new (image, class) -> [(score, box)]
        claimed = {i: gt_used[i].copy() for i in img_ids}
        for kind, i, c, d, score, target in sorted(errors, key=lambda e: -e[4]):  # fix in score order
            if kind not in fix_types:
                continue
            key = (i, c, d)
            suppress.add(key)
            if kind in ("Cls", "Loc"):
                if claimed[i][target]:
                    continue  # its GT is already matched: the fix is to suppress it
                claimed[i][target] = True
                new_c = int(data.gt[i][0][target])
                box = data.gt[i][1][target] if kind == "Loc" else data.dt[(i, c)][1][d]
                moved[(i, new_c)].append((score, box))
        dt = {}
        for (i, c), (s, b) in data.dt.items():
            keep = np.array([(i, c, d) not in suppress for d in range(len(s))], bool)
            dt[(i, c)] = (s[keep], b[keep])
        for (i, c), rows in moved.items():
            s0, b0 = dt.get((i, c), (np.zeros(0), np.zeros((0, 4))))
            s1 = np.concatenate([s0, [r[0] for r in rows]])
            b1 = np.concatenate([b0, np.array([r[1] for r in rows], float).reshape(-1, 4)])
            o = np.argsort(-s1, kind="mergesort")[:max_dets]
            dt[(i, c)] = (s1[o], b1[o])
        g = data.gt
        if drop_gt is not None:
            g = {i: tuple(x[~drop_gt[i]] for x in data.gt[i]) for i in img_ids}
        return _evaluate(g, dt, img_ids, cats, pos_thr)[0]

    fixed = {t: run({t}) for t in ("Cls", "Loc", "Both", "Dupe", "Bkg")}
    fixed["Miss"] = run(set(), drop_gt=missed)

    # FP oracle: suppress every false positive (no relabel / snap), i.e. keep only the TPs.
    dt_tp = {}
    for (i, c), (s, b) in data.dt.items():
        dt_gt, dt_ig = matches[(i, c)]
        keep = (dt_gt >= 0) | dt_ig
        dt_tp[(i, c)] = (s[keep], b[keep])
    fixed["FP"] = _evaluate(data.gt, dt_tp, img_ids, cats, pos_thr)[0]
    fixed["FN"] = run(set(), drop_gt=unmatched)

    delta_cls = {t: {c: (fixed[t][c] - base_ap[c]) if base_ap[c] > -1 else float("nan") for c in cats}
                 for t in ERROR_TYPES + SPECIAL_TYPES}
    delta = {t: _mean(fixed[t]) - _mean(base_ap) for t in ERROR_TYPES + SPECIAL_TYPES}
    counts = {t: int(sum(counts_cls[t].values())) for t in ERROR_TYPES}
    counts["TP"] = int(sum(int((dt_gt >= 0).sum() - dt_ig[dt_gt >= 0].sum()) for dt_gt, dt_ig in matches.values()))
    counts["GT"] = int(sum(int((~data.gt[i][2]).sum()) for i in img_ids))
    return TideResult(ap50=_mean(base_ap), ap50_per_class=dict(base_ap), delta=delta, delta_per_class=delta_cls,
                      counts=counts, counts_per_class={t: dict(v) for t, v in counts_cls.items()})
