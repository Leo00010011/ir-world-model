"""FLIR ADAS v2 dataset statistics and the class-imbalance study behind the 6-class choice.

Spec: ``spec/2026-10-08-flir-readers-class-mapping`` (FR8, FR9). Pure functions over
loaded ``FlirSplit``s, so they are testable on synthetic splits. The entry point that
writes the report is ``projects/data_stats/stats.py``.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from irwm.data.classes import CLASS_NAMES, LABEL_TO_CATEGORY_ID, normalised_area, size_bucket
from irwm.data.flir import _PIL_MODE, FlirSplit

# FLIR name → COCO-80 name, judged by meaning, not by id: FLIR reuses COCO-80 slot
# numbers but renames some (slot 12 "stop sign" → "sign", slot 18 "horse" → "deer").
COCO80_NAME: dict[str, str | None] = {
    "person": "person",
    "bike": "bicycle",
    "car": "car",
    "motor": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "light": "traffic light",
    "hydrant": "fire hydrant",
    "dog": "dog",
    "skateboard": "skateboard",
    "sign": None,  # COCO has only "stop sign"
    "deer": None,
    "stroller": None,
    "scooter": None,
    "other vehicle": None,
}

# Reporting flag only: < 1 % of the split's boxes, or < 100 boxes in train.
_UNDERREP_PCT = 1.0
_UNDERREP_TRAIN_BOXES = 100


def _key(s: FlirSplit) -> str:
    return f"{s.modality}/{s.split}"


def _coco80(name: str) -> str | None:
    if name not in COCO80_NAME:
        raise KeyError(f"FLIR class {name!r} is not in the COCO80_NAME table")
    return COCO80_NAME[name]


def _sml() -> dict[str, int]:
    return {"S": 0, "M": 0, "L": 0}


def _segments(frames: list[int]) -> tuple[list[int], list[int]]:
    """Gap sizes (consecutive sorted indices differing by > 1) and contiguous segment lengths."""
    gaps, segments, run = [], [], 1
    for a, b in zip(frames, frames[1:]):
        if b - a > 1:
            gaps.append(b - a)
            segments.append(run)
            run = 1
        else:
            run += 1
    if frames:
        segments.append(run)
    return gaps, segments


def _file_integrity(s: FlirSplit) -> dict[str, Any]:
    mode = _PIL_MODE[s.modality]
    size_mm, mode_mm = [], []
    for r in s:
        with Image.open(r.path) as im:  # header only, no .load()
            if im.size != (r.width, r.height):
                size_mm.append({"image_id": r.image_id, "file_name": r.file_name,
                                "coco": [r.width, r.height], "file": list(im.size)})
            if im.mode != mode:
                mode_mm.append({"image_id": r.image_id, "file_name": r.file_name,
                                "mode": im.mode, "expected": mode})
    known = {Path(r.file_name).name for r in s}
    orphans = sorted(set(os.listdir(s.split_dir / "data")) - known)
    return {
        "n_checked": len(s),
        "size_mismatches": size_mm,
        "mode_mismatches": mode_mm,
        "orphans": [f"{s.split_dir.name}/data/{n}" for n in orphans],
    }


def compute_stats(splits: Sequence[FlirSplit], *, check_files: bool = True) -> dict:
    """Statistics report over the passed (modality, split)s, as a JSON-serialisable dict.

    Sections (FR8): ``images``, ``classes`` (one row per raw class × modality × split),
    ``modality_comparison``, ``selected``, ``sequences`` and, with ``check_files``,
    ``file_integrity`` (JPEG headers + orphans). Raises ``KeyError`` for a FLIR class
    missing from ``COCO80_NAME``.
    """
    keys = [_key(s) for s in splits]
    images: dict[str, dict] = {}
    # (key, category_id) → counters
    boxes: Counter = Counter()
    img_count: Counter = Counter()
    sml_native: dict[tuple[str, int], dict[str, int]] = defaultdict(_sml)
    sml_norm: dict[tuple[str, int], dict[str, int]] = defaultdict(_sml)
    total_boxes: Counter = Counter()
    names: dict[int, str] = {}

    for s in splits:
        k = _key(s)
        names.update(s._category_names)
        sizes: Counter = Counter()
        no_kept = no_ann = 0
        for r in s:
            sizes[f"{r.width}x{r.height}"] += 1
            no_kept += len(r.labels) == 0
            no_ann += len(r.labels) == 0 and len(r.ignore_category_ids) == 0
            cats = [LABEL_TO_CATEGORY_ID[int(l)] for l in r.labels] + [
                int(c) for c in r.ignore_category_ids
            ]
            all_boxes = list(r.boxes) + list(r.ignore_boxes)
            for cid in set(cats):
                img_count[(k, cid)] += 1
            for cid, b in zip(cats, all_boxes):
                w, h = float(b[2]), float(b[3])
                boxes[(k, cid)] += 1
                total_boxes[k] += 1
                sml_native[(k, cid)][size_bucket(w * h)] += 1
                sml_norm[(k, cid)][size_bucket(normalised_area(w, h, r.width, r.height))] += 1
        images[k] = {
            "n_images": len(s),
            "sizes": dict(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))),
            "no_kept_box": no_kept,
            "no_annotation": no_ann,
        }

    present = sorted({cid for (_, cid), n in boxes.items() if n > 0})
    selected_ids = set(LABEL_TO_CATEGORY_ID.values())
    by_mod: dict[str, list[FlirSplit]] = defaultdict(list)
    for s in splits:
        by_mod[s.modality].append(s)

    def n_boxes(modality: str, split: str, cid: int) -> int | None:
        k = f"{modality}/{split}"
        return boxes[(k, cid)] if k in keys else None

    classes = []
    for cid in present:
        name = names[cid]
        coco80 = _coco80(name)
        for s in splits:
            k = _key(s)
            n = boxes[(k, cid)]
            share = 100.0 * n / total_boxes[k] if total_boxes[k] else 0.0
            n_train = n_boxes(s.modality, "train", cid)
            n_val = n_boxes(s.modality, "val", cid)
            n_test = n_boxes(s.modality, "test", cid)
            classes.append({
                "name": name,
                "category_id": cid,
                "modality": s.modality,
                "split": s.split,
                "boxes": n,
                "images": img_count[(k, cid)],
                "share_pct": round(share, 4),
                "size_native": dict(sml_native[(k, cid)]),
                "size_normalised": dict(sml_norm[(k, cid)]),
                "selected": cid in selected_ids,
                "coco80": coco80 is not None,
                "underrepresented": share < _UNDERREP_PCT
                or (n_train is not None and n_train < _UNDERREP_TRAIN_BOXES),
                "absent_val": None if n_val is None else n_val == 0,
                "absent_test": None if n_test is None else n_test == 0,
            })

    split_names = sorted({s.split for s in splits}, key=["train", "val", "test"].index)
    modality_comparison: dict[str, dict[str, float | None]] = {}
    for cid in present:
        row = {}
        for sp in split_names:
            rgb, th = n_boxes("rgb", sp, cid), n_boxes("thermal", sp, cid)
            if rgb is None or th is None:
                continue
            row[sp] = round(th / rgb, 4) if rgb else None
        modality_comparison[names[cid]] = row

    selected = {}
    for s in splits:
        k = _key(s)
        kept = {c: boxes[(k, LABEL_TO_CATEGORY_ID[i])] for i, c in enumerate(CLASS_NAMES)}
        ignore = {names[cid]: boxes[(k, cid)] for cid in present if cid not in selected_ids}
        selected[k] = {
            "kept": kept,
            "ignore": ignore,
            "total_kept": sum(kept.values()),
            "total_ignore": sum(ignore.values()),
        }

    sequences: dict[str, list[dict]] = {}
    for s in splits:
        if s.split != "test":
            continue
        tags = s.video_tags()
        frames: dict[str, list[int]] = defaultdict(list)
        for r in s:
            frames[r.video_id].append(r.frame_index)
        seqs = []
        for vid in sorted(frames):
            fr = sorted(frames[vid])
            gaps, segments = _segments(fr)
            seqs.append({
                "video_id": vid,
                "n_frames": len(fr),
                "frame_min": fr[0],
                "frame_max": fr[-1],
                "n_gaps": len(gaps),
                "gap_sizes": gaps,
                "segments": segments,
                "tags": tags.get(vid, []),
            })
        sequences[s.modality] = seqs

    out: dict[str, Any] = {
        "splits": keys,
        "images": images,
        "classes": classes,
        "modality_comparison": modality_comparison,
        "selected": selected,
        "sequences": sequences,
    }
    if check_files:
        out["file_integrity"] = {_key(s): _file_integrity(s) for s in splits}
    return out


def _table(header: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "yes" if v else "no"
        if v is None:
            return "–"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


_CLASS_HEADER = ["class", "id", "modality", "split", "boxes", "images", "share %",
                 "S/M/L native", "S/M/L normalised", "selected", "coco80",
                 "underrepresented", "absent_val", "absent_test"]


def _class_row(c: dict) -> list[Any]:
    sml = lambda d: f"{d['S']} / {d['M']} / {d['L']}"  # noqa: E731
    return [c["name"], c["category_id"], c["modality"], c["split"], c["boxes"], c["images"],
            float(c["share_pct"]), sml(c["size_native"]), sml(c["size_normalised"]),
            c["selected"], c["coco80"], c["underrepresented"], c["absent_val"], c["absent_test"]]


def render_markdown(stats: dict) -> str:
    """Render ``compute_stats`` output as GitHub-flavoured Markdown tables."""
    parts = ["# FLIR ADAS v2 statistics"]

    parts.append("## 1. Images")
    parts.append(_table(
        ["split", "images", "no kept box", "no annotation", "sizes (W×H: n)"],
        [[k, v["n_images"], v["no_kept_box"], v["no_annotation"],
          ", ".join(f"{s}: {n}" for s, n in v["sizes"].items())]
         for k, v in stats["images"].items()],
    ))

    parts.append("## 2. Per raw FLIR class")
    parts.append(
        "Size buckets use COCO thresholds (32², 96²); *normalised* areas are rescaled to a "
        "640×512 frame. *underrepresented*: < 1 % of the split's boxes or < 100 train boxes."
    )
    parts.append(_table(_CLASS_HEADER, [_class_row(c) for c in stats["classes"]]))

    parts.append("## 3. Modality comparison (thermal / RGB box count)")
    split_cols = sorted({sp for row in stats["modality_comparison"].values() for sp in row},
                        key=["train", "val", "test"].index)
    parts.append(_table(
        ["class", *split_cols],
        [[name, *(row.get(sp) for sp in split_cols)]
         for name, row in stats["modality_comparison"].items()],
    ))

    parts.append("## 4. Selected 6 + ignore")
    sel = stats["selected"]
    keys = list(sel)
    ignore_names = sorted({n for v in sel.values() for n in v["ignore"]})
    rows = [[c, "kept", *(sel[k]["kept"][c] for k in keys)] for c in CLASS_NAMES]
    rows += [[n, "ignore", *(sel[k]["ignore"].get(n, 0) for k in keys)] for n in ignore_names]
    rows.append(["total", "kept", *(sel[k]["total_kept"] for k in keys)])
    rows.append(["total", "ignore", *(sel[k]["total_ignore"] for k in keys)])
    parts.append(_table(["class", "role", *keys], rows))

    parts.append("## 5. Video sequences (test)")
    rows = [[mod, q["video_id"], q["n_frames"], q["frame_min"], q["frame_max"], q["n_gaps"],
             ", ".join(map(str, q["gap_sizes"])) or "–", ", ".join(map(str, q["segments"])),
             ", ".join(q["tags"])]
            for mod, seqs in stats["sequences"].items() for q in seqs]
    parts.append(_table(["modality", "video_id", "frames", "min", "max", "gaps", "gap sizes",
                         "segments", "tags"], rows))

    parts.append("## 6. File integrity")
    if "file_integrity" not in stats:
        parts.append("Not checked (`check_files=False`).")
    else:
        fi = stats["file_integrity"]
        parts.append(_table(
            ["split", "checked", "size mismatches", "mode mismatches", "orphans"],
            [[k, v["n_checked"], len(v["size_mismatches"]), len(v["mode_mismatches"]),
              len(v["orphans"])] for k, v in fi.items()],
        ))
        details = [f"- size mismatch `{k}`: {m}" for k, v in fi.items() for m in v["size_mismatches"]]
        details += [f"- mode mismatch `{k}`: {m}" for k, v in fi.items() for m in v["mode_mismatches"]]
        details += [f"- orphan: `{o}`" for v in fi.values() for o in v["orphans"]]
        if details:
            parts.append("\n".join(details))

    parts.append("## 6-class summary")
    parts.append(_table(_CLASS_HEADER, [_class_row(c) for c in stats["classes"] if c["selected"]]))

    return "\n\n".join(parts) + "\n"
