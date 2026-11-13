"""Result Table B of the aligned FLIR v1 backbone ablation (F19, Roadmap Phase 2b).

Cells are ``cells[head][test_modality] → metrics`` (``coco_map`` over person + car, plus
``AP_person_car``/``AP50_person_car`` from ``with_person_car``). Heads are named
``aligned_<backbone>_<rgb|ir>``. Transfer ratios follow F15: a cross-modal cell ÷ the in-domain cell
of the **same backbone** on the **same test modality** (same frames, same GT: the ratio isolates the head).

``render_table_b_markdown`` turns a result (``evaluate_aligned`` output or ``eval_aligned.json``) into
the markdown tables of the results note. ``rescore_detections`` re-scores a saved detections file, so
the table can be reproduced exactly without running inference again (Roadmap F19 "Done when").
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ALIGNED_DATASET = "aligned_v1"
BACKBONES: tuple[str, ...] = ("dinopca", "dino3072", "mae")
BACKBONE_LABEL: dict[str, str] = {"dinopca": "DINOv2-Reg + PCA (1152)", "dino3072": "DINOv2-Reg 3072 → learned 1152",
                                  "mae": "MAE ViT-B/16 ViTDet (fine-tuned)"}
SUFFIX_MODALITY: dict[str, str] = {"rgb": "rgb", "ir": "thermal"}
MODALITY_LABEL: dict[str, str] = {"rgb": "RGB", "thermal": "IR"}
TEST_MODALITIES: tuple[str, ...] = ("rgb", "thermal")
RATIO_KEYS: tuple[str, ...] = ("AP_person_car", "AP50_person_car", "AP_person", "AP50_person", "AP_car", "AP50_car")
ALL_HEADS: tuple[str, ...] = tuple(f"aligned_{b}_{s}" for b in BACKBONES for s in SUFFIX_MODALITY)


def parse_aligned_head(name: str) -> tuple[str, str]:
    """``"aligned_dino3072_ir"`` → ``("dino3072", "thermal")``; ``ValueError`` on any other name."""
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "aligned" or parts[1] not in BACKBONES or parts[2] not in SUFFIX_MODALITY:
        raise ValueError(f"{name!r} is not an aligned head; expected one of {list(ALL_HEADS)}")
    return parts[1], SUFFIX_MODALITY[parts[2]]


def _ratio(a: float, b: float) -> float:
    return a / b if b and not math.isnan(b) and not math.isnan(a) else math.nan


def transfer_ratios(cells: Mapping[str, Mapping[str, Mapping[str, float]]],
                    keys: Sequence[str] = RATIO_KEYS) -> dict[str, dict[str, dict[str, float]]]:
    """``ratio[head][test_modality][key]`` = cross-modal cell ÷ the same backbone's in-domain cell there.

    Only cross-modal cells whose in-domain reference was also scored get a ratio.
    """
    in_domain = {}
    for h, per in cells.items():
        b, m = parse_aligned_head(h)
        if m in per:
            in_domain[(b, m)] = per[m]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for h, per in cells.items():
        b, train = parse_aligned_head(h)
        for m, metrics in per.items():
            if m == train or (b, m) not in in_domain:
                continue
            ref = in_domain[(b, m)]
            out.setdefault(h, {})[m] = {k: _ratio(metrics[k], ref[k]) for k in keys}
    return out


def score_aligned(gt: dict, detections: list[dict]) -> dict[str, float]:
    """``coco_map`` over the aligned label space (person, car) plus the person + car means."""
    from irwm.data.aligned import ALIGNED_CLASS_NAMES
    from irwm.detection import coco_val
    from irwm.detection.inference import with_person_car

    return with_person_car(coco_val.coco_map(gt, detections, class_names=ALIGNED_CLASS_NAMES))


def rescore_detections(path: Path, *, root: Path | None = None) -> dict[str, float]:
    """Metrics of a saved aligned detections file, re-scored against its split's GT (no inference).

    ``ValueError`` if the file is not an aligned v1 file (Roadmap R12).
    """
    from irwm.data.aligned import AlignedSplit
    from irwm.detection.inference import load_detections

    obj = load_detections(Path(path))
    if obj.get("dataset") != ALIGNED_DATASET:
        raise ValueError(f"{path}: dataset {obj.get('dataset')!r}, expected {ALIGNED_DATASET!r} (Roadmap R12)")
    split = AlignedSplit.load(obj["modality"], obj["split"], root)
    ids = [im["image_id"] for im in obj["images"]]
    if ids != [r.image_id for r in split.records]:  # a smoke run's first-N subset (evaluate_aligned max_images)
        try:
            split = split.subset(ids)
        except KeyError as e:
            raise ValueError(f"{path}: its images are not in aligned {obj['modality']}/{obj['split']}") from e
    return score_aligned(split.to_coco_gt("normalised"), obj["detections"])


# --------------------------------------------------------------------------- markdown
def _f(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.3f}" if round(v, 3) != 0 else "0.000"  # no "-0.000"


def _ordered_heads(cells: Mapping[str, Any]) -> list[str]:
    return [h for h in ALL_HEADS if h in cells] + sorted(h for h in cells if h not in ALL_HEADS)


def render_table_b_markdown(result: Mapping[str, Any]) -> str:
    """Markdown of Table B from ``result["cells"]`` (and ``result["transfer_ratio"]`` if present).

    Sections: the person + car AP / AP50 matrix (backbone × train → test modality, in-domain bold),
    paired H1 (IR − RGB in-domain per backbone), per-cell class and size breakdown, transfer ratios.
    Values are rounded to 3 decimals; missing cells are ``–``.
    """
    cells: Mapping[str, Mapping[str, Mapping[str, float]]] = result["cells"]
    ratios = result.get("transfer_ratio") or transfer_ratios(cells)
    heads = _ordered_heads(cells)
    mods = [m for m in TEST_MODALITIES if any(m in cells[h] for h in heads)]
    lines: list[str] = []

    def cell(h: str, m: str, key: str) -> str:
        v = cells[h].get(m, {}).get(key)
        s = _f(v)
        return f"**{s}**" if v is not None and parse_aligned_head(h)[1] == m else s

    for key, title in (("AP_person_car", "AP, person + car"), ("AP50_person_car", "AP50, person + car")):
        lines += [f"**{title}** (in-domain bold)", "",
                  "| Backbone | Train | " + " | ".join(f"{MODALITY_LABEL[m]} test" for m in mods) + " |",
                  "|---|---|" + "---|" * len(mods)]
        for h in heads:
            b, train = parse_aligned_head(h)
            lines.append(f"| {b} | {MODALITY_LABEL[train]} | " + " | ".join(cell(h, m, key) for m in mods) + " |")
        lines.append("")

    lines += ["**Paired H1**: in-domain IR − in-domain RGB (same frames, same boxes)", "",
              "| Backbone | AP p+c RGB→RGB | AP p+c IR→IR | Δ AP | AP50 p+c RGB→RGB | AP50 p+c IR→IR | Δ AP50 |",
              "|---|---|---|---|---|---|---|"]
    for b in BACKBONES:
        rgb = cells.get(f"aligned_{b}_rgb", {}).get("rgb")
        ir = cells.get(f"aligned_{b}_ir", {}).get("thermal")
        if rgb is None and ir is None:
            continue
        row = [b]
        for key in ("AP_person_car", "AP50_person_car"):
            a = rgb[key] if rgb else None
            c = ir[key] if ir else None
            row += [_f(a), _f(c), _f(c - a) if a is not None and c is not None else "–"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += ["**Per cell**: AP / AP50 per class and size buckets (640×512-normalised areas)", "",
              "| Backbone | Train | Test | person | car | AP | AP50 | APs | APm | APl |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for h in heads:
        b, train = parse_aligned_head(h)
        for m in mods:
            c = cells[h].get(m)
            if c is None:
                continue
            lines.append(f"| {b} | {MODALITY_LABEL[train]} | {MODALITY_LABEL[m]} | "
                         f"{_f(c.get('AP_person'))} / {_f(c.get('AP50_person'))} | "
                         f"{_f(c.get('AP_car'))} / {_f(c.get('AP50_car'))} | {_f(c.get('AP'))} | {_f(c.get('AP50'))} | "
                         f"{_f(c.get('APs'))} | {_f(c.get('APm'))} | {_f(c.get('APl'))} |")
    lines.append("")

    lines += ["**Transfer ratios**: cross-modal ÷ the same backbone's in-domain head on the same test modality", "",
              "| Cell | AP p+c | AP50 p+c | AP person | AP car |", "|---|---|---|---|---|"]
    any_ratio = False
    for h in heads:
        b, train = parse_aligned_head(h)
        for m, r in (ratios.get(h) or {}).items():
            ref = f"aligned_{b}_{'rgb' if m == 'rgb' else 'ir'}"
            lines.append(f"| `{h}` on {MODALITY_LABEL[m]} (÷ `{ref}`) | {_f(r.get('AP_person_car'))} | "
                         f"{_f(r.get('AP50_person_car'))} | {_f(r.get('AP_person'))} | {_f(r.get('AP_car'))} |")
            any_ratio = True
    if not any_ratio:
        lines.append("| (no cross-modal cell with its in-domain reference) | – | – | – | – |")
    return "\n".join(lines) + "\n"
