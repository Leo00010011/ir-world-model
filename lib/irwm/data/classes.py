"""The project's single detection label space and its size buckets.

Decided in ``spec/2026-10-08-flir-readers-class-mapping`` (FR2, FR3). Every project
imports these; nobody redefines them (TechStack §3.4).

- **6 FLIR classes:** person, bike, car, motor, light, sign. Contiguous labels ``0..5``
  follow ascending FLIR category id. There is no coarse mapping.
- **Ignore rule:** a box of any other FLIR category (truck, bus, other vehicle, …) is an
  ignore region, neither a target nor background. ``label_of`` returns ``None`` for it.
- **Size buckets:** areas are normalised to a 640×512 frame,
  ``w·h·(640·512)/(W·H)``, then bucketed with the COCO thresholds 32² and 96². Most RGB
  frames have several times the pixels of thermal ones, so native buckets would compare
  different object populations. Thermal (640×512) is unchanged by the normalisation.

Pure Python, no numpy, so any module can import it.
"""

from __future__ import annotations

from typing import Literal

CLASS_NAMES: tuple[str, ...] = ("person", "bike", "car", "motor", "light", "sign")
NUM_CLASSES: int = len(CLASS_NAMES)

# FLIR category id (as in coco.json) → contiguous label.
CATEGORY_ID_TO_LABEL: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 12: 5}
LABEL_TO_CATEGORY_ID: dict[int, int] = {v: k for k, v in CATEGORY_ID_TO_LABEL.items()}

REF_WIDTH, REF_HEIGHT = 640, 512


def label_of(category_id: int) -> int | None:
    """Contiguous label ``0..5`` of a FLIR category id, or ``None`` for an ignore region."""
    return CATEGORY_ID_TO_LABEL.get(category_id)


def normalised_area(w: float, h: float, img_w: int, img_h: int) -> float:
    """Box area ``w·h`` rescaled to a 640×512 frame: ``w·h·(640·512)/(img_w·img_h)``."""
    return w * h * (REF_WIDTH * REF_HEIGHT) / (img_w * img_h)


def size_bucket(area: float) -> Literal["S", "M", "L"]:
    """COCO size bucket of an (already normalised) area: S < 32², M < 96², else L."""
    if area < 32**2:
        return "S"
    if area < 96**2:
        return "M"
    return "L"
