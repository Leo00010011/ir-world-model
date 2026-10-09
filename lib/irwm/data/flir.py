"""Read-only FLIR ADAS v2 readers: split index, image dataset and video metadata.

Spec: ``spec/2026-10-08-flir-readers-class-mapping`` (FR1, FR4–FR7). The data is read
from the as-downloaded layout::

    <root>/FLIR_ADAS_v2/<split_dir>/{coco.json, index.json, data/*.jpg}
    <root>/rgb_to_thermal_vid_map.json

with ``<root> = $IRWM_DATA_ROOT/<FLIR_DIRNAME>``. Annotations come from ``coco.json``.
Boxes of the 6 classes in ``irwm.data.classes`` are targets, and every other box is an
ignore region. ``image_id`` is unique only within one (modality, split), so every lookup
goes through a ``FlirSplit``. Pixels are returned at native resolution as ``uint8``;
resizing and 1→3 channel replication belong to the encoder (F4).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from irwm.data.classes import CATEGORY_ID_TO_LABEL, CLASS_NAMES, label_of, normalised_area
from irwm.utils.paths import data_root

Modality = Literal["rgb", "thermal"]
Split = Literal["train", "val", "test"]

# Folder name exactly as on disk. Also in conf/flir/default.yaml.
FLIR_DIRNAME = "teledyne_flir_adas_v2"
_DATASET_DIR = "FLIR_ADAS_v2"
_MAP_FILE = "rgb_to_thermal_vid_map.json"

_SPLIT_DIRS: dict[tuple[str, str], str] = {
    ("rgb", "train"): "images_rgb_train",
    ("rgb", "val"): "images_rgb_val",
    ("rgb", "test"): "video_rgb_test",
    ("thermal", "train"): "images_thermal_train",
    ("thermal", "val"): "images_thermal_val",
    ("thermal", "test"): "video_thermal_test",
}
# PIL mode each modality's JPEGs must decode to (no silent convert()).
_PIL_MODE: dict[str, str] = {"rgb": "RGB", "thermal": "L"}

_FILE_RE = re.compile(
    r"^data/video-(?P<video_id>[A-Za-z0-9]+)-frame-(?P<frame>\d{6})-[A-Za-z0-9]+\.jpg$"
)


def flir_root() -> Path:
    """``$IRWM_DATA_ROOT/<FLIR_DIRNAME>``, the as-downloaded dataset root."""
    return data_root() / FLIR_DIRNAME


def split_dir(modality: Modality, split: Split, root: Path | None = None) -> Path:
    """Directory of one (modality, split), e.g. ``<root>/FLIR_ADAS_v2/video_thermal_test``.

    Raises ``ValueError`` for an unknown modality or split. Existence is not checked.
    """
    if modality not in get_args(Modality):
        raise ValueError(f"Unknown modality {modality!r}; expected one of {list(get_args(Modality))}")
    if split not in get_args(Split):
        raise ValueError(f"Unknown split {split!r}; expected one of {list(get_args(Split))}")
    root = flir_root() if root is None else Path(root)
    return root / _DATASET_DIR / _SPLIT_DIRS[(modality, split)]


def load_rgb_thermal_map(root: Path | None = None) -> dict[str, str]:
    """RGB test basename → thermal test basename, from ``rgb_to_thermal_vid_map.json``.

    No validation; the consistency check belongs to the clip index (F3).
    """
    root = flir_root() if root is None else Path(root)
    with open(root / _MAP_FILE, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class FlirRecord:
    """One ``coco.json`` image with its parsed annotations.

    ``boxes``/``labels``/``ann_ids``/``track_ids`` hold the 6 kept classes and
    ``ignore_boxes``/``ignore_category_ids`` every other box, each in ``coco.json`` order.
    Boxes are COCO ``xywh`` in native pixels. ``track_ids`` is ``-1`` where the annotation
    has none (every image split).
    """

    image_id: int
    file_name: str  # as in coco.json, e.g. "data/video-…jpg"
    path: Path  # absolute
    width: int
    height: int
    video_id: str
    frame_index: int
    boxes: np.ndarray  # float32 [N, 4]
    labels: np.ndarray  # int64 [N], 0..5
    ann_ids: np.ndarray  # int64 [N]
    track_ids: np.ndarray  # int64 [N]
    ignore_boxes: np.ndarray  # float32 [M, 4]
    ignore_category_ids: np.ndarray  # int64 [M], raw FLIR ids


def _boxes(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def _ints(values: list[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


class FlirSplit:
    """Parsed metadata of one (modality, split): records sorted by ``image_id``, no pixels."""

    def __init__(
        self, modality: Modality, split: Split, root: Path, records: list[FlirRecord]
    ) -> None:
        self.modality = modality
        self.split = split
        self.root = root
        self.split_dir = split_dir(modality, split, root)
        self.records = records
        self._by_id = {r.image_id: r for r in records}
        # Raw coco.json id → name, for the statistics report (irwm.data.stats).
        self._category_names: dict[int, str] = {}

    @classmethod
    def load(cls, modality: Modality, split: Split, root: Path | None = None) -> FlirSplit:
        """Parse ``<split_dir>/coco.json`` once. No image I/O.

        Raises ``FileNotFoundError`` if ``coco.json`` is missing, and ``ValueError`` on a
        malformed file name, a ``video_id`` mismatch, a duplicate ``image_id``, an unknown
        ``category_id`` or a box with ``w <= 0`` or ``h <= 0``.
        """
        root = flir_root() if root is None else Path(root)
        sdir = split_dir(modality, split, root)
        coco_path = sdir / "coco.json"
        if not coco_path.is_file():
            raise FileNotFoundError(f"FLIR coco.json not found: {coco_path}")
        with open(coco_path, encoding="utf-8") as f:
            coco = json.load(f)

        cat_ids = {c["id"] for c in coco["categories"]}
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] not in cat_ids:
                raise ValueError(
                    f"{coco_path}: annotation {ann['id']} has category_id {ann['category_id']} "
                    "not in categories"
                )
            if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
                raise ValueError(f"{coco_path}: annotation {ann['id']} has a degenerate bbox {ann['bbox']}")
            anns_by_image[ann["image_id"]].append(ann)

        records: list[FlirRecord] = []
        seen: set[int] = set()
        for img in sorted(coco["images"], key=lambda im: im["id"]):
            image_id = img["id"]
            if image_id in seen:
                raise ValueError(f"{coco_path}: duplicate image_id {image_id}")
            seen.add(image_id)
            m = _FILE_RE.match(img["file_name"])
            if m is None:
                raise ValueError(
                    f"{coco_path}: image {image_id} file_name {img['file_name']!r} "
                    f"does not match {_FILE_RE.pattern}"
                )
            extra_vid = img.get("extra_info", {}).get("video_id")
            if extra_vid != m["video_id"]:
                raise ValueError(
                    f"{coco_path}: image {image_id} extra_info.video_id {extra_vid!r} != "
                    f"{m['video_id']!r} parsed from file_name"
                )
            boxes, labels, ann_ids, track_ids, ign_boxes, ign_cats = [], [], [], [], [], []
            for ann in anns_by_image.get(image_id, []):
                label = label_of(ann["category_id"])
                if label is None:
                    ign_boxes.append(ann["bbox"])
                    ign_cats.append(ann["category_id"])
                else:
                    boxes.append(ann["bbox"])
                    labels.append(label)
                    ann_ids.append(ann["id"])
                    track_ids.append(ann.get("track_id", -1))
            records.append(
                FlirRecord(
                    image_id=image_id,
                    file_name=img["file_name"],
                    path=(sdir / img["file_name"]).resolve(),
                    width=img["width"],
                    height=img["height"],
                    video_id=m["video_id"],
                    frame_index=int(m["frame"]),
                    boxes=_boxes(boxes),
                    labels=_ints(labels),
                    ann_ids=_ints(ann_ids),
                    track_ids=_ints(track_ids),
                    ignore_boxes=_boxes(ign_boxes),
                    ignore_category_ids=_ints(ign_cats),
                )
            )
        out = cls(modality, split, root, records)
        out._category_names = {c["id"]: c["name"] for c in coco["categories"]}
        return out

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[FlirRecord]:
        return iter(self.records)

    def by_image_id(self, image_id: int) -> FlirRecord:
        """The record with this ``image_id``. Raises ``KeyError`` if it is not in this split."""
        try:
            return self._by_id[image_id]
        except KeyError:
            raise KeyError(
                f"image_id {image_id} not in FLIR {self.modality}/{self.split}"
            ) from None

    def image_ids(self) -> np.ndarray:
        """``int64 [N]``, ascending."""
        return np.array([r.image_id for r in self.records], dtype=np.int64)

    def video_tags(self) -> dict[str, list[str]]:
        """``video_id`` → tags, from ``<split_dir>/index.json`` (``test`` split only)."""
        if self.split != "test":
            raise ValueError(f"video_tags() is only available for split 'test', not {self.split!r}")
        with open(self.split_dir / "index.json", encoding="utf-8") as f:
            d = json.load(f)
        return {v["id"]: list(v["tags"]) for v in d["videos"]}

    def to_coco_gt(self, area: Literal["normalised", "native"] = "normalised") -> dict:
        """COCO ground-truth dict over the 6 classes, with FLIR category ids.

        Kept boxes keep their original annotation ids. Every ignore box is emitted once per
        kept category with ``iscrowd: 1`` and fresh ids from ``max(original id) + 1``,
        because ``COCOeval`` matches crowd regions only within one category. ``area`` is
        normalised to 640×512 (FR3) or native ``w·h``. JSON-serialisable.
        """
        if area not in ("normalised", "native"):
            raise ValueError(f"Unknown area {area!r}; expected 'normalised' or 'native'")

        def box_area(b: np.ndarray, r: FlirRecord) -> float:
            w, h = float(b[2]), float(b[3])
            return normalised_area(w, h, r.width, r.height) if area == "normalised" else w * h

        cat_ids = sorted(CATEGORY_ID_TO_LABEL)
        images, annotations = [], []
        next_id = max((int(r.ann_ids.max()) for r in self.records if len(r.ann_ids)), default=0) + 1
        for r in self.records:
            images.append(
                {"id": int(r.image_id), "file_name": r.file_name,
                 "width": int(r.width), "height": int(r.height)}
            )
            for b, lab, aid in zip(r.boxes, r.labels, r.ann_ids):
                annotations.append(
                    {"id": int(aid), "image_id": int(r.image_id),
                     "category_id": cat_ids[int(lab)], "bbox": [float(v) for v in b],
                     "area": box_area(b, r), "iscrowd": 0}
                )
            for b in r.ignore_boxes:
                for cid in cat_ids:
                    annotations.append(
                        {"id": next_id, "image_id": int(r.image_id), "category_id": cid,
                         "bbox": [float(v) for v in b], "area": box_area(b, r), "iscrowd": 1}
                    )
                    next_id += 1
        categories = [{"id": cid, "name": CLASS_NAMES[CATEGORY_ID_TO_LABEL[cid]]} for cid in cat_ids]
        return {"images": images, "annotations": annotations, "categories": categories}


class FlirImageDataset(Dataset):
    """Lazily decoded native-resolution pixels of a ``FlirSplit``.

    ``__getitem__(i)`` → ``{"image": transform(img) | img, "image_id": int}`` with ``img``
    ``uint8 [H, W, 3]`` (rgb) or ``uint8 [H, W]`` (thermal). Targets are looked up with
    ``split.by_image_id``. No resize, no channel replication, and no open file handles
    between items, so the dataset pickles for DataLoader workers.
    """

    def __init__(
        self, split: FlirSplit, transform: Callable[[np.ndarray], Any] | None = None
    ) -> None:
        self.split = split
        self.transform = transform
        self._mode = _PIL_MODE[split.modality]

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> dict:
        r = self.split.records[i]
        with Image.open(r.path) as im:
            if im.mode != self._mode:
                raise ValueError(
                    f"{r.path}: decoded mode {im.mode!r}, expected {self._mode!r} for {self.split.modality}"
                )
            if im.size != (r.width, r.height):
                raise ValueError(
                    f"{r.path}: decoded size {im.size} != coco.json size {(r.width, r.height)}"
                )
            img = np.array(im, dtype=np.uint8)  # contiguous copy, detached from PIL buffers
        return {"image": self.transform(img) if self.transform else img, "image_id": r.image_id}
