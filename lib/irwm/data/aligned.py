"""Read-only aligned FLIR v1 readers: splits, VOC annotations and image dataset (F16).

Aligned FLIR v1 (Zhang et al., 2020) is a detection-only dataset of pixel-aligned
RGB/thermal pairs at 640×512 (Mission → 2026-11-13). It is read from::

    <root>/align_train.txt, align_validation.txt       # "FLIR_<id>_PreviewData" per line
    <root>/JPEGImages/FLIR_<id>_PreviewData.jpeg       # thermal, mode L
    <root>/JPEGImages/FLIR_<id>_RGB.jpg                # RGB, aligned to thermal
    <root>/Annotations/FLIR_<id>_PreviewData.xml       # Pascal VOC, one per pair

with ``<root> = $IRWM_DATA_ROOT/aligned/align``. ``*_mask.jpg`` and ``AnnotatedImages/``
are not read.

- **Label space:** its own 2-class space, person = 0 and car = 1 (COCO category ids 1 and
  2). It is deliberately not the 6-class F2 space, and the ids differ from FLIR v2's car id
  (3), so aligned GT never silently scores against v2 detections (Roadmap R12).
- **Bicycle and dog are background:** their boxes are dropped, with no ignore or crowd
  regions. Any other class name is an error.
- **Boxes:** VOC corners are 0-based (``xmin = 0`` occurs) and are read as continuous
  corners, ``w = xmax − xmin``, ``h = ymax − ymin`` (no +1), then clipped to the frame. The
  values look converted from FLIR v1's COCO ``xywh``, and this keeps parity with v2's COCO
  boxes. One real box (FLIR_03093, ``xmax = 641``) is clipped.
- **Splits:** ``test`` = ``align_validation.txt``. ``val`` = the committed ids in
  ``aligned_splits/aligned_val_ids.txt``: the ``round(0.10·n)`` highest ids of
  ``align_train.txt``. ``train`` = the rest of ``align_train.txt``, minus a purge band of
  the ids in ``[min(val) − 20, min(val))``, so near-identical boundary frames don't
  straddle train and val. Both modalities share frames, boxes and ``image_id`` (the integer
  ``<id>``, unique across splits).

Metadata loading never decodes images. Pixels come back at native resolution as ``uint8``.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from irwm.data.classes import normalised_area
from irwm.data.flir import Modality, Split
from irwm.utils.paths import data_root

ALIGNED_DIRNAME = "aligned/align"  # relative to $IRWM_DATA_ROOT
ALIGNED_CLASS_NAMES: tuple[str, ...] = ("person", "car")
ALIGNED_NUM_CLASSES: int = len(ALIGNED_CLASS_NAMES)
ALIGNED_LABEL_TO_CATEGORY_ID: dict[int, int] = {0: 1, 1: 2}
ALIGNED_CATEGORY_ID_TO_LABEL: dict[int, int] = {v: k for k, v in ALIGNED_LABEL_TO_CATEGORY_ID.items()}
ALIGNED_DROPPED: frozenset[str] = frozenset({"bicycle", "dog"})
VAL_FRACTION: float = 0.10
PURGE_GAP: int = 20
FRAME_WIDTH, FRAME_HEIGHT = 640, 512

_TRAIN_LIST = "align_train.txt"
_TEST_LIST = "align_validation.txt"
VAL_IDS_FILE: Path = Path(__file__).resolve().parent / "aligned_splits" / "aligned_val_ids.txt"
_ENTRY_RE = re.compile(r"^(?P<stem>FLIR_(?P<id>\d+))_PreviewData$")
_IMAGE_SUFFIX: dict[str, str] = {"thermal": "_PreviewData.jpeg", "rgb": "_RGB.jpg"}
_PIL_MODE: dict[str, str] = {"rgb": "RGB", "thermal": "L"}
_MAX_OBJECTS = 1000  # ann_id = image_id * 1000 + k


def aligned_root() -> Path:
    """``$IRWM_DATA_ROOT/aligned/align``. Existence is not checked."""
    return data_root() / ALIGNED_DIRNAME


def _read_list(path: Path) -> list[tuple[int, str]]:
    """``(id, "FLIR_<id>")`` per line of an ``align_*.txt``, in file order."""
    if not path.is_file():
        raise FileNotFoundError(f"aligned FLIR split list not found: {path}")
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        m = _ENTRY_RE.match(line)
        if m is None:
            raise ValueError(f"{path}:{n}: entry {line!r} does not match {_ENTRY_RE.pattern}")
        out.append((int(m["id"]), m["stem"]))
    ids = [i for i, _ in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    return out


def carve_val_ids(train_ids: Sequence[int], fraction: float = VAL_FRACTION) -> list[int]:
    """The ``round(fraction · n)`` highest of ``train_ids``, ascending."""
    ids = sorted(set(int(i) for i in train_ids))
    k = round(fraction * len(ids))
    if not 0 < k < len(ids):
        raise ValueError(f"val carve of {fraction} over {len(ids)} ids gives {k} ids")
    return ids[-k:]


def purge_ids(train_ids: Sequence[int], val_ids: Sequence[int], gap: int = PURGE_GAP) -> list[int]:
    """Ids of ``train_ids`` in ``[min(val_ids) − gap, min(val_ids))``, ascending."""
    lo = min(val_ids)
    return sorted(int(i) for i in train_ids if lo - gap <= int(i) < lo)


def render_val_file(root: Path | None = None) -> str:
    """Contents of the committed val-ids file, derived from ``<root>/align_train.txt``."""
    root = aligned_root() if root is None else Path(root)
    src = root / _TRAIN_LIST
    train = _read_list(src)
    ids = [i for i, _ in train]
    width = len(train[0][1]) - len("FLIR_")
    val = carve_val_ids(ids)
    purged = purge_ids(ids, val)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    fmt = lambda i: f"{i:0{width}d}"  # noqa: E731
    lines = [
        "# Aligned FLIR v1 val split (F16). Generated by irwm.data.aligned.render_val_file; do not edit.",
        f"# source: {_TRAIN_LIST} sha256={sha} n={len(ids)}",
        f"# rule: val = the round({VAL_FRACTION} * {len(ids)}) = {len(val)} highest ids of {_TRAIN_LIST}",
        f"# purge: train also drops ids in [min(val) - {PURGE_GAP}, min(val)) = "
        f"[{fmt(val[0] - PURGE_GAP)}, {fmt(val[0])}): {len(purged)} ids",
        "# purged: " + " ".join(fmt(i) for i in purged),
        *(fmt(i) for i in val),
    ]
    return "\n".join(lines) + "\n"


def _parse_val_file(path: Path) -> tuple[list[int], list[int]]:
    val, purged = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# purged:"):
            purged = [int(t) for t in line[len("# purged:"):].split()]
        elif line.strip() and not line.startswith("#"):
            val.append(int(line))
    return val, purged


def load_val_ids(path: Path = VAL_IDS_FILE) -> list[int]:
    """The committed aligned-val ids, ascending."""
    return _parse_val_file(path)[0]


def load_purged_ids(path: Path = VAL_IDS_FILE) -> list[int]:
    """The committed purge band: ids of ``align_train.txt`` in neither train nor val."""
    return _parse_val_file(path)[1]


def split_stems(split: Split, root: Path | None = None, val_file: Path = VAL_IDS_FILE) -> list[str]:
    """``FLIR_<id>`` stems of one split, ascending id.

    Raises ``ValueError`` if the two lists overlap or a committed val/purged id is not in
    ``align_train.txt``.
    """
    if split not in get_args(Split):
        raise ValueError(f"Unknown split {split!r}; expected one of {list(get_args(Split))}")
    root = aligned_root() if root is None else Path(root)
    train = dict(_read_list(root / _TRAIN_LIST))
    test = dict(_read_list(root / _TEST_LIST))
    if overlap := train.keys() & test.keys():
        raise ValueError(f"{_TRAIN_LIST} and {_TEST_LIST} share {len(overlap)} ids, e.g. {min(overlap)}")
    if split == "test":
        return [test[i] for i in sorted(test)]
    val, purged = _parse_val_file(val_file)
    if missing := (set(val) | set(purged)) - train.keys():
        raise ValueError(f"{val_file}: {len(missing)} ids not in {_TRAIN_LIST}, e.g. {min(missing)}")
    if split == "val":
        return [train[i] for i in sorted(val)]
    drop = set(val) | set(purged)
    return [train[i] for i in sorted(train) if i not in drop]


@dataclass(frozen=True)
class AlignedRecord:
    """One aligned pair seen through one modality, with its kept (person/car) boxes.

    Boxes are COCO ``xywh`` in native pixels, clipped to the frame, in XML order.
    ``ann_ids`` are ``image_id · 1000 + k`` (``k`` = index among kept boxes), unique across
    splits and identical for both modalities.
    """

    image_id: int
    file_name: str  # relative to the root, e.g. "JPEGImages/FLIR_00258_RGB.jpg"
    path: Path  # absolute
    width: int
    height: int
    boxes: np.ndarray  # float32 [N, 4]
    labels: np.ndarray  # int64 [N], 0..1
    ann_ids: np.ndarray  # int64 [N]


def _parse_voc(path: Path) -> tuple[int, int, np.ndarray, np.ndarray]:
    """``(width, height, boxes xywh float32 [N,4], labels int64 [N])`` of one VOC file."""
    if not path.is_file():
        raise FileNotFoundError(f"aligned FLIR annotation not found: {path}")
    root = ET.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{path}: no <size>")
    w, h = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
    if (w, h) != (FRAME_WIDTH, FRAME_HEIGHT):
        raise ValueError(f"{path}: size {w}x{h}, expected {FRAME_WIDTH}x{FRAME_HEIGHT}")
    boxes, labels = [], []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name in ALIGNED_DROPPED:
            continue
        if name not in ALIGNED_CLASS_NAMES:
            raise ValueError(f"{path}: unknown class {name!r}")
        bb = obj.find("bndbox")
        if bb is None:
            raise ValueError(f"{path}: object {name!r} has no <bndbox>")
        x0, y0, x1, y1 = (float(bb.findtext(t, "nan")) for t in ("xmin", "ymin", "xmax", "ymax"))
        x0, x1 = min(max(x0, 0.0), w), min(max(x1, 0.0), w)
        y0, y1 = min(max(y0, 0.0), h), min(max(y1, 0.0), h)
        if not (x1 > x0 and y1 > y0):  # also catches NaN
            raise ValueError(f"{path}: degenerate box {name!r} after clipping: {(x0, y0, x1, y1)}")
        boxes.append([x0, y0, x1 - x0, y1 - y0])
        labels.append(ALIGNED_CLASS_NAMES.index(name))
    if len(boxes) >= _MAX_OBJECTS:
        raise ValueError(f"{path}: {len(boxes)} boxes, ann_id scheme allows < {_MAX_OBJECTS}")
    return (w, h, np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(labels, dtype=np.int64).reshape(-1))


class AlignedSplit:
    """Parsed metadata of one (modality, split) of aligned FLIR v1: records by ``image_id``, no pixels.

    Duck-type compatible with ``FlirSplit`` for ``FlirDetectionDataset``: ``modality``,
    ``split``, ``root``, ``records``, ``by_image_id``, ``__len__`` and ``to_coco_gt``.
    """

    def __init__(self, modality: Modality, split: Split, root: Path,
                 records: list[AlignedRecord]) -> None:
        if modality not in get_args(Modality):
            raise ValueError(f"Unknown modality {modality!r}; expected one of {list(get_args(Modality))}")
        self.modality = modality
        self.split = split
        self.root = root
        self.records = records
        self._by_id = {r.image_id: r for r in records}
        if len(self._by_id) != len(records):
            raise ValueError("duplicate image_id in records")

    @classmethod
    def load(cls, modality: Modality, split: Split, root: Path | None = None,
             val_file: Path = VAL_IDS_FILE) -> AlignedSplit:
        """Parse the split's VOC files. No image I/O.

        Raises ``FileNotFoundError`` for a missing list or XML, and ``ValueError`` for an
        unknown modality/split/class, a frame other than 640×512 or a degenerate box.
        """
        if modality not in get_args(Modality):
            raise ValueError(f"Unknown modality {modality!r}; expected one of {list(get_args(Modality))}")
        root = aligned_root() if root is None else Path(root)
        records = []
        for stem in split_stems(split, root, val_file):
            image_id = int(stem[len("FLIR_"):])
            w, h, boxes, labels = _parse_voc(root / "Annotations" / f"{stem}_PreviewData.xml")
            file_name = f"JPEGImages/{stem}{_IMAGE_SUFFIX[modality]}"
            records.append(AlignedRecord(
                image_id=image_id, file_name=file_name, path=(root / file_name).resolve(),
                width=w, height=h, boxes=boxes, labels=labels,
                ann_ids=image_id * _MAX_OBJECTS + np.arange(len(labels), dtype=np.int64),
            ))
        return cls(modality, split, root, records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[AlignedRecord]:
        return iter(self.records)

    def by_image_id(self, image_id: int) -> AlignedRecord:
        """The record with this ``image_id``. Raises ``KeyError`` if it is not in this split."""
        try:
            return self._by_id[image_id]
        except KeyError:
            raise KeyError(f"image_id {image_id} not in aligned FLIR {self.modality}/{self.split}") from None

    def image_ids(self) -> np.ndarray:
        """``int64 [N]``, ascending."""
        return np.array([r.image_id for r in self.records], dtype=np.int64)

    def subset(self, image_ids: Iterable[int]) -> AlignedSplit:
        """A copy restricted to ``image_ids`` (order kept ascending). ``KeyError`` on unknown ids."""
        wanted = {self.by_image_id(int(i)).image_id for i in image_ids}
        return AlignedSplit(self.modality, self.split, self.root,
                            [r for r in self.records if r.image_id in wanted])

    def image_dataset(self, transform: Callable[[np.ndarray], Any] | None = None) -> AlignedImageDataset:
        """``AlignedImageDataset(self, transform)``: the pixel reader for this split."""
        return AlignedImageDataset(self, transform)

    def to_coco_gt(self, area: Literal["normalised", "native"] = "normalised") -> dict:
        """COCO ground-truth dict over person (id 1) and car (id 2). No crowd regions.

        ``area`` is normalised to 640×512 or native ``w·h`` (identical here, kept for
        parity with ``FlirSplit.to_coco_gt``). JSON-serialisable.
        """
        if area not in ("normalised", "native"):
            raise ValueError(f"Unknown area {area!r}; expected 'normalised' or 'native'")
        images, annotations = [], []
        for r in self.records:
            images.append({"id": int(r.image_id), "file_name": r.file_name,
                           "width": int(r.width), "height": int(r.height)})
            for b, lab, aid in zip(r.boxes, r.labels, r.ann_ids):
                bw, bh = float(b[2]), float(b[3])
                a = normalised_area(bw, bh, r.width, r.height) if area == "normalised" else bw * bh
                annotations.append({"id": int(aid), "image_id": int(r.image_id),
                                    "category_id": ALIGNED_LABEL_TO_CATEGORY_ID[int(lab)],
                                    "bbox": [float(v) for v in b], "area": a, "iscrowd": 0})
        categories = [{"id": ALIGNED_LABEL_TO_CATEGORY_ID[k], "name": n}
                      for k, n in enumerate(ALIGNED_CLASS_NAMES)]
        return {"images": images, "annotations": annotations, "categories": categories}


class AlignedImageDataset(Dataset):
    """Lazily decoded native-resolution pixels of an ``AlignedSplit``.

    ``__getitem__(i)`` → ``{"image": transform(img) | img, "image_id": int}`` with ``img``
    ``uint8 [H, W, 3]`` (rgb) or ``uint8 [H, W]`` (thermal). The decoded mode and size are
    checked, and no file handle stays open, so the dataset pickles for DataLoader workers.
    """

    def __init__(self, split: AlignedSplit,
                 transform: Callable[[np.ndarray], Any] | None = None) -> None:
        self.split = split
        self.transform = transform
        self._mode = _PIL_MODE[split.modality]

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> dict:
        r = self.split.records[i]
        with Image.open(r.path) as im:
            if im.mode != self._mode:
                raise ValueError(f"{r.path}: decoded mode {im.mode!r}, expected {self._mode!r} "
                                 f"for {self.split.modality}")
            if im.size != (r.width, r.height):
                raise ValueError(f"{r.path}: decoded size {im.size} != annotation size {(r.width, r.height)}")
            img = np.array(im, dtype=np.uint8)
        return {"image": self.transform(img) if self.transform else img, "image_id": r.image_id}


if __name__ == "__main__":  # regenerate the committed file: python -m irwm.data.aligned
    from irwm.utils.paths import load_env

    load_env()
    VAL_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    VAL_IDS_FILE.write_text(render_val_file(), encoding="utf-8")
    print(f"wrote {VAL_IDS_FILE}")
