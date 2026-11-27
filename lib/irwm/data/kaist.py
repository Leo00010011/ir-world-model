"""Read-only KAIST multispectral pedestrian readers: KAIST-Paired annotations, splits, image dataset.

KAIST (Hwang et al., CVPR 2015) is a set of 41 videos at 20 Hz from a pair of visible and LWIR
cameras, every frame 640×512. Two sources are read from ``$IRWM_DATA_ROOT``::

    kaist-cvpr15/images/setXX/VYYY/{visible,lwir}/I#####.jpg   # lwir JPEGs are stored as 3 channels
    kaist-cvpr15/imageSets/{train-all-04,test-all-20}.txt     # "setXX/VYYY/I#####" per line
    kaist-paired/annotations/setXX/VYYY/{visible,lwir}/I#####.txt

- **Annotations: KAIST-Paired** (Zhang et al., ICCV 2019; user decision 2026-11-27). Each camera
  has its own bbGt v3 file (``label x y w h occ xv yv wv hv ign ang``), so the RGB and thermal
  detectors each train and score on their own camera's boxes. A file with only the header is a
  negative frame. Paired test labels exist only on 1 frame in 20 (``test-all-20``).
- **Label space (user, 2026-11-27):** person = 0 and cyclist = 1 (COCO category ids 1 and 2).
  ``people``, ``person?`` and ``unpaired`` are **ignore regions**, following COCO: they are not
  loss targets, and in COCO GT each one is emitted once per kept category with ``iscrowd: 1``, so a
  detection on them is neither a TP nor an FP (the F2 rule, TechStack §3.4).
- **Occlusion:** column 5 (0 none, 1 partial, 2 heavy). In paired data only ``person`` boxes are
  ever occluded. ``to_coco_gt(occlusion=...)`` keeps only the person boxes at those levels as GT
  and turns every other box into an ignore region, giving AP on occluded pedestrians.
- **Splits (user, 2026-11-27).** Whole sets go to one split only:

  ============  ======================================  ==================  ==============
  split         sets                                    frames              time of day
  ============  ======================================  ==================  ==============
  ``train``     set00, set01 (day), set03, set04 (night)  ``train-all-04``  day + night
  ``val``       set02 (day), set05 (night)               ``train-all-04``  day + night
  ``test``      set06–set11 (official KAIST test)        ``test-all-20``   day + night
  ============  ======================================  ==================  ==============

  ``train-all-04`` is the official every-4th-frame list. ``train_eval_ids`` picks a fixed
  1-in-``stride`` subset of a split per video, for train-set metrics.
- **Boxes:** bbGt ``x y w h`` in pixels, top-left corner, clipped to the frame. A box with no area
  left after clipping is dropped and counted in ``KaistSplit.n_degenerate``.
- **ids:** ``image_id = set·10⁷ + video·10⁵ + frame`` (unique across splits and identical for both
  cameras). ``ann_id = image_id·1000 + k`` for kept boxes (k < 500) and ``image_id·1000 + 500 + 2j + c``
  for ignore box ``j`` emitted under category index ``c`` (j < 250), so ids are unique by construction.

Metadata loading never decodes images. Pixels come back at native resolution as ``uint8``:
``[H, W, 3]`` for RGB, ``[H, W]`` for thermal (the 3-channel lwir JPEG is converted to ``L``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from irwm.data.classes import normalised_area
from irwm.data.flir import Modality
from irwm.utils.paths import data_root

KAIST_IMAGES_DIRNAME = "kaist-cvpr15"  # relative to $IRWM_DATA_ROOT
KAIST_PAIRED_DIRNAME = "kaist-paired/annotations"
KAIST_CLASS_NAMES: tuple[str, ...] = ("person", "cyclist")
KAIST_NUM_CLASSES: int = len(KAIST_CLASS_NAMES)
KAIST_LABEL_TO_CATEGORY_ID: dict[int, int] = {0: 1, 1: 2}
KAIST_IGNORED: frozenset[str] = frozenset({"people", "person?", "unpaired"})
FRAME_WIDTH, FRAME_HEIGHT = 640, 512
CAMERA: dict[str, str] = {"rgb": "visible", "thermal": "lwir"}
OCCLUSION_LEVELS: dict[str, tuple[int, ...]] = {"occ_partial": (1,), "occ_heavy": (2,), "occ_any": (1, 2)}

KaistSplitName = Literal["train", "val", "test"]
SPLIT_SETS: dict[str, tuple[str, ...]] = {
    "train": ("set00", "set01", "set03", "set04"),
    "val": ("set02", "set05"),
    "test": ("set06", "set07", "set08", "set09", "set10", "set11"),
}
SPLIT_LIST: dict[str, str] = {"train": "train-all-04.txt", "val": "train-all-04.txt", "test": "test-all-20.txt"}
TIME_OF_DAY: dict[str, str] = {
    **{s: "day" for s in ("set00", "set01", "set02", "set06", "set07", "set08")},
    **{s: "night" for s in ("set03", "set04", "set05", "set09", "set10", "set11")},
}
_ENTRY_RE = re.compile(r"^(?P<set>set\d\d)/(?P<video>V\d\d\d)/(?P<frame>I\d{5})$")
_MAX_KEPT = 500  # ann_id = image_id * 1000 + k (kept) | + 500 + 2j + c (ignore j, category c)


def kaist_root() -> Path:
    """``$IRWM_DATA_ROOT/kaist-cvpr15`` (images + imageSets). Existence is not checked."""
    return data_root() / KAIST_IMAGES_DIRNAME


def kaist_paired_root() -> Path:
    """``$IRWM_DATA_ROOT/kaist-paired/annotations``. Existence is not checked."""
    return data_root() / KAIST_PAIRED_DIRNAME


def image_id_of(set_name: str, video: str, frame: str) -> int:
    """``set·10⁷ + video·10⁵ + frame`` from ``("set02", "V001", "I00144")``."""
    return int(set_name[3:]) * 10_000_000 + int(video[1:]) * 100_000 + int(frame[1:])


def read_image_set(name: str, root: Path | None = None) -> list[tuple[str, str, str]]:
    """``(set, video, frame)`` per line of ``imageSets/<name>``, in file order."""
    root = kaist_root() if root is None else Path(root)
    path = root / "imageSets" / name
    if not path.is_file():
        raise FileNotFoundError(f"KAIST image set not found: {path}")
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").split(), 1):
        m = _ENTRY_RE.match(line.strip())
        if m is None:
            raise ValueError(f"{path}: entry {n} {line!r} does not match {_ENTRY_RE.pattern}")
        out.append((m["set"], m["video"], m["frame"]))
    return out


def split_frames(split: KaistSplitName, root: Path | None = None) -> list[tuple[str, str, str]]:
    """The ``(set, video, frame)`` of one split, sorted. Raises if the split's sets are empty."""
    if split not in get_args(KaistSplitName):
        raise ValueError(f"Unknown split {split!r}; expected one of {list(get_args(KaistSplitName))}")
    sets = set(SPLIT_SETS[split])
    frames = sorted(f for f in read_image_set(SPLIT_LIST[split], root) if f[0] in sets)
    if not frames:
        raise ValueError(f"KAIST {split}: no frame of {sorted(sets)} in {SPLIT_LIST[split]}")
    return frames


@dataclass(frozen=True)
class KaistRecord:
    """One KAIST frame seen through one camera, with its kept (person / cyclist) and ignore boxes.

    Boxes are COCO ``xywh`` in native pixels, clipped to the frame, in file order.
    """

    image_id: int
    file_name: str  # relative to kaist_root(), e.g. "images/set00/V000/lwir/I00003.jpg"
    path: Path  # absolute
    width: int
    height: int
    boxes: np.ndarray  # float32 [N, 4]
    labels: np.ndarray  # int64 [N], 0..1
    occlusion: np.ndarray  # int64 [N], 0 none / 1 partial / 2 heavy
    ann_ids: np.ndarray  # int64 [N]
    ignore_boxes: np.ndarray  # float32 [M, 4]: people / person? / unpaired
    set_name: str
    video: str
    frame: int
    time_of_day: str  # "day" | "night"


def _clip(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    x0, y0 = min(max(x, 0.0), FRAME_WIDTH), min(max(y, 0.0), FRAME_HEIGHT)
    x1, y1 = min(max(x + w, 0.0), FRAME_WIDTH), min(max(y + h, 0.0), FRAME_HEIGHT)
    if not (x1 > x0 and y1 > y0):  # also catches NaN
        return None
    return x0, y0, x1 - x0, y1 - y0


def parse_bbgt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """``(boxes [N,4], labels [N], occlusion [N], ignore_boxes [M,4], n_degenerate)`` of one bbGt v3 file.

    Raises ``FileNotFoundError`` for a missing file and ``ValueError`` for a bad header, a short
    line or an unknown class.
    """
    if not path.is_file():
        raise FileNotFoundError(f"KAIST-Paired annotation not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("% bbGt"):
        raise ValueError(f"{path}: missing '% bbGt' header")
    boxes, labels, occ, ignore, n_degenerate = [], [], [], [], 0
    for n, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        name, *vals = line.split()
        if len(vals) < 5:
            raise ValueError(f"{path}:{n}: expected 'label x y w h occ ...', got {line!r}")
        box = _clip(*(float(v) for v in vals[:4]))
        if box is None:
            n_degenerate += 1
            continue
        if name in KAIST_IGNORED:
            ignore.append(box)
        elif name in KAIST_CLASS_NAMES:
            boxes.append(box)
            labels.append(KAIST_CLASS_NAMES.index(name))
            occ.append(int(float(vals[4])))
        else:
            raise ValueError(f"{path}:{n}: unknown class {name!r}")
    if len(boxes) >= _MAX_KEPT or 2 * len(ignore) > 1000 - _MAX_KEPT:
        raise ValueError(f"{path}: {len(boxes)} kept / {len(ignore)} ignore boxes exceed the ann_id scheme")
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64),
            np.asarray(occ, dtype=np.int64), np.asarray(ignore, dtype=np.float32).reshape(-1, 4), n_degenerate)


class KaistSplit:
    """Parsed metadata of one (modality, split) of KAIST-Paired: records by ``image_id``, no pixels.

    Duck-type compatible with ``FlirSplit`` / ``AlignedSplit`` for ``FlirDetectionDataset``:
    ``modality``, ``split``, ``root``, ``records``, ``by_image_id``, ``image_dataset`` and ``to_coco_gt``.
    """

    def __init__(self, modality: Modality, split: KaistSplitName, root: Path,
                 records: list[KaistRecord], n_degenerate: int = 0) -> None:
        if modality not in get_args(Modality):
            raise ValueError(f"Unknown modality {modality!r}; expected one of {list(get_args(Modality))}")
        self.modality = modality
        self.split = split
        self.root = root
        self.records = records
        self.n_degenerate = n_degenerate
        self._by_id = {r.image_id: r for r in records}
        if len(self._by_id) != len(records):
            raise ValueError("duplicate image_id in records")

    @classmethod
    def load(cls, modality: Modality, split: KaistSplitName, root: Path | None = None,
             paired_root: Path | None = None) -> KaistSplit:
        """Parse the split's paired annotations for ``modality``'s camera. No image I/O."""
        if modality not in get_args(Modality):
            raise ValueError(f"Unknown modality {modality!r}; expected one of {list(get_args(Modality))}")
        root = kaist_root() if root is None else Path(root)
        paired_root = kaist_paired_root() if paired_root is None else Path(paired_root)
        cam = CAMERA[modality]
        records, n_degenerate = [], 0
        for s, v, f in split_frames(split, root):
            boxes, labels, occ, ignore, bad = parse_bbgt(paired_root / s / v / cam / f"{f}.txt")
            n_degenerate += bad
            image_id = image_id_of(s, v, f)
            file_name = f"images/{s}/{v}/{cam}/{f}.jpg"
            records.append(KaistRecord(
                image_id=image_id, file_name=file_name, path=(root / file_name).resolve(),
                width=FRAME_WIDTH, height=FRAME_HEIGHT, boxes=boxes, labels=labels, occlusion=occ,
                ann_ids=image_id * 1000 + np.arange(len(labels), dtype=np.int64), ignore_boxes=ignore,
                set_name=s, video=v, frame=int(f[1:]), time_of_day=TIME_OF_DAY[s],
            ))
        return cls(modality, split, root, records, n_degenerate)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[KaistRecord]:
        return iter(self.records)

    def by_image_id(self, image_id: int) -> KaistRecord:
        """The record with this ``image_id``. Raises ``KeyError`` if it is not in this split."""
        try:
            return self._by_id[image_id]
        except KeyError:
            raise KeyError(f"image_id {image_id} not in KAIST {self.modality}/{self.split}") from None

    def image_ids(self) -> np.ndarray:
        """``int64 [N]``, ascending."""
        return np.array([r.image_id for r in self.records], dtype=np.int64)

    def videos(self) -> list[str]:
        """``"setXX/VYYY"`` of every video in the split, sorted."""
        return sorted({f"{r.set_name}/{r.video}" for r in self.records})

    def subset(self, image_ids: Iterable[int]) -> KaistSplit:
        """A copy restricted to ``image_ids`` (order kept ascending). ``KeyError`` on unknown ids."""
        wanted = {self.by_image_id(int(i)).image_id for i in image_ids}
        return KaistSplit(self.modality, self.split, self.root,
                          [r for r in self.records if r.image_id in wanted], self.n_degenerate)

    def image_dataset(self, transform: Callable[[np.ndarray], Any] | None = None) -> KaistImageDataset:
        """``KaistImageDataset(self, transform)``: the pixel reader for this split."""
        return KaistImageDataset(self, transform)

    def to_coco_gt(self, area: Literal["normalised", "native"] = "normalised",
                   occlusion: Sequence[int] | None = None) -> dict:
        """COCO ground truth over person (id 1) and cyclist (id 2), ignore regions as crowd.

        ``occlusion=None``: every kept box is GT. ``occlusion=(1,)`` etc.: only person boxes at those
        levels are GT; every other kept box becomes a crowd region of its own category, so a
        category left with no GT (cyclist) drops out of the mean AP. Ignore boxes are emitted once
        per category with ``iscrowd: 1``. ``area`` is kept for parity with ``FlirSplit.to_coco_gt``
        (every KAIST frame is 640×512, so both are equal). JSON-serialisable.
        """
        if area not in ("normalised", "native"):
            raise ValueError(f"Unknown area {area!r}; expected 'normalised' or 'native'")
        levels = None if occlusion is None else {int(o) for o in occlusion}
        person = KAIST_CLASS_NAMES.index("person")
        cat_ids = list(KAIST_LABEL_TO_CATEGORY_ID.values())

        def _area(b: np.ndarray, r: KaistRecord) -> float:
            bw, bh = float(b[2]), float(b[3])
            return normalised_area(bw, bh, r.width, r.height) if area == "normalised" else bw * bh

        images, annotations = [], []
        for r in self.records:
            images.append({"id": int(r.image_id), "file_name": r.file_name,
                           "width": int(r.width), "height": int(r.height)})
            for b, lab, occ, aid in zip(r.boxes, r.labels, r.occlusion, r.ann_ids):
                crowd = levels is not None and not (int(lab) == person and int(occ) in levels)
                annotations.append({"id": int(aid), "image_id": int(r.image_id),
                                    "category_id": KAIST_LABEL_TO_CATEGORY_ID[int(lab)],
                                    "bbox": [float(v) for v in b], "area": _area(b, r), "iscrowd": int(crowd)})
            base = int(r.image_id) * 1000 + _MAX_KEPT
            for j, b in enumerate(r.ignore_boxes):
                for c, cid in enumerate(cat_ids):  # COCOeval matches crowd only within a category
                    annotations.append({"id": base + 2 * j + c, "image_id": int(r.image_id),
                                        "category_id": cid, "bbox": [float(v) for v in b],
                                        "area": _area(b, r), "iscrowd": 1})
        categories = [{"id": KAIST_LABEL_TO_CATEGORY_ID[k], "name": n} for k, n in enumerate(KAIST_CLASS_NAMES)]
        return {"images": images, "annotations": annotations, "categories": categories}


def train_eval_ids(split: KaistSplit, stride: int) -> list[int]:
    """Every ``stride``-th frame of each video of ``split`` (the first one included), ascending ids.

    Deterministic, metadata only: the fixed train subset scored at every validation.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    by_video: dict[tuple[str, str], list[int]] = {}
    for r in split.records:
        by_video.setdefault((r.set_name, r.video), []).append(r.image_id)
    return sorted(i for ids in by_video.values() for i in sorted(ids)[::stride])


class KaistImageDataset(Dataset):
    """Lazily decoded native-resolution pixels of a ``KaistSplit``.

    ``__getitem__(i)`` → ``{"image": transform(img) | img, "image_id": int}`` with ``img`` ``uint8
    [H, W, 3]`` (rgb) or ``uint8 [H, W]`` (thermal; the 3-channel lwir JPEG is converted to ``L``).
    The decoded size is checked; no file handle stays open, so the dataset pickles for workers.
    """

    def __init__(self, split: KaistSplit, transform: Callable[[np.ndarray], Any] | None = None) -> None:
        self.split = split
        self.transform = transform
        self._mode = "RGB" if split.modality == "rgb" else "L"

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> dict:
        r = self.split.records[i]
        with Image.open(r.path) as im:
            if im.size != (r.width, r.height):
                raise ValueError(f"{r.path}: decoded size {im.size} != {(r.width, r.height)}")
            img = np.array(im.convert(self._mode), dtype=np.uint8)
        return {"image": self.transform(img) if self.transform else img, "image_id": r.image_id}
