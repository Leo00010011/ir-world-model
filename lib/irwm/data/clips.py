"""Video clip index over the FLIR test split: build, serialise and read evaluation clips.

Spec: ``spec/2026-10-14-video-clip-index`` (FR1–FR10). A clip is ``n_context`` context and
``n_future`` future frames at ``frame_stride`` raw frames apart (default 4 + 6 at stride 5,
0.167 s at 30 FPS), inside one gap-free segment of one video.

Two rules shape this module:

- **References only.** A clip stores frame indices, ``image_id``s and file basenames, plus
  two kept-box counts for the evaluable flag. Never pixels or annotations; those are looked
  up by ``image_id`` in ``FlirSplit`` / ``FlirImageDataset`` or the F5 feature cache.
- **One index shared by both modalities.** One row is one time window and lists the RGB
  and the thermal references of the same time-synced frames, so ``clip_id`` 15 is the same
  window in both. Clips are ordered by RGB ``video_id`` then start frame, and ``clip_id`` is
  assigned before (and independently of) the evaluable flag.

The committed clip set lives in ``irwm/data/clip_sets/<clip_set>.json`` and is read with
``ClipIndex.load()``, which needs neither the dataset nor an env var.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import irwm.data
from irwm.data.flir import FlirRecord, FlirSplit, Modality

DEFAULT_CLIP_SET = "flir_test_fs5_c4f6_cs10"
FORMAT_VERSION = 1
FPS = 30
FPS_SOURCE = (
    "FLIR README.txt: 30 FPS; not verifiable from the data, the test split has no "
    "per-frame timestamps (F2)"
)
EVALUABLE_RULE = "kept future boxes >= 1 in both modalities"

_MODALITIES: tuple[str, str] = ("rgb", "thermal")
_MAX_REPORTED = 20
_PER_FRAME_KEYS = (
    "frame_indices", "rgb_image_ids", "thermal_image_ids", "rgb_file_names", "thermal_file_names",
)


def _basename(r: FlirRecord) -> str:
    return Path(r.file_name).name


def check_pairing(
    rgb: FlirSplit, thermal: FlirSplit, rgb_to_thermal: dict[str, str]
) -> list[tuple[str, str]]:
    """Validate the RGB↔thermal test map; return ``(rgb_video_id, thermal_video_id)`` pairs.

    ``rgb_to_thermal`` maps basenames (``load_rgb_thermal_map``). Every violation of FR1 is
    collected, then one ``ValueError`` gives their count and the first 20. Pairs are sorted
    by ``rgb_video_id``.
    """
    if rgb.split != "test" or thermal.split != "test":
        raise ValueError(
            f"check_pairing needs the test splits, got rgb={rgb.split!r}, thermal={thermal.split!r}"
        )
    if rgb.modality != "rgb" or thermal.modality != "thermal":
        raise ValueError(
            f"check_pairing needs (rgb, thermal) splits, got ({rgb.modality!r}, {thermal.modality!r})"
        )

    violations: list[str] = []
    by_name: dict[str, dict[str, FlirRecord]] = {}
    for split in (rgb, thermal):
        counts = Counter(_basename(r) for r in split)
        for name, n in sorted(counts.items()):
            if n > 1:
                violations.append(f"basename {name} appears in {n} {split.modality} test records")
        by_name[split.modality] = {_basename(r): r for r in split}

    if not len(rgb_to_thermal) == len(rgb) == len(thermal):
        violations.append(
            f"size mismatch: map has {len(rgb_to_thermal)} entries, rgb test has {len(rgb)} "
            f"records, thermal test has {len(thermal)} records"
        )
    sources: dict[str, list[str]] = defaultdict(list)
    for key, value in rgb_to_thermal.items():
        sources[value].append(key)
    for value, keys in sorted(sources.items()):
        if len(keys) > 1:
            violations.append(
                f"duplicate value: thermal {value} is mapped from {len(keys)} RGB basenames "
                f"({', '.join(sorted(keys))})"
            )

    rgb_to_tv: dict[str, set[str]] = defaultdict(set)
    tv_to_rgb: dict[str, set[str]] = defaultdict(set)
    for key, value in sorted(rgb_to_thermal.items()):
        r = by_name["rgb"].get(key)
        t = by_name["thermal"].get(value)
        if r is None:
            violations.append(f"unknown key: {key} is not an RGB test basename")
        if t is None:
            violations.append(f"unknown value: {value} is not a thermal test basename")
        if r is None or t is None:
            continue
        if r.frame_index != t.frame_index:
            violations.append(
                f"frame_index mismatch: rgb {key} ({r.frame_index}) -> thermal {value} "
                f"({t.frame_index})"
            )
        rgb_to_tv[r.video_id].add(t.video_id)
        tv_to_rgb[t.video_id].add(r.video_id)
    for src, a, b in ((rgb_to_tv, "RGB", "thermal"), (tv_to_rgb, "thermal", "RGB")):
        for vid, targets in sorted(src.items()):
            if len(targets) > 1:
                violations.append(
                    f"{a} video {vid} maps to {len(targets)} {b} videos ({', '.join(sorted(targets))})"
                )

    if violations:
        shown = "\n".join(f"  - {v}" for v in violations[:_MAX_REPORTED])
        raise ValueError(
            f"RGB↔thermal pairing check failed: {len(violations)} violation(s), "
            f"first {min(len(violations), _MAX_REPORTED)}:\n{shown}"
        )
    return sorted({(rv, next(iter(tvs))) for rv, tvs in rgb_to_tv.items()})


def find_segments(frame_indices: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive ``frame_index`` values as inclusive ``(first, last)``.

    The input may be unsorted. A duplicate raises ``ValueError``; an empty input gives ``[]``.
    """
    idx = sorted(int(i) for i in frame_indices)
    for a, b in zip(idx, idx[1:]):
        if a == b:
            raise ValueError(f"duplicate frame_index {a}")
    segments: list[tuple[int, int]] = []
    for i in idx:
        if segments and i == segments[-1][1] + 1:
            segments[-1] = (segments[-1][0], i)
        else:
            segments.append((i, i))
    return segments


def clip_set_name(frame_stride: int, n_context: int, n_future: int, clip_stride: int) -> str:
    """E.g. ``flir_test_fs5_c4f6_cs10``; the ``<clip_set>`` of the prediction cache path."""
    return f"flir_test_fs{frame_stride}_c{n_context}f{n_future}_cs{clip_stride}"


def build_clip_index(
    rgb: FlirSplit,
    thermal: FlirSplit,
    rgb_to_thermal: dict[str, str],
    *,
    frame_stride: int = 5,
    n_context: int = 4,
    n_future: int = 6,
    clip_stride: int = 10,
    inputs: dict[str, str] | None = None,
) -> dict:
    """Build the JSON-native clip index document (FR3–FR5). Calls ``check_pairing`` first.

    Windows start at ``a + k·clip_stride`` in each gap-free segment ``(a, b)`` and span
    ``frame_stride·(T − 1) + 1`` raw frames, ``T = n_context + n_future``. Any parameter
    ``< 1`` raises ``ValueError``.
    """
    params = {"frame_stride": frame_stride, "n_context": n_context, "n_future": n_future,
              "clip_stride": clip_stride}
    for name, value in params.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    pairs = check_pairing(rgb, thermal, rgb_to_thermal)

    T = n_context + n_future
    last_offset = frame_stride * (T - 1)
    frames: dict[str, dict[tuple[str, int], FlirRecord]] = {
        s.modality: {(r.video_id, r.frame_index): r for r in s} for s in (rgb, thermal)
    }
    rgb_frames_by_video: dict[str, list[int]] = defaultdict(list)
    for r in rgb:
        rgb_frames_by_video[r.video_id].append(r.frame_index)

    pair_entries: list[dict] = []
    clips: list[dict] = []
    for rgb_vid, th_vid in pairs:
        segments = find_segments(rgb_frames_by_video[rgb_vid])
        n_clips = n_evaluable = 0
        for seg_idx, (a, b) in enumerate(segments):
            s = a
            while s + last_offset <= b:
                fidx = [s + frame_stride * j for j in range(T)]
                recs = {
                    "rgb": [frames["rgb"][(rgb_vid, f)] for f in fidx],
                    "thermal": [frames["thermal"][(th_vid, f)] for f in fidx],
                }
                n_boxes = {m: int(sum(len(r.boxes) for r in recs[m][n_context:])) for m in _MODALITIES}
                evaluable = n_boxes["rgb"] >= 1 and n_boxes["thermal"] >= 1
                clips.append({
                    "clip_id": len(clips),
                    "key": f"{rgb_vid}:{s:06d}",
                    "rgb_video_id": rgb_vid,
                    "thermal_video_id": th_vid,
                    "segment_index": seg_idx,
                    "frame_indices": fidx,
                    "rgb_image_ids": [int(r.image_id) for r in recs["rgb"]],
                    "thermal_image_ids": [int(r.image_id) for r in recs["thermal"]],
                    "rgb_file_names": [_basename(r) for r in recs["rgb"]],
                    "thermal_file_names": [_basename(r) for r in recs["thermal"]],
                    "n_future_boxes": n_boxes,
                    "evaluable": evaluable,
                })
                n_clips += 1
                n_evaluable += int(evaluable)
                s += clip_stride
        pair_entries.append({
            "rgb_video_id": rgb_vid,
            "thermal_video_id": th_vid,
            "n_frames": len(rgb_frames_by_video[rgb_vid]),
            "segments": [[a, b] for a, b in segments],
            "n_clips": n_clips,
            "n_evaluable": n_evaluable,
        })

    return {
        "format_version": FORMAT_VERSION,
        "clip_set": clip_set_name(frame_stride, n_context, n_future, clip_stride),
        "params": {"split": "test", **params, "fps": FPS, "fps_source": FPS_SOURCE,
                   "evaluable_rule": EVALUABLE_RULE},
        "inputs": dict(inputs) if inputs else {},
        "pairs": pair_entries,
        "clips": clips,
    }


def dump_clip_index(doc: dict) -> str:
    """Deterministic text: ``indent=2`` for the head, one clip per line, one trailing newline."""
    head = json.dumps({k: v for k, v in doc.items() if k != "clips"}, indent=2)
    lines = [f"    {json.dumps(c, separators=(', ', ': '))}" for c in doc["clips"]]
    body = "[\n" + ",\n".join(lines) + "\n  ]" if lines else "[]"
    text = f"{head[:-2]},\n  \"clips\": {body}\n}}\n"
    if json.loads(text) != doc:
        raise AssertionError("dump_clip_index does not round-trip; this is a bug")
    return text


def clip_set_path(clip_set: str) -> Path:
    """``<irwm.data package>/clip_sets/<clip_set>.json``: a repo path, not a data path."""
    return Path(irwm.data.__file__).parent / "clip_sets" / f"{clip_set}.json"


def write_clip_index(doc: dict, path: Path) -> bool:
    """Write ``dump_clip_index(doc)`` as UTF-8 (``\\n`` line ends on every OS).

    Returns ``False`` without writing if ``path`` already holds exactly that text, and raises
    ``FileExistsError`` (file untouched) if it holds anything else.
    """
    data = dump_clip_index(doc).encode("utf-8")
    path = Path(path)
    if path.exists():
        if path.read_bytes() == data:
            return False
        raise FileExistsError(
            f"{path} exists with different content; a clip set is never overwritten "
            "(use new parameters, which give a new clip_set name)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of the file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_modality(modality: str) -> None:
    if modality not in _MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}; expected one of {list(_MODALITIES)}")


@dataclass(frozen=True)
class Clip:
    """One evaluation clip: ``T = len(frame_indices)`` frames, the first ``n_context`` context."""

    clip_id: int
    key: str
    rgb_video_id: str
    thermal_video_id: str
    segment_index: int
    n_context: int
    frame_indices: tuple[int, ...]
    rgb_image_ids: tuple[int, ...]
    thermal_image_ids: tuple[int, ...]
    rgb_file_names: tuple[str, ...]
    thermal_file_names: tuple[str, ...]
    n_future_boxes_rgb: int
    n_future_boxes_thermal: int
    evaluable: bool

    def video_id(self, modality: Modality) -> str:
        _check_modality(modality)
        return self.rgb_video_id if modality == "rgb" else self.thermal_video_id

    def image_ids(self, modality: Modality) -> np.ndarray:
        """``int64 [T]``, context then future."""
        _check_modality(modality)
        ids = self.rgb_image_ids if modality == "rgb" else self.thermal_image_ids
        return np.asarray(ids, dtype=np.int64)

    def context_ids(self, modality: Modality) -> np.ndarray:
        """``int64 [n_context]``."""
        return self.image_ids(modality)[: self.n_context]

    def future_ids(self, modality: Modality) -> np.ndarray:
        """``int64 [T − n_context]``."""
        return self.image_ids(modality)[self.n_context:]

    def file_names(self, modality: Modality) -> tuple[str, ...]:
        """Basenames, no ``data/`` prefix."""
        _check_modality(modality)
        return self.rgb_file_names if modality == "rgb" else self.thermal_file_names

    def n_future_boxes(self, modality: Modality) -> int:
        """Kept (6-class) boxes summed over the future frames."""
        _check_modality(modality)
        return self.n_future_boxes_rgb if modality == "rgb" else self.n_future_boxes_thermal


class ClipIndex:
    """A loaded clip set: ``Clip``s in ``clip_id`` order, metadata only."""

    def __init__(self, clip_set: str, params: dict, pairs: list[dict], clips: list[Clip]) -> None:
        self.clip_set = clip_set
        self.params = params
        self.pairs = pairs
        self._clips = clips
        self._by_key = {c.key: c for c in clips}

    @classmethod
    def load(cls, clip_set: str = DEFAULT_CLIP_SET, path: Path | None = None) -> ClipIndex:
        """Read ``path``, or the committed ``clip_set_path(clip_set)`` if ``path`` is ``None``.

        Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` on a wrong
        ``format_version``, a ``clip_id`` that differs from its position, a duplicate key or
        a per-frame list whose length is not ``n_context + n_future``.
        """
        path = clip_set_path(clip_set) if path is None else Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"clip set not found: {path}")
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path}: format_version {doc.get('format_version')!r}, expected {FORMAT_VERSION}"
            )
        params = doc["params"]
        n_context = int(params["n_context"])
        T = n_context + int(params["n_future"])
        clips: list[Clip] = []
        seen: set[str] = set()
        for pos, c in enumerate(doc["clips"]):
            if c["clip_id"] != pos:
                raise ValueError(f"{path}: clip at position {pos} has clip_id {c['clip_id']}")
            if c["key"] in seen:
                raise ValueError(f"{path}: duplicate key {c['key']!r}")
            seen.add(c["key"])
            for k in _PER_FRAME_KEYS:
                if len(c[k]) != T:
                    raise ValueError(f"{path}: clip {pos} {k} has length {len(c[k])}, expected {T}")
            clips.append(Clip(
                clip_id=int(c["clip_id"]),
                key=c["key"],
                rgb_video_id=c["rgb_video_id"],
                thermal_video_id=c["thermal_video_id"],
                segment_index=int(c["segment_index"]),
                n_context=n_context,
                frame_indices=tuple(int(i) for i in c["frame_indices"]),
                rgb_image_ids=tuple(int(i) for i in c["rgb_image_ids"]),
                thermal_image_ids=tuple(int(i) for i in c["thermal_image_ids"]),
                rgb_file_names=tuple(c["rgb_file_names"]),
                thermal_file_names=tuple(c["thermal_file_names"]),
                n_future_boxes_rgb=int(c["n_future_boxes"]["rgb"]),
                n_future_boxes_thermal=int(c["n_future_boxes"]["thermal"]),
                evaluable=bool(c["evaluable"]),
            ))
        return cls(doc["clip_set"], params, doc["pairs"], clips)

    def __len__(self) -> int:
        return len(self._clips)

    def __iter__(self) -> Iterator[Clip]:
        return iter(self._clips)

    def __getitem__(self, clip_id: int) -> Clip:
        """Raises ``IndexError`` outside ``0 … len − 1``; negative ids are not positions."""
        if not 0 <= clip_id < len(self._clips):
            raise IndexError(f"clip_id {clip_id} not in 0..{len(self._clips) - 1} ({self.clip_set})")
        return self._clips[clip_id]

    def by_key(self, key: str) -> Clip:
        """Raises ``KeyError`` for an unknown key."""
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"clip key {key!r} not in {self.clip_set}") from None

    def evaluable(self) -> list[Clip]:
        """Clips with ``evaluable == True``, in ``clip_id`` order."""
        return [c for c in self._clips if c.evaluable]


def summarise(doc: dict) -> dict:
    """Counts for the report (FR10): totals, per pair, per modality. JSON-native."""
    clips = doc["clips"]
    pairs = []
    for p in doc["pairs"]:
        ids = [c["clip_id"] for c in clips if c["rgb_video_id"] == p["rgb_video_id"]]
        pairs.append({**p, "clip_id_first": min(ids, default=None), "clip_id_last": max(ids, default=None)})
    modalities = {
        m: {
            "n_frames_covered": len({i for c in clips for i in c[f"{m}_image_ids"]}),
            "n_not_evaluable_caused": sum(c["n_future_boxes"][m] == 0 for c in clips),
        }
        for m in _MODALITIES
    }
    return {
        "clip_set": doc["clip_set"],
        "params": doc["params"],
        "n_pairs": len(doc["pairs"]),
        "n_segments": sum(len(p["segments"]) for p in doc["pairs"]),
        "n_clips": len(clips),
        "n_evaluable": sum(c["evaluable"] for c in clips),
        "n_not_evaluable_both": sum(
            all(c["n_future_boxes"][m] == 0 for m in _MODALITIES) for c in clips
        ),
        "shared_across_modalities": True,
        "pairs": pairs,
        "modalities": modalities,
    }


def render_markdown(summary: dict) -> str:
    """The FR10 report as GitHub tables."""
    p = summary["params"]
    out = [
        f"# Clip set `{summary['clip_set']}`",
        "",
        f"Split `{p['split']}`, frame stride {p['frame_stride']}, {p['n_context']} context + "
        f"{p['n_future']} future frames, clip stride {p['clip_stride']}. "
        f"FPS {p['fps']} ({p['fps_source']}). Evaluable rule: {p['evaluable_rule']}.",
        "",
        "The index is shared by both modalities: one row is one time window with the RGB and "
        "the thermal frames of the same time-synced frames, so the clip and sequence counts "
        "below are identical for RGB and thermal.",
        "",
        "## Totals",
        "",
        "| pairs | segments | clips | evaluable | not evaluable (both empty) |",
        "|---:|---:|---:|---:|---:|",
        f"| {summary['n_pairs']} | {summary['n_segments']} | {summary['n_clips']} | "
        f"{summary['n_evaluable']} | {summary['n_clips'] - summary['n_evaluable']} "
        f"({summary['n_not_evaluable_both']}) |",
        "",
        "## Per pair",
        "",
        "| rgb_video_id | thermal_video_id | n_frames | segments | n_clips | n_evaluable | clip_id range |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for q in summary["pairs"]:
        segs = ", ".join(f"[{a}, {b}]" for a, b in q["segments"])
        rng = "–" if q["clip_id_first"] is None else f"{q['clip_id_first']}–{q['clip_id_last']}"
        out.append(
            f"| {q['rgb_video_id']} | {q['thermal_video_id']} | {q['n_frames']} | {segs} | "
            f"{q['n_clips']} | {q['n_evaluable']} | {rng} |"
        )
    out += [
        "",
        "## Per modality",
        "",
        "| modality | frames covered by ≥ 1 clip | not evaluable because this modality has 0 future boxes |",
        "|---|---:|---:|",
    ]
    for m, s in summary["modalities"].items():
        out.append(
            f"| {m} | {s['n_frames_covered']} | {s['n_not_evaluable_caused']} |"
        )
    return "\n".join(out) + "\n"
