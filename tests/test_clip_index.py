"""F3 · Video clip index.

See spec/2026-10-14-video-clip-index/validation.md. Groups 1–5 run on a tiny synthetic test
split (``coco.json`` files and the map only, no JPEGs). Group 6 reads the committed clip set
and needs no data. Group 7 (``-m data``) reads the real dataset under ``$IRWM_DATA_ROOT``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import irwm.data
from irwm.data.clips import (
    DEFAULT_CLIP_SET,
    ClipIndex,
    build_clip_index,
    check_pairing,
    clip_set_name,
    clip_set_path,
    dump_clip_index,
    file_sha256,
    find_segments,
    render_markdown,
    summarise,
    write_clip_index,
)
from irwm.data.flir import FLIR_DIRNAME, FlirSplit, flir_root, load_rgb_thermal_map, split_dir
from irwm.utils.paths import repo_root
from irwm.utils.tracking import git_info

REPO = repo_root()
# SHA-256 of lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10.json (plan Step 6).
COMMITTED_SHA256 = "f9b3735c13cb1ab91eb42ac72bd6b1617eea5e3b54cd40624df8137c729a6c65"

# Copied from tests/test_flir_data.py: car (3) is kept, truck (8) is an ignore region.
CATEGORIES = [
    {"id": 1, "name": "person"}, {"id": 2, "name": "bike"}, {"id": 3, "name": "car"},
    {"id": 4, "name": "motor"}, {"id": 6, "name": "bus"}, {"id": 8, "name": "truck"},
    {"id": 10, "name": "light"}, {"id": 12, "name": "sign"}, {"id": 79, "name": "other vehicle"},
]
CAR, TRUCK = 3, 8
RGB_FRAMES = {"vidA": [*range(0, 5), *range(7, 13)], "vidB": list(range(10))}
PAIRS = {"vidA": "tvZ", "vidB": "tvY"}
# Thermal ids are numbered tvY first, so they differ from the RGB ids of the same frames.
THERMAL_ORDER = ["tvY", "tvZ"]
TEST_PARAMS = {"frame_stride": 2, "n_context": 2, "n_future": 1, "clip_stride": 2}
KEYS = ["vidA:000000", "vidA:000007", "vidB:000000", "vidB:000002", "vidB:000004"]


def _frames(modality: str) -> list[tuple[int, str, int]]:
    """``[(image_id, video_id, frame_index)]`` in image_id order."""
    if modality == "rgb":
        order = [(v, f) for v in sorted(RGB_FRAMES) for f in RGB_FRAMES[v]]
    else:
        rgb_of = {t: v for v, t in PAIRS.items()}
        order = [(t, f) for t in THERMAL_ORDER for f in RGB_FRAMES[rgb_of[t]]]
    return [(i, v, f) for i, (v, f) in enumerate(order)]


def _basename(modality: str, image_id: int, vid: str, frame: int) -> str:
    return f"video-{vid}-frame-{frame:06d}-{modality[0]}{image_id}.jpg"


def _ids(modality: str) -> dict[tuple[str, int], int]:
    return {(v, f): i for i, v, f in _frames(modality)}


def _names(modality: str) -> dict[tuple[str, int], str]:
    return {(v, f): _basename(modality, i, v, f) for i, v, f in _frames(modality)}


@pytest.fixture
def fake_video_test(irwm_env) -> Path:
    """Synthetic ``video_{rgb,thermal}_test`` coco.json files and map. Returns the FLIR root."""
    root = irwm_env["data"] / FLIR_DIRNAME
    for m in ("rgb", "thermal"):
        sdir = split_dir(m, "test", root)
        sdir.mkdir(parents=True)
        images, annotations = [], []
        for i, v, f in _frames(m):
            images.append({"id": i, "file_name": f"data/{_basename(m, i, v, f)}", "width": 64,
                           "height": 48, "extra_info": {"video_id": v}})
            cid = TRUCK if (m == "thermal" and v == "tvY" and f == 6) else CAR
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": cid,
                                "bbox": [1, 2, 10, 10], "area": 100, "iscrowd": False,
                                "track_id": 0})
        (sdir / "coco.json").write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": CATEGORIES})
        )
    rgb_names, th_names = _names("rgb"), _names("thermal")
    mapping = {rgb_names[(v, f)]: th_names[(PAIRS[v], f)] for v in RGB_FRAMES for f in RGB_FRAMES[v]}
    _write_map(root, mapping)
    return root


def _write_map(root: Path, mapping: dict[str, str]) -> None:
    (root / "rgb_to_thermal_vid_map.json").write_text(json.dumps(mapping))


def _edit_coco(root: Path, modality: str, fn) -> None:
    path = split_dir(modality, "test", root) / "coco.json"
    coco = json.loads(path.read_text())
    fn(coco)
    path.write_text(json.dumps(coco))


def _load(root: Path) -> tuple[FlirSplit, FlirSplit, dict[str, str]]:
    return (FlirSplit.load("rgb", "test", root), FlirSplit.load("thermal", "test", root),
            load_rgb_thermal_map(root))


def _build(root: Path, **kw) -> dict:
    return build_clip_index(*_load(root), **{**TEST_PARAMS, **kw})


def _rgb(v: str, f: int) -> str:
    return _names("rgb")[(v, f)]


def _th(v: str, f: int) -> str:
    return _names("thermal")[(v, f)]


# --------------------------------------------------------------------------- Group 1
class TestPairing:
    def test_fixture_pairs(self, fake_video_test):
        assert check_pairing(*_load(fake_video_test)) == [("vidA", "tvZ"), ("vidB", "tvY")]

    def test_missing_entry(self, fake_video_test):
        rgb, th, mp = _load(fake_video_test)
        del mp[_rgb("vidA", 3)]
        with pytest.raises(ValueError, match="size mismatch"):
            check_pairing(rgb, th, mp)

    def test_unknown_value(self, fake_video_test):
        rgb, th, mp = _load(fake_video_test)
        mp[_rgb("vidA", 3)] = "video-ghost-frame-000003-x.jpg"
        with pytest.raises(ValueError, match="video-ghost-frame-000003-x.jpg"):
            check_pairing(rgb, th, mp)

    def test_duplicate_value(self, fake_video_test):
        rgb, th, mp = _load(fake_video_test)
        mp[_rgb("vidA", 3)] = mp[_rgb("vidA", 4)]
        with pytest.raises(ValueError, match="duplicate value"):
            check_pairing(rgb, th, mp)

    def test_frame_index_mismatch(self, fake_video_test):
        rgb, th, mp = _load(fake_video_test)
        key, value = _rgb("vidA", 3), _th("tvZ", 4)
        mp[key] = value
        with pytest.raises(ValueError, match="frame_index mismatch") as e:
            check_pairing(rgb, th, mp)
        assert key in str(e.value) and value in str(e.value)

    def test_video_maps_to_two(self, fake_video_test):
        rgb, th, mp = _load(fake_video_test)
        mp[_rgb("vidA", 3)] = _th("tvY", 3)
        with pytest.raises(ValueError, match="RGB video vidA maps to 2 thermal videos"):
            check_pairing(rgb, th, mp)

    def test_three_violations_one_error(self, fake_video_test):
        rgb, th, mp = _load(fake_video_test)
        ghost = "video-ghost-frame-000001-x.jpg"
        mp[_rgb("vidA", 1)] = ghost
        mismatch_key, mismatch_value = _rgb("vidB", 2), _th("tvY", 5)
        mp[mismatch_key] = mismatch_value
        del mp[_rgb("vidB", 9)]
        with pytest.raises(ValueError) as e:
            check_pairing(rgb, th, mp)
        msg = str(e.value)
        n = int(re.search(r"(\d+) violation", msg).group(1))
        assert n >= 3
        assert ghost in msg and "size mismatch" in msg
        assert "frame_index mismatch" in msg and mismatch_key in msg and mismatch_value in msg

    def test_wrong_split_or_swapped(self, fake_video_test):
        rgb, th, _ = _load(fake_video_test)
        train = FlirSplit("rgb", "train", fake_video_test, rgb.records)
        # An empty map would also fail the map checks; the split check must come first.
        with pytest.raises(ValueError, match="test splits"):
            check_pairing(train, th, {})
        with pytest.raises(ValueError, match=r"\(rgb, thermal\)"):
            check_pairing(th, rgb, {})


# --------------------------------------------------------------------------- Group 2
class TestSegments:
    def test_gap(self):
        idx = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
        assert find_segments(idx) == [(0, 4), (7, 12)]
        shuffled = list(np.random.default_rng(0).permutation(idx))
        assert find_segments(shuffled) == [(0, 4), (7, 12)]

    def test_single_and_empty(self):
        assert find_segments([5]) == [(5, 5)]
        assert find_segments([]) == []

    def test_duplicate(self):
        with pytest.raises(ValueError):
            find_segments([1, 2, 2, 3])


# --------------------------------------------------------------------------- Group 3
class TestBuild:
    def test_ids_and_keys(self, fake_video_test):
        clips = _build(fake_video_test)["clips"]
        assert [c["clip_id"] for c in clips] == list(range(5))
        assert [c["key"] for c in clips] == KEYS

    def test_clip_1(self, fake_video_test):
        c = _build(fake_video_test)["clips"][1]
        assert c["frame_indices"] == [7, 9, 11]
        assert c["segment_index"] == 1
        assert c["thermal_video_id"] == "tvZ"
        rgb_ids, th_ids = _ids("rgb"), _ids("thermal")
        assert c["rgb_image_ids"] == [rgb_ids[("vidA", f)] for f in (7, 9, 11)]
        assert c["thermal_image_ids"] == [th_ids[("tvZ", f)] for f in (7, 9, 11)]
        assert c["rgb_image_ids"] != c["thermal_image_ids"]

    def test_no_window_crosses_gap(self, fake_video_test):
        for c in _build(fake_video_test)["clips"]:
            if c["rgb_video_id"] != "vidA":
                continue
            assert not set(c["frame_indices"]) & {5, 6}
            assert all(f <= 4 for f in c["frame_indices"]) or all(f >= 7 for f in c["frame_indices"])

    def test_short_segment_one_clip(self, fake_video_test):
        seg0 = [c for c in _build(fake_video_test)["clips"]
                if c["rgb_video_id"] == "vidA" and c["segment_index"] == 0]
        assert [c["frame_indices"][0] for c in seg0] == [0]

    def test_evaluable(self, fake_video_test):
        clips = _build(fake_video_test)["clips"]
        for c in clips:
            if c["clip_id"] == 3:
                assert c["n_future_boxes"] == {"rgb": 1, "thermal": 0}
                assert c["evaluable"] is False
            else:
                assert c["n_future_boxes"] == {"rgb": 1, "thermal": 1}
                assert c["evaluable"] is True

    def test_ids_independent_of_labels(self, fake_video_test):
        _edit_coco(fake_video_test, "thermal", lambda c: c.update(annotations=[]))
        clips = _build(fake_video_test)["clips"]
        assert [(c["clip_id"], c["key"]) for c in clips] == list(enumerate(KEYS))
        assert not any(c["evaluable"] for c in clips)

    def test_defaults_give_no_clips(self, fake_video_test):
        doc = build_clip_index(*_load(fake_video_test))
        assert doc["clips"] == []
        assert json.loads(dump_clip_index(doc)) == doc

    @pytest.mark.parametrize("bad", [{"frame_stride": 0}, {"n_future": 0}, {"clip_stride": -1}])
    def test_bad_params(self, fake_video_test, bad):
        with pytest.raises(ValueError):
            _build(fake_video_test, **bad)

    def test_clip_set_name(self):
        assert clip_set_name(5, 4, 6, 10) == "flir_test_fs5_c4f6_cs10"


# --------------------------------------------------------------------------- Group 4
CLIP_KEYS = ["clip_id", "key", "rgb_video_id", "thermal_video_id", "segment_index",
             "frame_indices", "rgb_image_ids", "thermal_image_ids", "rgb_file_names",
             "thermal_file_names", "n_future_boxes", "evaluable"]


class TestDocument:
    def test_key_order(self, fake_video_test):
        doc = _build(fake_video_test)
        assert list(doc) == ["format_version", "clip_set", "params", "inputs", "pairs", "clips"]
        for c in doc["clips"]:
            assert list(c) == CLIP_KEYS

    def test_round_trip_and_layout(self, fake_video_test):
        doc = _build(fake_video_test)
        text = dump_clip_index(doc)
        assert json.loads(text) == doc
        assert text.endswith("}\n") and not text.endswith("\n\n")
        assert sum(line.startswith('    {"clip_id"') for line in text.splitlines()) == len(doc["clips"])

    def test_deterministic(self, fake_video_test):
        assert dump_clip_index(_build(fake_video_test)) == dump_clip_index(_build(fake_video_test))

    def test_no_provenance_in_document(self, fake_video_test, tmp_path):
        text = dump_clip_index(_build(fake_video_test))
        assert str(tmp_path) not in text
        commit = git_info()["commit"]
        assert commit == "unknown" or commit not in text
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", text)

    def test_no_annotations(self, fake_video_test):
        text = dump_clip_index(_build(fake_video_test))
        for field in ('"bbox"', '"category_id"', '"track_id"'):
            assert field not in text

    def test_write_guard(self, fake_video_test, tmp_path):
        doc = _build(fake_video_test)
        path = tmp_path / "a" / "b" / "set.json"
        assert write_clip_index(doc, path) is True
        assert path.read_text(encoding="utf-8") == dump_clip_index(doc)
        mtime, data = path.stat().st_mtime_ns, path.read_bytes()
        assert write_clip_index(doc, path) is False
        assert path.stat().st_mtime_ns == mtime
        with pytest.raises(FileExistsError):
            write_clip_index(_build(fake_video_test, clip_stride=3), path)
        assert path.read_bytes() == data

    def test_clip_set_path(self):
        assert clip_set_path("x") == Path(irwm.data.__file__).parent / "clip_sets" / "x.json"


# --------------------------------------------------------------------------- Group 5
@pytest.fixture
def fake_index(fake_video_test, tmp_path) -> tuple[ClipIndex, dict, Path]:
    doc = _build(fake_video_test)
    path = tmp_path / "idx.json"
    write_clip_index(doc, path)
    return ClipIndex.load(path=path), doc, path


class TestReader:
    def test_load(self, fake_index):
        idx, _, _ = fake_index
        assert len(idx) == 5
        assert idx[3].key == "vidB:000002"
        assert idx.by_key("vidA:000007").clip_id == 1
        assert [c.clip_id for c in idx.evaluable()] == [0, 1, 2, 4]
        assert [c.clip_id for c in idx] == list(range(5))

    def test_lookup_errors(self, fake_index):
        idx, _, _ = fake_index
        with pytest.raises(IndexError):
            idx[5]
        with pytest.raises(IndexError):
            idx[-1]
        with pytest.raises(KeyError):
            idx.by_key("nope:000000")

    def test_ids(self, fake_index):
        c = fake_index[0][1]
        ids = c.image_ids("thermal")
        assert ids.dtype == np.int64 and ids.shape == (3,)
        assert c.context_ids("thermal").shape == (2,)
        assert c.future_ids("thermal").shape == (1,)
        assert np.array_equal(np.concatenate([c.context_ids("thermal"), c.future_ids("thermal")]), ids)

    def test_video_and_file_names(self, fake_index):
        c = fake_index[0][1]
        assert c.video_id("rgb") == "vidA" and c.video_id("thermal") == "tvZ"
        assert c.file_names("rgb")[0].startswith("video-vidA-frame-000007-")
        assert not c.file_names("rgb")[0].startswith("data/")
        assert c.n_future_boxes("thermal") == 1

    def test_bad_modality(self, fake_index):
        with pytest.raises(ValueError):
            fake_index[0][1].image_ids("ir")

    def test_frozen(self, fake_index):
        with pytest.raises(dataclasses.FrozenInstanceError):
            fake_index[0][0].clip_id = 9

    @pytest.mark.parametrize("corrupt", ["version", "swapped_ids", "dup_key", "short_list"])
    def test_corrupt(self, fake_index, tmp_path, corrupt):
        _, doc, _ = fake_index
        doc = json.loads(json.dumps(doc))
        if corrupt == "version":
            doc["format_version"] = 2
        elif corrupt == "swapped_ids":
            doc["clips"][0]["clip_id"], doc["clips"][1]["clip_id"] = 1, 0
        elif corrupt == "dup_key":
            doc["clips"][1]["key"] = doc["clips"][0]["key"]
        else:
            doc["clips"][2]["frame_indices"] = doc["clips"][2]["frame_indices"][:2]
        path = tmp_path / f"{corrupt}.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(ValueError):
            ClipIndex.load(path=path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ClipIndex.load(path=tmp_path / "missing.json")

    def test_no_data_root_needed(self, fake_index):
        _, _, path = fake_index
        env = {k: v for k, v in os.environ.items() if not k.startswith("IRWM_")}
        code = ("import sys; from pathlib import Path; from irwm.data.clips import ClipIndex; "
                "print(len(ClipIndex.load(path=Path(sys.argv[1]))))")
        # Run outside the repo so load_env() could not pick up the local .env either.
        out = subprocess.run([sys.executable, "-c", code, str(path)], env=env, cwd=path.parent,
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "5"


class TestSummary:
    def test_counts_and_markdown(self, fake_video_test):
        s = summarise(_build(fake_video_test))
        assert (s["n_pairs"], s["n_segments"], s["n_clips"], s["n_evaluable"]) == (2, 3, 5, 4)
        assert s["modalities"]["thermal"]["n_not_evaluable_caused"] == 1
        assert s["modalities"]["rgb"]["n_not_evaluable_caused"] == 0
        assert s["n_not_evaluable_both"] == 0
        assert [(p["clip_id_first"], p["clip_id_last"]) for p in s["pairs"]] == [(0, 1), (2, 4)]
        md = render_markdown(s)
        assert "identical for RGB and thermal" in md
        assert "| vidA | tvZ | 11 | [0, 4], [7, 12] | 2 | 2 | 0–1 |" in md
        json.dumps(s)


# --------------------------------------------------------------------------- Group 6 (committed)
COMMITTED = clip_set_path(DEFAULT_CLIP_SET)
needs_committed = pytest.mark.skipif(
    not COMMITTED.is_file(), reason=f"{COMMITTED.name} not built yet (plan Step 5)"
)


@needs_committed
class TestCommitted:
    def test_sha256(self):
        assert file_sha256(COMMITTED) == COMMITTED_SHA256

    def test_counts(self):
        idx = ClipIndex.load()
        assert len(idx) == 330
        assert len(idx.evaluable()) == 304
        assert len(idx.pairs) == 8

    def test_clip_15(self):
        c = ClipIndex.load()[15]
        assert c.key == "BzZspxAweF8AnKhWK:000895"
        assert c.thermal_video_id == "4FRnNpmSmwktFJKjg"
        assert list(c.frame_indices) == list(range(895, 941, 5))
        assert list(c.rgb_image_ids) == list(range(150, 196, 5))
        assert list(c.thermal_image_ids) == list(range(150, 196, 5))

    def test_landmarks(self):
        idx = ClipIndex.load()
        assert idx[296].key == "msNEBxJE5PPDqenBM:006393"
        assert idx[329].rgb_video_id == "msNEBxJE5PPDqenBM"

    def test_size(self):
        assert COMMITTED.stat().st_size < 1_000_000

    def test_not_git_ignored(self):
        r = subprocess.run(["git", "check-ignore", str(COMMITTED.relative_to(REPO))], cwd=REPO,
                           capture_output=True)
        assert r.returncode == 1

    def test_no_eol_conversion(self):
        # `* text=auto` would let a Windows checkout rewrite line ends and break the pinned SHA-256.
        r = subprocess.run(["git", "check-attr", "text", str(COMMITTED.relative_to(REPO))], cwd=REPO,
                           capture_output=True, text=True, check=True)
        assert r.stdout.strip().endswith("text: unset")


# --------------------------------------------------------------------------- Group 7 (real data)
@pytest.fixture(scope="module")
def real():
    from irwm.utils.paths import load_env

    load_env()
    if not os.environ.get("IRWM_DATA_ROOT"):
        pytest.skip("IRWM_DATA_ROOT is not set")
    root = flir_root()
    if not root.is_dir():
        pytest.skip(f"{root} does not exist")
    return {"root": root, "rgb": FlirSplit.load("rgb", "test", root),
            "thermal": FlirSplit.load("thermal", "test", root), "map": load_rgb_thermal_map(root)}


@pytest.mark.data
class TestRealData:
    def test_pairing(self, real):
        assert len(check_pairing(real["rgb"], real["thermal"], real["map"])) == 8

    @needs_committed
    def test_rebuild_is_byte_identical(self, real):
        root = real["root"]
        inputs = {
            "rgb_coco_sha256": file_sha256(split_dir("rgb", "test", root) / "coco.json"),
            "thermal_coco_sha256": file_sha256(split_dir("thermal", "test", root) / "coco.json"),
            "rgb_to_thermal_map_sha256": file_sha256(root / "rgb_to_thermal_vid_map.json"),
        }
        doc = build_clip_index(real["rgb"], real["thermal"], real["map"], inputs=inputs)
        assert dump_clip_index(doc).encode("utf-8") == COMMITTED.read_bytes()

    @needs_committed
    def test_ids_resolve(self, real):
        for c in ClipIndex.load():
            for m in ("rgb", "thermal"):
                for i, f, name in zip(c.image_ids(m), c.frame_indices, c.file_names(m)):
                    r = real[m].by_image_id(int(i))
                    assert (r.video_id, r.frame_index, Path(r.file_name).name) == (c.video_id(m), f, name)

    @needs_committed
    def test_box_counts(self, real):
        for c in ClipIndex.load():
            for m in ("rgb", "thermal"):
                assert c.n_future_boxes(m) == sum(len(real[m].by_image_id(int(i)).boxes)
                                                  for i in c.future_ids(m))
