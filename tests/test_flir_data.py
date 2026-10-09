"""F2 · FLIR ADAS v2 readers and class mapping.

See spec/2026-10-08-flir-readers-class-mapping/validation.md. Groups 1–6 run on a tiny
synthetic tree with the as-downloaded layout; Group 7 (``-m data``) reads the real dataset
under ``$IRWM_DATA_ROOT``.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from PIL import Image
from torch.utils.data import DataLoader

from irwm.data import classes as C
from irwm.data.flir import (
    FLIR_DIRNAME,
    FlirImageDataset,
    FlirSplit,
    flir_root,
    load_rgb_thermal_map,
    split_dir,
)
from irwm.data.stats import COCO80_NAME, compute_stats, render_markdown
from irwm.utils.paths import repo_root

REPO = repo_root()
STATS_CONF = REPO / "projects" / "data_stats" / "conf"

CATEGORIES = [
    {"id": 1, "name": "person"}, {"id": 2, "name": "bike"}, {"id": 3, "name": "car"},
    {"id": 4, "name": "motor"}, {"id": 6, "name": "bus"}, {"id": 8, "name": "truck"},
    {"id": 10, "name": "light"}, {"id": 12, "name": "sign"}, {"id": 79, "name": "other vehicle"},
]
# (category_id, bbox) in coco.json order: car, person, truck.
ANNS_A = [(3, [1, 2, 10, 10]), (1, [5, 5, 4, 8]), (8, [20, 20, 30, 20])]
ANNS_PERSON = [(1, [3, 3, 6, 6])]
SIZE = {"rgb": (80, 64), "thermal": (64, 48)}
MODE = {"rgb": "RGB", "thermal": "L"}
# (modality, split) → [(image_id, video_id, frame_index, annotations, (W, H) override)]
FAKE = {
    ("rgb", "train"): [(3, "trA", 0, ANNS_A, None), (7, "trA", 1, [], None)],
    ("rgb", "val"): [(1, "vaA", 0, ANNS_A, None), (2, "vaA", 1, [], None)],
    ("rgb", "test"): [(0, "vidR", 0, ANNS_A, None), (1, "vidR", 1, ANNS_PERSON, None),
                      (2, "vidR", 2, [], (64, 48))],
    ("thermal", "train"): [(0, "trB", 0, ANNS_A, None), (1, "trB", 1, [], None)],
    ("thermal", "val"): [(5, "vaB", 0, ANNS_A, None), (9, "vaB", 1, [], None)],
    ("thermal", "test"): [(0, "vidT", 0, ANNS_A, None), (1, "vidT", 1, ANNS_PERSON, None),
                          (2, "vidT", 2, [], None), (3, "vidT", 5, [], None),
                          (4, "vidT", 6, [], None)],
}
TAGS = {"rgb": {"vidR": ["day", "city"]}, "thermal": {"vidT": ["night"]}}
RGB_THERMAL_MAP = {"video-vidR-frame-000000-u0.jpg": "video-vidT-frame-000000-u0.jpg"}
ORPHAN = "video-orphan1-frame-000000-zz.jpg"


def _file_name(video_id: str, frame: int, uid: str) -> str:
    return f"data/video-{video_id}-frame-{frame:06d}-{uid}.jpg"


def _write_split(root: Path, modality: str, split: str, items: list) -> dict:
    sdir = split_dir(modality, split, root)
    (sdir / "data").mkdir(parents=True)
    images, annotations, ann_id = [], [], 1
    for image_id, vid, frame, anns, size in items:
        w, h = size or SIZE[modality]
        fname = _file_name(vid, frame, f"u{image_id}")
        Image.new(MODE[modality], (w, h), color=128 if modality == "thermal" else (10, 20, 30)).save(
            sdir / fname
        )
        images.append({"id": image_id, "file_name": fname, "width": w, "height": h,
                       "extra_info": {"video_id": vid}, "subdirs": "data"})
        for k, (cid, bbox) in enumerate(anns):
            ann = {"id": ann_id, "image_id": image_id, "category_id": cid, "bbox": bbox,
                   "area": bbox[2] * bbox[3], "iscrowd": False}
            if split == "test":
                ann["track_id"] = 100 * image_id + k
            annotations.append(ann)
            ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    (sdir / "coco.json").write_text(json.dumps(coco))
    if split == "test":
        videos = [{"id": v, "tags": t} for v, t in TAGS[modality].items()]
        (sdir / "index.json").write_text(json.dumps({"videos": videos, "frames": []}))
    return coco


@pytest.fixture
def fake_flir(irwm_env) -> Path:
    """A tiny FLIR tree at ``$IRWM_DATA_ROOT/teledyne_flir_adas_v2``. Returns its root."""
    root = irwm_env["data"] / FLIR_DIRNAME
    for (modality, split), items in FAKE.items():
        _write_split(root, modality, split, items)
    (root / "rgb_to_thermal_vid_map.json").write_text(json.dumps(RGB_THERMAL_MAP))
    Image.new("RGB", (80, 64)).save(split_dir("rgb", "train", root) / "data" / ORPHAN)
    return root


def _edit_coco(root: Path, modality: str, split: str, fn) -> None:
    path = split_dir(modality, split, root) / "coco.json"
    coco = json.loads(path.read_text())
    fn(coco)
    path.write_text(json.dumps(coco))


def _all_splits(root: Path) -> list[FlirSplit]:
    return [FlirSplit.load(m, s, root) for m in ("rgb", "thermal") for s in ("train", "val", "test")]


# --------------------------------------------------------------------------- Group 1
class TestRootAndConfig:
    def test_root_and_split_dir(self, irwm_env):
        tmp = irwm_env["data"].resolve()
        assert flir_root() == tmp / "teledyne_flir_adas_v2"
        assert split_dir("thermal", "test") == tmp / "teledyne_flir_adas_v2" / "FLIR_ADAS_v2" / "video_thermal_test"

    def test_unknown_modality_or_split(self, irwm_env):
        with pytest.raises(ValueError) as exc:
            split_dir("ir", "train")
        assert "rgb" in str(exc.value) and "thermal" in str(exc.value)
        with pytest.raises(ValueError) as exc:
            split_dir("rgb", "video")
        assert all(s in str(exc.value) for s in ("train", "val", "test"))

    def test_missing_coco_json(self, irwm_env):
        with pytest.raises(FileNotFoundError) as exc:
            FlirSplit.load("rgb", "train")
        assert str(split_dir("rgb", "train") / "coco.json") in str(exc.value)

    def test_hydra_flir_root_matches_dirname(self, irwm_env):
        with initialize_config_dir(config_dir=str(STATS_CONF), version_base="1.3"):
            cfg = compose("stats")
        assert cfg.flir.root == str(flir_root())
        assert cfg.project == "data_stats"
        assert list(cfg.modalities) == ["rgb", "thermal"]
        assert list(cfg.splits) == ["train", "val", "test"] and cfg.check_files is True

    def test_dirname_only_in_two_places(self):
        hits = []
        for top in ("lib", "projects"):
            for f in (REPO / top).rglob("*"):
                if not f.is_file() or "__pycache__" in f.parts or f.suffix not in {".py", ".yaml", ".toml", ".md"}:
                    continue
                for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if "teledyne" in line:
                        hits.append((f.relative_to(REPO).as_posix(), line.strip()))
                    assert "/home/" not in line, f"absolute path in {f}:{i}"
        assert sorted(h[0] for h in hits) == ["lib/irwm/conf/flir/default.yaml", "lib/irwm/data/flir.py"]
        assert [h[1] for h in hits if h[0].endswith("flir.py")] == ['FLIR_DIRNAME = "teledyne_flir_adas_v2"']


# --------------------------------------------------------------------------- Group 2
class TestLabelSpace:
    def test_constants(self):
        assert C.CLASS_NAMES == ("person", "bike", "car", "motor", "light", "sign")
        assert C.NUM_CLASSES == 6
        assert C.CATEGORY_ID_TO_LABEL == {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 12: 5}
        assert C.LABEL_TO_CATEGORY_ID == {v: k for k, v in C.CATEGORY_ID_TO_LABEL.items()}
        assert len(C.LABEL_TO_CATEGORY_ID) == 6

    def test_ignore_categories(self):
        for cid in (8, 6, 79, 999):
            assert C.label_of(cid) is None
        assert C.label_of(12) == 5

    def test_normalised_area(self):
        assert C.normalised_area(10, 10, 640, 512) == 100.0
        assert C.normalised_area(20, 20, 1280, 1024) == 100.0

    def test_size_bucket(self):
        assert C.size_bucket(1023.9) == "S"
        assert C.size_bucket(1024) == "M"
        assert C.size_bucket(9215.9) == "M"
        assert C.size_bucket(9216) == "L"

    def test_no_coarse_mapping(self):
        banned = ("coarse", "vehicle", "pedestrian", "cyclist")
        for mod in ("classes.py", "flir.py"):
            tree = ast.parse((REPO / "lib" / "irwm" / "data" / mod).read_text(encoding="utf-8"))
            names = []
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    names += [t.id for t in ast.walk(node) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.append(node.target.id)
            assert names, mod
            assert not [n for n in names if any(b in n.lower() for b in banned)], (mod, names)


# --------------------------------------------------------------------------- Group 3
class TestFlirSplit:
    def test_len_and_ids(self, fake_flir):
        s = FlirSplit.load("thermal", "test", fake_flir)
        assert len(s) == len(FAKE[("thermal", "test")]) == 5
        ids = s.image_ids()
        assert ids.dtype == np.int64 and bool((np.diff(ids) > 0).all())
        assert [r.image_id for r in s] == ids.tolist()
        assert s.split_dir == split_dir("thermal", "test", fake_flir) and s.root == fake_flir

    def test_kept_and_ignored(self, fake_flir):
        r = FlirSplit.load("rgb", "train", fake_flir).by_image_id(3)
        assert r.boxes.shape == (2, 4) and r.labels.tolist() == [2, 0]
        assert r.boxes.tolist() == [[1, 2, 10, 10], [5, 5, 4, 8]]
        assert r.ignore_boxes.shape == (1, 4) and r.ignore_category_ids.tolist() == [8]
        assert (r.boxes.dtype, r.labels.dtype, r.ignore_boxes.dtype, r.ignore_category_ids.dtype) == (
            np.float32, np.int64, np.float32, np.int64)
        assert r.ann_ids.dtype == np.int64 and r.ann_ids.tolist() == [1, 2]
        assert r.path.is_absolute() and r.path.is_file()
        assert (r.width, r.height) == (80, 64)

    def test_empty_image(self, fake_flir):
        r = FlirSplit.load("rgb", "train", fake_flir).by_image_id(7)
        assert r.boxes.shape == (0, 4) and r.labels.shape == (0,)
        assert r.ignore_boxes.shape == (0, 4) and r.ignore_category_ids.shape == (0,)
        assert r.track_ids.shape == (0,) and r.ann_ids.shape == (0,)

    def test_track_ids(self, fake_flir):
        test = FlirSplit.load("thermal", "test", fake_flir)
        assert test.by_image_id(0).track_ids.tolist() == [0, 1]  # car, person (truck ignored)
        assert test.by_image_id(1).track_ids.tolist() == [100]
        train = FlirSplit.load("thermal", "train", fake_flir)
        assert all((r.track_ids == -1).all() for r in train)
        assert train.by_image_id(0).track_ids.tolist() == [-1, -1]

    def test_video_id_and_frame_parsed(self, fake_flir):
        def rename(coco):
            coco["images"][0]["file_name"] = "data/video-abc123-frame-000042-xyz.jpg"
            coco["images"][0]["extra_info"]["video_id"] = "abc123"

        _edit_coco(fake_flir, "rgb", "val", rename)
        r = FlirSplit.load("rgb", "val", fake_flir).by_image_id(1)
        assert (r.video_id, r.frame_index) == ("abc123", 42)
        assert r.file_name == "data/video-abc123-frame-000042-xyz.jpg"

    @pytest.mark.parametrize("case", ["file_name", "video_id", "duplicate", "category", "zero_w"])
    def test_malformed(self, fake_flir, case):
        def edit(coco):
            if case == "file_name":
                coco["images"][0]["file_name"] = "data/frame-0001.jpg"
            elif case == "video_id":
                coco["images"][0]["extra_info"]["video_id"] = "other"
            elif case == "duplicate":
                coco["images"][1]["id"] = coco["images"][0]["id"]
            elif case == "category":
                coco["annotations"][0]["category_id"] = 55
            elif case == "zero_w":
                coco["annotations"][0]["bbox"][2] = 0

        _edit_coco(fake_flir, "rgb", "train", edit)
        with pytest.raises(ValueError) as exc:
            FlirSplit.load("rgb", "train", fake_flir)
        msg = str(exc.value)
        expected = {"file_name": "image 3", "video_id": "image 3", "duplicate": "image_id 3",
                    "category": "annotation 1", "zero_w": "annotation 1"}[case]
        assert expected in msg, msg

    def test_by_image_id(self, fake_flir):
        s = FlirSplit.load("thermal", "val", fake_flir)
        assert s.by_image_id(9).image_id == 9
        with pytest.raises(KeyError):
            s.by_image_id(-5)

    def test_video_tags(self, fake_flir):
        assert FlirSplit.load("rgb", "test", fake_flir).video_tags() == TAGS["rgb"]
        assert FlirSplit.load("thermal", "test", fake_flir).video_tags() == TAGS["thermal"]
        with pytest.raises(ValueError):
            FlirSplit.load("rgb", "train", fake_flir).video_tags()

    def test_rgb_thermal_map(self, fake_flir):
        assert load_rgb_thermal_map(fake_flir) == RGB_THERMAL_MAP
        assert load_rgb_thermal_map() == RGB_THERMAL_MAP  # default root = flir_root()


# --------------------------------------------------------------------------- Group 4
class TestCocoExport:
    def test_counts_and_crowd_copies(self, fake_flir):
        s = FlirSplit.load("thermal", "test", fake_flir)
        K = sum(len(r.labels) for r in s)
        I = sum(len(r.ignore_boxes) for r in s)
        assert (K, I) == (3, 1)
        gt = s.to_coco_gt()
        anns = gt["annotations"]
        assert len(anns) == K + 6 * I
        crowd = [a for a in anns if a["iscrowd"] == 1]
        assert len(crowd) == 6 * I
        assert {a["category_id"] for a in crowd} == {1, 2, 3, 4, 10, 12}
        assert all(a["bbox"] == [20.0, 20.0, 30.0, 20.0] for a in crowd)
        assert [im["id"] for im in gt["images"]] == s.image_ids().tolist()

    def test_ids(self, fake_flir):
        s = FlirSplit.load("rgb", "test", fake_flir)
        gt = s.to_coco_gt()
        ids = [a["id"] for a in gt["annotations"]]
        assert len(ids) == len(set(ids))
        kept = [a for a in gt["annotations"] if a["iscrowd"] == 0]
        orig = [int(i) for r in s for i in r.ann_ids]
        assert [a["id"] for a in kept] == orig
        assert sorted(a["id"] for a in gt["annotations"] if a["iscrowd"] == 1) == list(
            range(max(orig) + 1, max(orig) + 7))
        assert [a["category_id"] for a in kept] == [3, 1, 1]

    def test_categories(self, fake_flir):
        gt = FlirSplit.load("rgb", "val", fake_flir).to_coco_gt()
        assert gt["categories"] == [
            {"id": 1, "name": "person"}, {"id": 2, "name": "bike"}, {"id": 3, "name": "car"},
            {"id": 4, "name": "motor"}, {"id": 10, "name": "light"}, {"id": 12, "name": "sign"}]

    def test_area(self, fake_flir):
        s = FlirSplit.load("rgb", "train", fake_flir)
        car = lambda gt: next(a for a in gt["annotations"] if a["id"] == 1)  # noqa: E731
        assert car(s.to_coco_gt())["area"] == 100 * 640 * 512 / (80 * 64) == 6400.0
        assert car(s.to_coco_gt(area="native"))["area"] == 100.0

    def test_json_serialisable(self, fake_flir):
        for s in _all_splits(fake_flir):
            json.dumps(s.to_coco_gt())
            json.dumps(s.to_coco_gt(area="native"))
    # pycocotools is not a dependency yet (F6), so the export is checked structurally only.


# --------------------------------------------------------------------------- Group 5
class TestImageDataset:
    def test_decode(self, fake_flir):
        th = FlirImageDataset(FlirSplit.load("thermal", "train", fake_flir))
        item = th[0]
        assert item["image"].dtype == np.uint8 and item["image"].shape == (48, 64)
        assert item["image_id"] == th.split.records[0].image_id and isinstance(item["image_id"], int)
        rgb = FlirImageDataset(FlirSplit.load("rgb", "test", fake_flir))
        assert rgb[0]["image"].shape == (64, 80, 3) and rgb[0]["image"].dtype == np.uint8
        assert rgb[2]["image"].shape == (48, 64, 3)  # mixed sizes in RGB test
        assert len(rgb) == 3

    def test_transform(self, fake_flir):
        ds = FlirImageDataset(FlirSplit.load("thermal", "val", fake_flir), transform=lambda a: a.shape)
        assert ds[0]["image"] == (48, 64)

    def test_wrong_mode(self, fake_flir):
        s = FlirSplit.load("thermal", "train", fake_flir)
        Image.new("RGB", (64, 48)).save(s.records[0].path)
        with pytest.raises(ValueError, match="mode"):
            FlirImageDataset(s)[0]

    def test_wrong_size(self, fake_flir):
        _edit_coco(fake_flir, "thermal", "train", lambda c: c["images"][0].update(width=65))
        with pytest.raises(ValueError, match="size"):
            FlirImageDataset(FlirSplit.load("thermal", "train", fake_flir))[0]

    def test_pickle_and_dataloader(self, fake_flir):
        ds = FlirImageDataset(FlirSplit.load("rgb", "test", fake_flir))
        clone = pickle.loads(pickle.dumps(ds))
        assert np.array_equal(clone[0]["image"], ds[0]["image"])
        items = list(DataLoader(ds, batch_size=None, num_workers=0))
        assert [int(it["image_id"]) for it in items] == [0, 1, 2]


# --------------------------------------------------------------------------- Group 6
class TestStats:
    def test_counts(self, fake_flir):
        st = compute_stats(_all_splits(fake_flir))
        assert {k: v["n_images"] for k, v in st["images"].items()} == {
            "rgb/train": 2, "rgb/val": 2, "rgb/test": 3,
            "thermal/train": 2, "thermal/val": 2, "thermal/test": 5}
        assert st["images"]["rgb/test"]["sizes"] == {"80x64": 2, "64x48": 1}
        assert st["images"]["thermal/test"]["no_kept_box"] == 3
        assert st["images"]["thermal/test"]["no_annotation"] == 3
        sel = st["selected"]
        assert sel["thermal/test"]["kept"] == {"person": 2, "bike": 0, "car": 1, "motor": 0, "light": 0, "sign": 0}
        assert sel["rgb/train"]["kept"]["person"] == 1 and sel["rgb/train"]["total_kept"] == 2
        assert sel["rgb/val"]["ignore"] == {"truck": 1} and sel["rgb/val"]["total_ignore"] == 1
        # Per-class rows: only classes with ≥1 box (person, car, truck), one row per split.
        rows = {(c["name"], c["modality"], c["split"]): c for c in st["classes"]}
        assert {c["name"] for c in st["classes"]} == {"person", "car", "truck"}
        assert len(rows) == 3 * 6
        car = rows[("car", "rgb", "train")]
        assert (car["boxes"], car["images"], car["selected"], car["coco80"]) == (1, 1, True, True)
        assert car["share_pct"] == pytest.approx(100 / 3, abs=1e-3)
        assert car["size_native"] == {"S": 1, "M": 0, "L": 0}
        assert car["size_normalised"] == {"S": 0, "M": 1, "L": 0}  # 6400 normalised
        assert rows[("truck", "thermal", "test")]["selected"] is False
        assert car["underrepresented"] is True  # < 100 train boxes
        assert car["absent_val"] is False and car["absent_test"] is False
        assert st["modality_comparison"]["person"]["test"] == 1.0
        fi = st["file_integrity"]
        assert fi["rgb/train"]["orphans"] == [f"images_rgb_train/data/{ORPHAN}"]
        assert sum(len(v["orphans"]) for v in fi.values()) == 1
        assert all(not v["size_mismatches"] and not v["mode_mismatches"] for v in fi.values())

    def test_sequence_gaps(self, fake_flir):
        st = compute_stats(_all_splits(fake_flir), check_files=False)
        (t,) = st["sequences"]["thermal"]
        assert (t["video_id"], t["n_frames"], t["frame_min"], t["frame_max"]) == ("vidT", 5, 0, 6)
        assert (t["n_gaps"], t["gap_sizes"], t["segments"]) == (1, [3], [3, 2])
        assert t["tags"] == ["night"]
        (r,) = st["sequences"]["rgb"]
        assert (r["n_gaps"], r["segments"], r["tags"]) == (0, [3], ["day", "city"])

    def test_file_integrity_mismatch(self, fake_flir):
        _edit_coco(fake_flir, "thermal", "val", lambda c: c["images"][1].update(height=50))
        st = compute_stats([FlirSplit.load("thermal", "val", fake_flir)])
        mm = st["file_integrity"]["thermal/val"]["size_mismatches"]
        assert [m["image_id"] for m in mm] == [9] and mm[0]["file"] == [64, 48]

    def test_check_files_false_opens_nothing(self, fake_flir, monkeypatch):
        splits = _all_splits(fake_flir)

        def boom(*a, **k):
            raise AssertionError("image file opened")

        monkeypatch.setattr(Image, "open", boom)
        st = compute_stats(splits, check_files=False)
        assert "file_integrity" not in st

    def test_coco80_table(self, fake_flir):
        assert set(COCO80_NAME) == {
            "person", "bike", "car", "motor", "bus", "train", "truck", "light", "hydrant",
            "sign", "dog", "deer", "skateboard", "stroller", "scooter", "other vehicle"}
        assert COCO80_NAME["sign"] is None and COCO80_NAME["motor"] == "motorcycle"

        def unicorn(coco):
            coco["categories"].append({"id": 50, "name": "unicorn"})
            coco["annotations"][0]["category_id"] = 50

        _edit_coco(fake_flir, "rgb", "val", unicorn)
        with pytest.raises(KeyError, match="unicorn"):
            compute_stats([FlirSplit.load("rgb", "val", fake_flir)], check_files=False)

    def test_render_markdown(self, fake_flir):
        st = compute_stats(_all_splits(fake_flir))
        md = render_markdown(st)
        for h in ("## 1. Images", "## 2. Per raw FLIR class", "## 3. Modality comparison",
                  "## 4. Selected 6 + ignore", "## 5. Video sequences", "## 6. File integrity",
                  "## 6-class summary"):
            assert h in md
        per_class = md.split("## 2.")[1].split("## 3.")[0]
        for name in ("person", "car", "truck"):
            assert f"| {name} |" in per_class
        assert ORPHAN in md

    def test_render_markdown_all_16_names(self):
        """Every one of the 16 raw class names appears in the per-class table."""
        rows = [{"name": n, "category_id": i, "modality": "rgb", "split": "train", "boxes": 0,
                 "images": 0, "share_pct": 0.0, "size_native": {"S": 0, "M": 0, "L": 0},
                 "size_normalised": {"S": 0, "M": 0, "L": 0}, "selected": False, "coco80": False,
                 "underrepresented": True, "absent_val": None, "absent_test": None}
                for i, n in enumerate(COCO80_NAME)]
        st = {"images": {}, "classes": rows, "modality_comparison": {}, "sequences": {},
              "selected": {"rgb/train": {"kept": dict.fromkeys(C.CLASS_NAMES, 0), "ignore": {},
                                         "total_kept": 0, "total_ignore": 0}}}
        per_class = render_markdown(st).split("## 2.")[1].split("## 3.")[0]
        for name in COCO80_NAME:
            assert f"| {name} |" in per_class

    def test_json_serialisable(self, fake_flir):
        json.dumps(compute_stats(_all_splits(fake_flir)))


# --------------------------------------------------------------------------- Group 7 (real data)
@pytest.fixture(scope="module")
def real():
    from irwm.utils.paths import load_env

    load_env()
    if not os.environ.get("IRWM_DATA_ROOT"):
        pytest.skip("IRWM_DATA_ROOT is not set")
    if not flir_root().is_dir():
        pytest.skip(f"{flir_root()} does not exist")
    return {(m, s): FlirSplit.load(m, s) for m in ("rgb", "thermal") for s in ("train", "val", "test")}


@pytest.mark.data
class TestRealData:
    def test_image_counts(self, real):
        got = {k: len(v) for k, v in real.items()}
        assert got == {("rgb", "train"): 10318, ("rgb", "val"): 1085, ("rgb", "test"): 3749,
                       ("thermal", "train"): 10742, ("thermal", "val"): 1144, ("thermal", "test"): 3749}

    def test_kept_ignore_counts(self, real):
        got = {k: (sum(len(r.labels) for r in s), sum(len(r.ignore_boxes) for r in s))
               for k, s in real.items()}
        assert got == {
            ("rgb", "train"): (163856, 5318), ("rgb", "val"): (16502, 407), ("rgb", "test"): (81221, 3565),
            ("thermal", "train"): (169422, 5618), ("thermal", "val"): (16305, 391),
            ("thermal", "test"): (58685, 3632)}

    def test_kept_per_class(self, real):
        expected = {
            "rgb": {"person": (35007, 3223, 11278), "bike": (7560, 193, 446), "car": (71281, 7285, 30893),
                    "motor": (1837, 77, 3573), "light": (18640, 2143, 17818), "sign": (29531, 3581, 17213)},
            "thermal": {"person": (50478, 4470, 12323), "bike": (7237, 170, 113),
                        "car": (73623, 7133, 30517), "motor": (1116, 55, 3314),
                        "light": (16198, 2005, 6758), "sign": (20770, 2472, 5660)},
        }
        for m, per_class in expected.items():
            for label, name in enumerate(C.CLASS_NAMES):
                got = tuple(sum(int((r.labels == label).sum()) for r in real[(m, s)])
                            for s in ("train", "val", "test"))
                assert got == per_class[name], (m, name, got)

    def test_empty_images(self, real):
        no_ann = {k: sum(len(r.labels) == 0 and len(r.ignore_boxes) == 0 for r in s) for k, s in real.items()}
        no_kept = {k: sum(len(r.labels) == 0 for r in s) for k, s in real.items()}
        sp = ("train", "val", "test")
        assert [no_ann[("rgb", s)] for s in sp] == [265, 16, 0]
        assert [no_ann[("thermal", s)] for s in sp] == [264, 16, 256]
        assert [no_kept[("rgb", s)] for s in sp] == [280, 16, 98]
        assert [no_kept[("thermal", s)] for s in sp] == [280, 19, 256]

    def test_normalised_size_buckets(self, real):
        expected = {
            ("rgb", "train"): (125708, 32608, 5540), ("rgb", "val"): (13499, 2476, 527),
            ("rgb", "test"): (72480, 7100, 1641), ("thermal", "train"): (126045, 36650, 6727),
            ("thermal", "val"): (12218, 3395, 692), ("thermal", "test"): (50312, 7162, 1211)}
        for k, s in real.items():
            norm = {"S": 0, "M": 0, "L": 0}
            native = {"S": 0, "M": 0, "L": 0}
            for r in s:
                for b in r.boxes:
                    w, h = float(b[2]), float(b[3])
                    norm[C.size_bucket(C.normalised_area(w, h, r.width, r.height))] += 1
                    native[C.size_bucket(w * h)] += 1
            assert (norm["S"], norm["M"], norm["L"]) == expected[k], k
            if k[0] == "thermal":
                assert norm == native

    def test_sequences_and_gt_export(self, real):
        for m in ("rgb", "thermal"):
            frames: dict[str, int] = {}
            for r in real[(m, "test")]:
                frames[r.video_id] = frames.get(r.video_id, 0) + 1
            assert len(frames) == 8
            assert sorted(frames.values(), reverse=True) == [1033, 565, 540, 428, 411, 338, 226, 208]
        assert len(real[("thermal", "test")].to_coco_gt()["annotations"]) == 58685 + 6 * 3632 == 80477
        assert len(real[("rgb", "test")].to_coco_gt()["annotations"]) == 81221 + 6 * 3565 == 102611

    def test_raw_classes(self, real):
        st = compute_stats(list(real.values()), check_files=False)
        assert {c["name"] for c in st["classes"]} == {
            "person", "bike", "car", "motor", "bus", "train", "truck", "light", "hydrant",
            "sign", "dog", "deer", "skateboard", "stroller", "scooter", "other vehicle"}

    def test_decoding(self, real):
        th = FlirImageDataset(real[("thermal", "test")])
        rgb = FlirImageDataset(real[("rgb", "test")])
        i_th = real[("thermal", "test")].image_ids().tolist().index(0)
        i_rgb = real[("rgb", "test")].image_ids().tolist().index(0)
        assert th[i_th]["image"].shape == (512, 640)
        assert rgb[i_rgb]["image"].shape == (1024, 1224, 3)
