"""Tests of the aligned FLIR v1 readers (F16, ``irwm.data.aligned``).

Groups 1–5 run on a tiny synthetic tree with the as-downloaded layout (CPU, seconds).
Group 6 (``-m data``) reads the real dataset under ``$IRWM_DATA_ROOT/aligned/align``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from irwm.data import aligned as A
from irwm.data.aligned import (
    ALIGNED_CATEGORY_ID_TO_LABEL,
    ALIGNED_CLASS_NAMES,
    ALIGNED_LABEL_TO_CATEGORY_ID,
    AlignedImageDataset,
    AlignedRecord,
    AlignedSplit,
    aligned_root,
    carve_val_ids,
    load_purged_ids,
    load_val_ids,
    purge_ids,
    render_val_file,
    split_stems,
)

VAL_FILE_SHA256 = "f7103c0f7a0dc52f79fcafd21c906445165a9e105c73830c51439f9682f9f6dc"

# Synthetic ids: n = 10 → val = [40] (round(1.0)); purge band [20, 40) → 21, 34, 35, 36.
TRAIN_IDS = [2, 3, 5, 8, 13, 21, 34, 35, 36, 40]
TEST_IDS = [50, 51]
OBJECTS: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = {
    2: [("person", (10, 20, 30, 60)), ("bicycle", (0, 0, 5, 5)), ("car", (100, 100, 200, 150)),
        ("dog", (1, 1, 2, 2))],
    3: [("car", (600, 400, 641, 512))],  # xmax beyond the frame: clipped to 640
    5: [("bicycle", (1, 2, 3, 4))],  # only background: empty after dropping
}
DEFAULT_OBJECTS = [("person", (0, 0, 32, 64))]


def _voc(objects, width=640, height=512) -> str:
    objs = "".join(
        f"<object><name>{n}</name><difficult>0</difficult><bndbox><xmin>{b[0]:g}</xmin>"
        f"<ymin>{b[1]:g}</ymin><xmax>{b[2]:g}</xmax><ymax>{b[3]:g}</ymax></bndbox></object>"
        for n, b in objects
    )
    return (f"<annotation><filename>x.jpeg</filename><size><width>{width}</width>"
            f"<height>{height}</height><depth>3</depth></size>{objs}</annotation>")


def _stem(i: int) -> str:
    return f"FLIR_{i:05d}"


def _xml_path(root: Path, i: int) -> Path:
    return root / "Annotations" / f"{_stem(i)}_PreviewData.xml"


@pytest.fixture
def fake(irwm_env) -> dict:
    root = irwm_env["data"] / "aligned" / "align"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    (root / "align_train.txt").write_text(
        "".join(f"{_stem(i)}_PreviewData\n" for i in reversed(TRAIN_IDS)), encoding="utf-8")
    (root / "align_validation.txt").write_text(
        "".join(f"{_stem(i)}_PreviewData\n" for i in TEST_IDS), encoding="utf-8")
    for i in TRAIN_IDS + TEST_IDS:
        _xml_path(root, i).write_text(_voc(OBJECTS.get(i, DEFAULT_OBJECTS)), encoding="utf-8")
        Image.new("L", (640, 512), color=i).save(root / "JPEGImages" / f"{_stem(i)}_PreviewData.jpeg")
        Image.new("RGB", (640, 512), color=(i, 0, 0)).save(root / "JPEGImages" / f"{_stem(i)}_RGB.jpg")
    val_file = irwm_env["data"] / "val_ids.txt"
    val_file.write_text(render_val_file(root), encoding="utf-8")
    return {"root": root, "val_file": val_file}


def _load(fake, modality="thermal", split="train") -> AlignedSplit:
    return AlignedSplit.load(modality, split, fake["root"], val_file=fake["val_file"])


# --------------------------------------------------------------------------- Group 1: constants, paths
class TestConstants:
    def test_label_space(self):
        assert ALIGNED_CLASS_NAMES == ("person", "car")
        assert A.ALIGNED_NUM_CLASSES == 2
        assert ALIGNED_LABEL_TO_CATEGORY_ID == {0: 1, 1: 2}
        assert ALIGNED_CATEGORY_ID_TO_LABEL == {1: 0, 2: 1}
        assert A.ALIGNED_DROPPED == {"bicycle", "dog"}

    def test_root_from_env(self, irwm_env):
        assert aligned_root() == irwm_env["data"].resolve() / "aligned" / "align"


# --------------------------------------------------------------------------- Group 2: carve, splits
class TestCarve:
    def test_round_ten_percent(self):
        assert len(carve_val_ids(range(4129))) == 413
        assert carve_val_ids([5, 1, 9, 3, 7, 2, 8, 4, 6, 10]) == [10]
        assert carve_val_ids(list(range(1, 16))) == [14, 15]  # round(1.5) = 2 (banker's), highest ids

    def test_carve_degenerate(self):
        with pytest.raises(ValueError):
            carve_val_ids([1, 2, 3])  # round(0.3) = 0

    def test_purge(self):
        assert purge_ids(TRAIN_IDS, [40]) == [21, 34, 35, 36]
        assert purge_ids(TRAIN_IDS, [40], gap=5) == [35, 36]
        assert purge_ids([1, 2, 3], [100]) == []

    def test_render_and_load(self, fake):
        text = fake["val_file"].read_text(encoding="utf-8")
        assert load_val_ids(fake["val_file"]) == [40]
        assert load_purged_ids(fake["val_file"]) == [21, 34, 35, 36]
        sha = hashlib.sha256((fake["root"] / "align_train.txt").read_bytes()).hexdigest()
        assert f"sha256={sha} n=10" in text
        assert "[00020, 00040): 4 ids" in text

    def test_split_stems(self, fake):
        got = {s: split_stems(s, fake["root"], fake["val_file"]) for s in ("train", "val", "test")}
        assert got == {"train": [_stem(i) for i in (2, 3, 5, 8, 13)], "val": [_stem(40)],
                       "test": [_stem(50), _stem(51)]}

    def test_overlap_rejected(self, fake):
        with open(fake["root"] / "align_validation.txt", "a", encoding="utf-8") as f:
            f.write(f"{_stem(2)}_PreviewData\n")
        with pytest.raises(ValueError, match="share"):
            split_stems("test", fake["root"], fake["val_file"])

    def test_val_id_not_in_train(self, fake):
        fake["val_file"].write_text("# purged: 00021\n00099\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not in"):
            split_stems("val", fake["root"], fake["val_file"])

    @pytest.mark.parametrize("bad", ["FLIR_00002_RGB", "FLIR_x_PreviewData"])
    def test_malformed_entry(self, fake, bad):
        (fake["root"] / "align_validation.txt").write_text(bad + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="does not match"):
            split_stems("test", fake["root"], fake["val_file"])

    def test_unknown_split_or_modality(self, fake):
        with pytest.raises(ValueError):
            split_stems("validation", fake["root"])  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            AlignedSplit.load("ir", "train", fake["root"])  # type: ignore[arg-type]

    def test_missing_list(self, fake):
        (fake["root"] / "align_train.txt").unlink()
        with pytest.raises(FileNotFoundError):
            _load(fake)


# --------------------------------------------------------------------------- Group 3: records
class TestRecords:
    def test_ids_and_paths(self, fake):
        for m, suffix in (("thermal", "_PreviewData.jpeg"), ("rgb", "_RGB.jpg")):
            s = _load(fake, m)
            assert (s.modality, s.split, s.root) == (m, "train", fake["root"])
            assert s.image_ids().tolist() == [2, 3, 5, 8, 13] and len(s) == 5
            r = s.records[0]
            assert r.file_name == f"JPEGImages/FLIR_00002{suffix}"
            assert r.path == (fake["root"] / r.file_name).resolve() and r.path.is_absolute()
            assert (r.width, r.height) == (640, 512)

    def test_boxes_labels_dropped(self, fake):
        r = _load(fake).by_image_id(2)
        assert r.boxes.dtype == np.float32 and r.boxes.shape == (2, 4)
        np.testing.assert_array_equal(r.boxes, [[10, 20, 20, 40], [100, 100, 100, 50]])  # w = xmax − xmin
        assert r.labels.dtype == np.int64 and r.labels.tolist() == [0, 1]
        assert r.ann_ids.dtype == np.int64 and r.ann_ids.tolist() == [2000, 2001]

    def test_clipping(self, fake):
        np.testing.assert_array_equal(_load(fake).by_image_id(3).boxes, [[600, 400, 40, 112]])

    def test_background_only_frame_is_empty(self, fake):
        r = _load(fake).by_image_id(5)
        assert r.boxes.shape == (0, 4) and r.labels.shape == (0,) and r.ann_ids.shape == (0,)

    def test_modalities_share_boxes(self, fake):
        t, c = _load(fake, "thermal"), _load(fake, "rgb")
        for a, b in zip(t, c, strict=True):
            assert a.image_id == b.image_id
            np.testing.assert_array_equal(a.boxes, b.boxes)
            np.testing.assert_array_equal(a.ann_ids, b.ann_ids)

    @pytest.mark.parametrize("case", ["unknown_class", "size", "degenerate", "outside", "missing"])
    def test_malformed(self, fake, case):
        p = _xml_path(fake["root"], 8)
        if case == "unknown_class":
            p.write_text(_voc([("truck", (0, 0, 10, 10))]), encoding="utf-8")
        elif case == "size":
            p.write_text(_voc(DEFAULT_OBJECTS, width=320), encoding="utf-8")
        elif case == "degenerate":
            p.write_text(_voc([("car", (10, 10, 10, 20))]), encoding="utf-8")
        elif case == "outside":
            p.write_text(_voc([("car", (700, 10, 720, 20))]), encoding="utf-8")
        else:
            p.unlink()
        with pytest.raises(FileNotFoundError if case == "missing" else ValueError):
            _load(fake)

    def test_load_opens_no_image(self, fake, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("image opened during metadata load")

        monkeypatch.setattr(Image, "open", boom)
        assert len(_load(fake, "rgb", "test")) == 2

    def test_by_image_id_and_subset(self, fake):
        s = _load(fake)
        assert isinstance(s.by_image_id(8), AlignedRecord)
        with pytest.raises(KeyError, match="thermal/train"):
            s.by_image_id(40)
        sub = s.subset([13, 2])
        assert sub.image_ids().tolist() == [2, 13]
        assert (sub.modality, sub.split, sub.root) == (s.modality, s.split, s.root)
        with pytest.raises(KeyError):
            s.subset([99])

    def test_duck_type_surface(self, fake):
        """The attributes ``FlirDetectionDataset`` reads from a split and its records."""
        s = _load(fake)
        for attr in ("modality", "split", "root", "records", "by_image_id", "to_coco_gt", "image_dataset"):
            assert hasattr(s, attr)
        for attr in ("image_id", "width", "height", "boxes", "labels"):
            assert hasattr(s.records[0], attr)


# --------------------------------------------------------------------------- Group 4: COCO GT
class TestCocoGt:
    def test_schema(self, fake):
        gt = _load(fake).to_coco_gt()
        assert gt["categories"] == [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
        assert [im["id"] for im in gt["images"]] == [2, 3, 5, 8, 13]
        assert gt["images"][0] == {"id": 2, "file_name": "JPEGImages/FLIR_00002_PreviewData.jpeg",
                                   "width": 640, "height": 512}
        anns = gt["annotations"]
        assert len(anns) == 2 + 1 + 0 + 1 + 1
        assert all(a["iscrowd"] == 0 for a in anns)
        assert len({a["id"] for a in anns}) == len(anns)
        a0 = anns[0]
        assert a0 == {"id": 2000, "image_id": 2, "category_id": 1, "bbox": [10.0, 20.0, 20.0, 40.0],
                      "area": 800.0, "iscrowd": 0}
        assert anns[1]["category_id"] == 2
        json.dumps(gt)

    def test_area_modes_equal_at_640x512(self, fake):
        s = _load(fake)
        assert s.to_coco_gt("normalised") == s.to_coco_gt("native")
        with pytest.raises(ValueError):
            s.to_coco_gt("pixels")  # type: ignore[arg-type]

    def test_pycocotools_perfect_detections(self, fake):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt = _load(fake).to_coco_gt()
        dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
                for a in gt["annotations"]]
        coco = COCO()
        coco.dataset = copy.deepcopy(gt)
        coco.createIndex()
        E = COCOeval(coco, coco.loadRes(dets), iouType="bbox")
        E.evaluate(), E.accumulate(), E.summarize()
        assert E.stats[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- Group 5: pixels
class TestImageDataset:
    def test_decode(self, fake):
        th = AlignedImageDataset(_load(fake, "thermal"))
        it = th[0]
        assert it["image_id"] == 2 and it["image"].dtype == np.uint8 and it["image"].shape == (512, 640)
        rgb = _load(fake, "rgb").image_dataset()
        assert isinstance(rgb, AlignedImageDataset) and len(rgb) == 5
        assert rgb[1]["image"].shape == (512, 640, 3) and rgb[1]["image_id"] == 3

    def test_transform(self, fake):
        ds = _load(fake).image_dataset(transform=lambda x: x.shape)
        assert ds[0]["image"] == (512, 640)

    def test_wrong_mode_or_size(self, fake):
        img = fake["root"] / "JPEGImages" / "FLIR_00002_PreviewData.jpeg"
        Image.new("RGB", (640, 512)).save(img, format="JPEG")
        with pytest.raises(ValueError, match="mode"):
            AlignedImageDataset(_load(fake))[0]
        Image.new("L", (320, 256)).save(img, format="JPEG")
        with pytest.raises(ValueError, match="size"):
            AlignedImageDataset(_load(fake))[0]

    def test_pickle_and_dataloader(self, fake):
        ds = pickle.loads(pickle.dumps(_load(fake, "rgb", "test").image_dataset()))
        items = list(DataLoader(ds, batch_size=None, num_workers=2))
        assert [int(it["image_id"]) for it in items] == TEST_IDS


# --------------------------------------------------------------------------- Group 6 (real data)
# (person, car) kept boxes per split. With the 6 purged frames (24 person, 20 car) they add up
# to the 13,094 person and 24,732 car boxes of the whole dataset.
REAL_BOX_COUNTS = {"train": (7903, 18263), "val": (1060, 2325), "test": (4107, 4124)}


@pytest.fixture(scope="module")
def real():
    from irwm.utils.paths import load_env

    load_env()
    if not os.environ.get("IRWM_DATA_ROOT"):
        pytest.skip("IRWM_DATA_ROOT is not set")
    if not aligned_root().is_dir():
        pytest.skip(f"{aligned_root()} does not exist")
    return {(m, s): AlignedSplit.load(m, s) for m in ("rgb", "thermal") for s in ("train", "val", "test")}


class TestCommittedValFile:
    def test_sha256_pin(self):
        assert hashlib.sha256(A.VAL_IDS_FILE.read_bytes()).hexdigest() == VAL_FILE_SHA256

    def test_contents(self):
        val, purged = load_val_ids(), load_purged_ids()
        assert len(val) == 413 and (val[0], val[-1]) == (7104, 8859) and val == sorted(val)
        assert purged == [7089, 7091, 7093, 7094, 7098, 7102]


@pytest.mark.data
class TestRealData:
    def test_committed_file_reproduces(self, real):
        assert render_val_file() == A.VAL_IDS_FILE.read_text(encoding="utf-8")

    def test_split_counts(self, real):
        assert {k: len(v) for k, v in real.items()} == {
            (m, s): n for m in ("rgb", "thermal") for s, n in (("train", 3710), ("val", 413), ("test", 1013))}
        ids = {s: set(real[("thermal", s)].image_ids().tolist()) for s in ("train", "val", "test")}
        assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
        assert max(ids["train"]) < 7104 - A.PURGE_GAP
        assert (min(ids["test"]), max(ids["test"])) == (8864, 10227)

    def test_box_counts(self, real):
        got = {s: tuple(int(sum((r.labels == k).sum() for r in real[("thermal", s)])) for k in (0, 1))
               for s in ("train", "val", "test")}
        assert got == REAL_BOX_COUNTS
        for s in ("train", "val", "test"):
            for a, b in zip(real[("thermal", s)], real[("rgb", s)], strict=True):
                assert a.image_id == b.image_id and np.array_equal(a.boxes, b.boxes)

    def test_boxes_inside_frame(self, real):
        for s in real.values():
            for r in s:
                b = r.boxes
                assert (b[:, 2:] > 0).all() and (b[:, 0] + b[:, 2] <= 640).all() and (b[:, 1] + b[:, 3] <= 512).all()
        r = real[("thermal", "train")].by_image_id(3093)  # the one box with xmax = 641
        assert (r.boxes[:, 0] + r.boxes[:, 2]).max() == 640

    def test_decode_samples(self, real):
        for (m, _), s in real.items():
            ds = s.image_dataset()
            for i in (0, len(ds) - 1):
                img = ds[i]["image"]
                assert img.dtype == np.uint8 and img.shape == ((512, 640) if m == "thermal" else (512, 640, 3))

    def test_files_exist(self, real):
        for s in real.values():
            assert all(r.path.is_file() for r in s)

