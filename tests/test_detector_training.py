"""F8 detector training pipeline (spec/2026-10-26-detector-training-pipeline, validation.md Groups 1-9).

CPU groups run on a tiny synthetic FLIR tree with a stub encoder and a reduced detector, in fp32
(CUDA fp16 autocast does not exist on CPU). Group 8 (``gpu``) runs a real ``Trainer.fit`` in
``16-mixed``; Group 9 (``data`` + ``gpu``) the real encoder on real frames.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir, initialize_config_module
from PIL import Image
from torch import nn
from torch.utils.data import RandomSampler

from irwm.data.classes import CLASS_NAMES, LABEL_TO_CATEGORY_ID
from irwm.data.detection import FlirDetectionDataModule, FlirDetectionDataset, collate_detection, first_nonempty_ids
from irwm.data.flir import FLIR_DIRNAME, FlirSplit
from irwm.detection import DINODetector, cuda_ops_available
from irwm.detection.coco_val import coco_map, to_flir_detections
from irwm.features.preprocess import POLICY_SIZE, Preprocess
from irwm.utils.paths import repo_root
from irwm.worldmodel.reference import center_crop_box

REPO = repo_root()
PROJECT = REPO / "projects" / "detection"
# F7 plan.md Step 7: reduced detector for fast tests; the 32x40 grid has >= 100 pyramid tokens.
REDUCED = {"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1,
           "transformer.decoder.num_layers": 2}
DETECTOR = {"in_channels": 1152, "stem_dim": 768, "num_classes": 6, "overrides": REDUCED}
ENCODER = {"policy": "native", "precision": "fp16", "hub_version": "dinof_release", "attention": "sdpa"}
OPTIM = {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 1e-4, "backbone_lr_factor": 1.0, "lr_drop_epoch": 40}
requires_cuda_ops = pytest.mark.skipif(not cuda_ops_available(), reason="needs CUDA and detrex's CUDA ops")

# --------------------------------------------------------------------------- tiny FLIR tree
# FLIR ids: 1 person, 2 bike, 3 car, 4 motor, 10 light, 12 sign; 8 is an ignore class.
CATEGORIES = [{"id": i, "name": n} for i, n in
              [(1, "person"), (2, "bike"), (3, "car"), (4, "motor"), (8, "truck"), (10, "light"), (12, "sign")]]
SIZES = {"thermal": (80, 64), "rgb": (100, 64)}  # (W, H); native crops RGB to x = 10..90, scale 7
IGNORE = [(8, [30, 20, 10, 10])]
# image_id -> [(category_id, xywh)]; RGB boxes inside x = 10..90 unless stated.
FRAMES = {
    ("thermal", "train"): {1: IGNORE,
                           2: [(1, [8, 16, 16, 8]), (3, [40, 30, 20, 12])],
                           3: [(3, [10, 10, 20, 10])],
                           4: [(2, [50, 5, 8, 12]), (1, [20, 40, 6, 14])],
                           5: [(4, [60, 40, 10, 8]), (10, [5, 5, 4, 4]), (12, [70, 10, 5, 5])]},
    ("rgb", "train"): {1: IGNORE,
                       2: [(1, [2, 10, 5, 8])],  # only box entirely in the cropped edge x < 10
                       3: [(1, [20, 8, 10, 10]), (3, [40, 30, 20, 12])],
                       4: [(3, [20, 10, 20, 10])],
                       5: [(2, [50, 5, 8, 12]), (1, [30, 40, 6, 14])],
                       6: [(4, [60, 40, 10, 8]), (10, [15, 5, 4, 4]), (12, [75, 10, 5, 5])]},
    ("thermal", "val"): {1: IGNORE,
                         2: [(1, [10, 10, 8, 16]), (3, [30, 30, 20, 10])],
                         3: [(2, [50, 20, 8, 10])],
                         4: [(4, [5, 40, 12, 8]), (10, [60, 5, 4, 6])],
                         5: [(12, [70, 30, 6, 6]), (3, [20, 45, 15, 10]), (1, [40, 10, 6, 12])]},
    ("rgb", "val"): {1: IGNORE,
                     2: [(1, [20, 10, 8, 16]), (3, [40, 30, 20, 10])],
                     3: [(2, [60, 20, 8, 10])],
                     4: [(4, [15, 40, 12, 8]), (10, [70, 5, 4, 6])],
                     5: [(12, [80, 30, 6, 6]), (3, [30, 45, 15, 10]), (1, [50, 10, 6, 12])]},
}
_SPLIT_DIRS = {("thermal", "train"): "images_thermal_train", ("thermal", "val"): "images_thermal_val",
               ("rgb", "train"): "images_rgb_train", ("rgb", "val"): "images_rgb_val"}


def write_tiny_flir(flir_root: Path) -> None:
    """``coco.json`` + noise JPEGs for (rgb, thermal) × (train, val) in FLIR's layout."""
    rng = np.random.default_rng(0)
    for (modality, split), frames in FRAMES.items():
        sdir = flir_root / "FLIR_ADAS_v2" / _SPLIT_DIRS[(modality, split)]
        (sdir / "data").mkdir(parents=True)
        w, h = SIZES[modality]
        images, annotations = [], []
        for image_id, boxes in frames.items():
            vid = f"vid{split}{modality}"
            fname = f"data/video-{vid}-frame-{image_id:06d}-u{image_id}.jpg"
            shape = (h, w) if modality == "thermal" else (h, w, 3)
            Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(sdir / fname)
            images.append({"id": image_id, "file_name": fname, "width": w, "height": h,
                           "extra_info": {"video_id": vid}})
            for cat, bbox in boxes:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": cat,
                                    "bbox": [float(v) for v in bbox], "area": float(bbox[2] * bbox[3]),
                                    "iscrowd": 0})
        coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
        (sdir / "coco.json").write_text(json.dumps(coco))


@pytest.fixture
def tiny_flir(irwm_env) -> Path:
    root = irwm_env["data"] / FLIR_DIRNAME
    write_tiny_flir(root)
    return root


def _split(modality: str, split: str) -> FlirSplit:
    return FlirSplit.load(modality, split)  # type: ignore[arg-type]


class StubEncoder(nn.Module):
    """``encode_pca``: ``[B, 3, H, W]`` → avg-pool 14 → fixed ``Linear(3, 1152)`` → ``[B, H/14, W/14, 1152]``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3, 1152)
        self.requires_grad_(False)

    def encode_pca(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.proj(F.avg_pool2d(x, 14).permute(0, 2, 3, 1)).float()


def make_module(**kwargs):
    from irwm.detection.lightning import DetectorModule

    kw = dict(detector=DETECTOR, encoder=ENCODER, optim=OPTIM, encoder_module=StubEncoder())
    kw.update(kwargs)
    return DetectorModule(**kw)


# --------------------------------------------------------------------------- Group 1
class TestPreprocessGeometry:
    @pytest.mark.parametrize("modality", ["rgb", "thermal"])
    @pytest.mark.parametrize("size", [(80, 64), (100, 64), (640, 512), (1224, 1024)])
    def test_matches_call(self, modality, size):
        w, h = size
        img = np.zeros((h, w) if modality == "thermal" else (h, w, 3), dtype=np.uint8)
        pre = Preprocess("native", modality)
        assert pre.geometry(w, h) == pre(img)[1]

    def test_values(self):
        g = Preprocess("native", "rgb").geometry(1224, 1024)
        assert g.crop_box == center_crop_box(1224, 1024, (5, 4)) and g.out_size == (560, 448)
        assert Preprocess("stretch", "rgb").geometry(1224, 1024).crop_box == (0, 0, 1224, 1024)


# --------------------------------------------------------------------------- Group 2
class TestDataset:
    def test_items(self, tiny_flir):
        split = _split("thermal", "train")
        ds = FlirDetectionDataset(split, filter_empty=False)
        assert len(ds) == len(split.records)
        for i in range(len(ds)):
            it = ds[i]
            assert it["image"].dtype == torch.float32 and it["image"].shape == (3, 448, 560)
            assert it["boxes"].dtype == torch.float32 and it["boxes"].shape[1:] == (4,)
            assert it["labels"].dtype == torch.int64 and it["labels"].shape == it["boxes"].shape[:1]

    def test_box_mapping_thermal(self, tiny_flir):
        ds = FlirDetectionDataset(_split("thermal", "train"), image_ids=[2])
        torch.testing.assert_close(ds[0]["boxes"][0], torch.tensor([56.0, 112.0, 168.0, 168.0]), atol=1e-4, rtol=0)

    def test_rgb_crop(self, tiny_flir):
        ds = FlirDetectionDataset(_split("rgb", "train"), image_ids=[2, 3])
        by_id = {ds[i]["image_id"]: ds[i] for i in range(len(ds))}
        torch.testing.assert_close(by_id[3]["boxes"][0], torch.tensor([70.0, 56.0, 140.0, 126.0]), atol=1e-4, rtol=0)
        assert by_id[2]["boxes"].shape == (0, 4)  # box entirely in x < 10 dropped

    @pytest.mark.parametrize("modality", ["rgb", "thermal"])
    def test_ignore_classes(self, tiny_flir, modality):
        for split in ("train", "val"):
            ds = FlirDetectionDataset(_split(modality, split))
            items = {ds[i]["image_id"]: ds[i] for i in range(len(ds))}
            assert items[1]["boxes"].shape == (0, 4) and items[1]["labels"].shape == (0,)
            assert all(((it["labels"] >= 0) & (it["labels"] <= 5)).all() for it in items.values())

    def test_empty_filter(self, tiny_flir):
        rgb = FlirDetectionDataset(_split("rgb", "train"), filter_empty=True)
        assert {rgb[i]["image_id"] for i in range(len(rgb))} == {3, 4, 5, 6} and rgb.n_dropped_empty == 2
        th = FlirDetectionDataset(_split("thermal", "train"), filter_empty=True)
        assert 1 not in {th[i]["image_id"] for i in range(len(th))} and th.n_dropped_empty == 1

    def test_flip(self, tiny_flir):
        split = _split("rgb", "train")
        plain = FlirDetectionDataset(split, flip_p=0.0)
        flip = FlirDetectionDataset(split, flip_p=1.0)
        for i in range(len(plain)):
            a, b = plain[i], flip[i]
            assert a["flipped"] is False and b["flipped"] is True
            assert torch.equal(b["image"], a["image"].flip(-1))
            assert torch.equal(b["boxes"][:, 0], 560 - a["boxes"][:, 2])
            assert torch.equal(b["boxes"][:, 2], 560 - a["boxes"][:, 0])
            assert torch.equal(b["boxes"][:, [1, 3]], a["boxes"][:, [1, 3]])
            assert b["geometry"] == a["geometry"]

    def test_flip_rate(self, tiny_flir):
        ds = FlirDetectionDataset(_split("thermal", "train"), flip_p=0.5, image_ids=[2])
        torch.manual_seed(0)
        frac = sum(ds[0]["flipped"] for _ in range(400)) / 400
        assert 0.42 <= frac <= 0.58

    def test_errors(self, tiny_flir):
        split = _split("thermal", "train")
        with pytest.raises(ValueError):
            FlirDetectionDataset(split, flip_p=1.5)
        with pytest.raises(KeyError):
            FlirDetectionDataset(split, image_ids=[999999])

    def test_first_nonempty_ids(self, tiny_flir):
        assert first_nonempty_ids(_split("rgb", "train"), 2) == [3, 4]
        assert first_nonempty_ids(_split("thermal", "train"), 2) == [2, 3]
        with pytest.raises(ValueError):
            first_nonempty_ids(_split("thermal", "train"), 5)


# --------------------------------------------------------------------------- Group 3
class TestDataModule:
    def test_collate(self, tiny_flir):
        ds = FlirDetectionDataset(_split("thermal", "train"), filter_empty=True)
        items = [ds[i] for i in range(3)]
        b = collate_detection(items)
        assert b["images"].shape == (3, 3, 448, 560)
        assert b["image_ids"] == [it["image_id"] for it in items]
        assert b["geometries"] == [it["geometry"] for it in items]
        for t, it in zip(b["targets"], items):
            assert torch.equal(t["boxes"], it["boxes"]) and torch.equal(t["labels"], it["labels"])

    def test_setup_idempotent(self, tiny_flir, monkeypatch):
        orig = FlirSplit.load.__func__
        calls = []

        def spy(cls, *args, **kwargs):
            calls.append(args)
            return orig(cls, *args, **kwargs)

        monkeypatch.setattr(FlirSplit, "load", classmethod(spy))
        dm = FlirDetectionDataModule(modality="thermal", batch_size=2)
        dm.setup("fit")
        dm.setup("fit")
        assert len(calls) == 2

    def test_val_gt(self, tiny_flir):
        dm = FlirDetectionDataModule(modality="thermal", batch_size=2)
        dm.setup("fit")
        val_ids = [dm.val_set[i]["image_id"] for i in range(len(dm.val_set))]
        assert [im["id"] for im in dm.val_gt["images"]] == val_ids == [1, 2, 3, 4, 5]
        crowd = [a for a in dm.val_gt["annotations"] if a["iscrowd"] == 1]
        assert len(crowd) == 6 and {a["category_id"] for a in crowd} == set(LABEL_TO_CATEGORY_ID.values())

    def test_loaders(self, tiny_flir):
        dm = FlirDetectionDataModule(modality="thermal", batch_size=2)
        dm.setup("fit")
        tr, va = dm.train_dataloader(), dm.val_dataloader()
        assert tr.drop_last is True and isinstance(tr.sampler, RandomSampler)
        assert va.drop_last is False
        ids = [i for b in va for i in b["image_ids"]]
        assert ids == sorted(ids) == [1, 2, 3, 4, 5]

    def test_val_on_train_subset(self, tiny_flir):
        ids = [3, 5]
        dm = FlirDetectionDataModule(modality="rgb", batch_size=2, val_split="train", val_image_ids=ids,
                                     train_image_ids=ids, flip_p=0.0)
        dm.setup("fit")
        assert [dm.val_set[i]["image_id"] for i in range(len(dm.val_set))] == ids
        assert [im["id"] for im in dm.val_gt["images"]] == ids


# --------------------------------------------------------------------------- Group 4
def _perfect(gt: dict) -> list[dict]:
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
            for a in gt["annotations"] if not a["iscrowd"]]


class TestCocoMap:
    @pytest.mark.parametrize("modality", ["rgb", "thermal"])
    def test_perfect(self, tiny_flir, modality):
        gt = _split(modality, "val").to_coco_gt()
        m = coco_map(gt, _perfect(gt))
        assert m["AP"] == pytest.approx(1.0, abs=1e-6) and m["AP50"] == pytest.approx(1.0, abs=1e-6)
        for c in CLASS_NAMES:  # every class has a val box in the tiny tree
            assert m[f"AP_{c}"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("modality", ["rgb", "thermal"])
    def test_round_trip(self, tiny_flir, modality):
        split = _split(modality, "val")
        pre = Preprocess("native", modality)
        preds, ids, geoms = [], [], []
        for r in split.records:
            g = pre.geometry(r.width, r.height)
            xywh, keep = g.apply_boxes(r.boxes)
            xyxy = np.concatenate([xywh[keep, :2], xywh[keep, :2] + xywh[keep, 2:]], axis=1)
            preds.append({"boxes": torch.from_numpy(xyxy), "scores": torch.ones(int(keep.sum())),
                          "labels": torch.from_numpy(r.labels[keep])})
            ids.append(r.image_id)
            geoms.append(g)
        m = coco_map(split.to_coco_gt(), to_flir_detections(preds, ids, geoms))
        assert m["AP50"] == pytest.approx(1.0, abs=1e-6)

    def test_ignore_region(self, tiny_flir):
        gt = _split("thermal", "val").to_coco_gt()
        dets = [dict(d, score=0.1 + 0.05 * i) for i, d in enumerate(_perfect(gt)) if i % 2 == 0]
        base = coco_map(gt, dets)["AP"]
        on_ignore = {"image_id": 1, "category_id": 1, "bbox": IGNORE[0][1], "score": 0.99}
        assert coco_map(gt, dets + [on_ignore])["AP"] == pytest.approx(base, abs=1e-12)
        assert 0.0 < base < 1.0

    def test_wrong_class(self, tiny_flir):
        gt = _split("thermal", "val").to_coco_gt()
        cats = sorted(LABEL_TO_CATEGORY_ID.values())
        dets = [dict(d, category_id=cats[(cats.index(d["category_id"]) + 1) % 6]) for d in _perfect(gt)]
        assert coco_map(gt, dets)["AP"] == 0.0

    def test_to_flir_detections(self):
        g = Preprocess("native", "rgb").geometry(100, 64)  # crop x = 10..90, scale 7 on both axes
        pred = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 7.0, 7.0]]),
                "scores": torch.tensor([0.9, 0.5]), "labels": torch.tensor([4, 5])}
        dets = to_flir_detections([pred], [7], [g])
        assert [d["category_id"] for d in dets] == [10, 12] and dets[0]["image_id"] == 7
        np.testing.assert_allclose(dets[0]["bbox"], [10 / 7 + 10, 20 / 7, 20 / 7, 40 / 7], rtol=1e-6)
        assert dets[0]["score"] == pytest.approx(0.9)

    def test_empty_and_absent_class(self, tiny_flir):
        split = _split("thermal", "val")
        only = [split.by_image_id(1), split.by_image_id(3)]  # ignore-only + one bike
        gt = FlirSplit("thermal", "val", split.root, only).to_coco_gt()
        m = coco_map(gt, [])
        assert m["AP_bike"] == 0.0 and m["AP"] == 0.0 and m["APl"] == 0.0
        assert all(math.isnan(m[f"AP_{c}"]) for c in CLASS_NAMES if c != "bike")
        m = coco_map(gt, _perfect(gt))
        assert m["AP_bike"] == pytest.approx(1.0) and math.isnan(m["AP_person"]) and math.isnan(m["AP50_car"])

    def test_silent(self, tiny_flir, capsys):
        gt = _split("thermal", "val").to_coco_gt()
        coco_map(gt, _perfect(gt))
        assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- Group 5
@pytest.fixture(scope="module")
def module():
    return make_module()


class TestDetectorModule:
    def test_encode(self, module):
        f = module.encode(torch.randn(2, 3, 448, 560))
        assert f.dtype == torch.float32 and f.shape == (2, 1152, 32, 40) and f.is_contiguous()

    def test_optimizer_groups(self, module):
        opt = module.configure_optimizers()["optimizer"]
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids))
        assert set(ids) == {id(p) for p in module.detector.parameters()}
        assert not set(ids) & {id(p) for p in module.encoder.parameters()}
        group_of = {id(p): g for g in opt.param_groups for p in g["params"]}
        norms = [m for m in module.detector.modules() if isinstance(m, (nn.LayerNorm, nn.GroupNorm))]
        assert norms and all(group_of[id(p)]["weight_decay"] == 0.0 for m in norms for p in m.parameters())
        lin = next(m for m in module.detector.modules() if isinstance(m, nn.Linear))
        assert group_of[id(lin.weight)]["weight_decay"] == 1e-4
        assert all(g["lr"] == 1e-4 for g in opt.param_groups)

    def test_backbone_lr_factor(self, module):
        module.hparams.optim = dict(OPTIM, backbone_lr_factor=0.1)
        try:
            opt = module.configure_optimizers()["optimizer"]
        finally:
            module.hparams.optim = dict(OPTIM)
        group_of = {id(p): g for g in opt.param_groups for p in g["params"]}
        for name, p in module.detector.named_parameters():
            expected = 1e-5 if name.startswith("detr.backbone.") else 1e-4
            assert group_of[id(p)]["lr"] == pytest.approx(expected, rel=1e-12), name

    def test_schedule(self, module):
        cfg = module.configure_optimizers()
        opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
        assert cfg["lr_scheduler"]["interval"] == "epoch"
        for epoch in range(1, 51):
            opt.step()
            sched.step()
            lr = opt.param_groups[0]["lr"]
            assert lr == pytest.approx(1e-4 if epoch < 40 else 1e-5), epoch

    def test_constant_lr_without_drop(self, module):
        module.hparams.optim = dict(OPTIM, lr_drop_epoch=None)
        try:
            assert "lr_scheduler" not in module.configure_optimizers()
        finally:
            module.hparams.optim = dict(OPTIM)

    def test_checkpoint_strip(self, module, tmp_path):
        module.metrics_dir = tmp_path / "val_metrics"
        ckpt = {"state_dict": module.state_dict()}
        assert any(k.startswith("encoder.") for k in ckpt["state_dict"])
        module.on_save_checkpoint(ckpt)
        assert not any(k.startswith("encoder.") for k in ckpt["state_dict"])
        assert ckpt["irwm"]["metrics_dir"] == str(tmp_path / "val_metrics")
        fresh = make_module()
        fresh.on_load_checkpoint(ckpt)
        fresh.load_state_dict(ckpt["state_dict"], strict=True)
        module.metrics_dir = None

    def test_nonfinite_guard(self, module):
        nan, ok = torch.tensor(float("nan")), torch.tensor(1.0)
        for _ in range(99):
            module._count_nonfinite(nan)
        module._count_nonfinite(ok)
        for _ in range(99):
            module._count_nonfinite(nan)
        with pytest.raises(RuntimeError):
            module._count_nonfinite(nan)
        module._count_nonfinite(ok)

    def test_missing_assets(self, irwm_env):
        with pytest.raises(FileNotFoundError, match="download.py"):
            make_module(encoder_module=None)


# --------------------------------------------------------------------------- Group 6
@pytest.fixture
def fake_ckpt(irwm_env, module):
    def make(epoch: int = 9, perturb: float = 0.0) -> Path:
        run = irwm_env["ckpt"] / "detection" / "slurm-1"
        metrics_dir = irwm_env["output"] / "detection" / "run" / "val_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"epoch_{epoch:03d}.json").write_text(json.dumps({"metrics": {"AP": 0.25}, "epoch": epoch}))
        state = {f"detector.{k}": v.clone() + perturb for k, v in module.detector.state_dict().items()}
        state["_nonfinite_steps"] = torch.zeros((), dtype=torch.long)
        ckpt = {"state_dict": state, "epoch": epoch, "global_step": 1234,
                "hyper_parameters": {"detector": DETECTOR, "encoder": ENCODER, "optim": OPTIM, "perf_every": 50},
                "irwm": {"metrics_dir": str(metrics_dir)}}
        run.mkdir(parents=True, exist_ok=True)
        path = run / f"best-e{epoch:03d}.ckpt"
        torch.save(ckpt, path)
        return path
    return make


class TestFrozen:
    def test_export_and_load(self, irwm_env, fake_ckpt, module):
        from irwm.detection import frozen_meta, load_frozen_detector
        from irwm.detection.frozen import export_frozen, frozen_dir

        out = export_frozen("det_ir", fake_ckpt())
        assert out == frozen_dir("det_ir") == irwm_env["ckpt"] / "detection" / "frozen" / "det_ir"
        head = torch.load(out / "head.pt")
        assert head and all(k.startswith("detr.") for k in head)
        meta = frozen_meta("det_ir")
        assert meta["name"] == "det_ir" and meta["modality"] == "thermal" and meta["precision"] == "16-mixed"
        assert meta["epoch"] == 9 and meta["global_step"] == 1234
        assert meta["val_metrics"] == {"metrics": {"AP": 0.25}, "epoch": 9}
        assert meta["source_ckpt"] == "detection/slurm-1/best-e009.ckpt"
        assert meta["encoder"]["attention"] == "sdpa"
        assert meta["encoder"]["encoder_tag"] == "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152"
        assert meta["image_size"] == [448, 560] and meta["detector"] == DETECTOR
        assert len(meta["head_sha256"]) == 64 and int(meta["head_sha256"], 16) >= 0
        assert "commit" in meta["git"] and meta["created_at"]

        det = load_frozen_detector("det_ir")
        assert isinstance(det, DINODetector) and det.training is False
        assert all(not p.requires_grad for p in det.parameters())
        src = module.detector.state_dict()
        for k, v in det.state_dict().items():
            assert torch.equal(v, src[k]), k

    def test_reexport(self, irwm_env, fake_ckpt):
        from irwm.detection.frozen import export_frozen, frozen_meta

        out = export_frozen("det_ir", fake_ckpt())
        sha, mtime = frozen_meta("det_ir")["head_sha256"], (out / "meta.json").stat().st_mtime_ns
        assert export_frozen("det_ir", fake_ckpt()) == out
        assert (out / "meta.json").stat().st_mtime_ns == mtime  # no-op
        other = fake_ckpt(perturb=1e-3)
        with pytest.raises(FileExistsError):
            export_frozen("det_ir", other)
        assert frozen_meta("det_ir")["head_sha256"] == sha
        export_frozen("det_ir", other, overwrite=True)
        assert frozen_meta("det_ir")["head_sha256"] != sha

    def test_sha_mismatch(self, irwm_env, fake_ckpt):
        from irwm.detection import load_frozen_detector
        from irwm.detection.frozen import export_frozen

        out = export_frozen("det_rgb", fake_ckpt())
        raw = bytearray((out / "head.pt").read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        (out / "head.pt").write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="SHA-256"):
            load_frozen_detector("det_rgb")

    def test_names_and_paths(self, irwm_env, fake_ckpt):
        from irwm.detection.frozen import export_frozen, frozen_dir

        with pytest.raises(ValueError):
            frozen_dir("det_x")
        export_frozen("det_rgb", fake_ckpt())
        written = [p for p in irwm_env["ckpt"].parent.rglob("*") if p.is_file()]
        outside = [p for p in written
                   if not p.is_relative_to(irwm_env["ckpt"]) and not p.is_relative_to(irwm_env["output"])]
        assert not outside
        assert {p.name for p in frozen_dir("det_rgb").iterdir()} == {"head.pt", "meta.json"}


# --------------------------------------------------------------------------- Group 7
def _compose(name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(PROJECT / "conf"), version_base="1.3"):
        return compose(name, overrides=overrides)


def _load_entry(name: str):
    spec = importlib.util.spec_from_file_location(f"detection_{name}", PROJECT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestConfigs:
    def test_train_det_rgb(self, irwm_env):
        cfg = _compose("train", ["experiment=det_rgb"])
        assert cfg.modality == "rgb" and cfg.name == "det_rgb" and list(cfg.wandb.tags) == ["det_rgb"]
        assert cfg.trainer.precision == "16-mixed" and cfg.data.batch_size == 16
        assert cfg.trainer.max_epochs == 50 and cfg.trainer.check_val_every_n_epoch == 5
        assert cfg.optim.lr_drop_epoch == 40 and cfg.optim.backbone_lr_factor == 1.0
        assert cfg.encoder.attention == "sdpa" and cfg.encoder.policy == "native"
        assert cfg.overfit.enabled is False and list(cfg.overfit.classes) == ["person", "car"]
        assert _compose("train", ["experiment=det_ir"]).modality == "thermal"

    def test_precision_not_from_cluster(self, irwm_env):
        assert _compose("train", ["experiment=det_ir"]).trainer.precision == "16-mixed"  # cluster=local: 32-true

    def test_experiment_mandatory(self, irwm_env):
        from hydra.errors import ConfigCompositionException
        from omegaconf.errors import MissingMandatoryValue

        with pytest.raises((ConfigCompositionException, MissingMandatoryValue)):
            _compose("train", [])

    def test_base_attention_plain(self, irwm_env):
        with initialize_config_module(config_module="irwm.conf", version_base="1.3"):
            assert compose("irwm_base").encoder.attention == "plain"

    def test_export_config(self, irwm_env):
        cfg = _compose("export", ["experiment=det_ir", "source_run=slurm-7"])
        assert cfg.name == "det_ir" and cfg.source_run == "slurm-7" and cfg.overwrite is False

    def test_pick_best_ckpt(self, tmp_path):
        pick = _load_entry("export").pick_best_ckpt
        with pytest.raises(FileNotFoundError):
            pick(tmp_path)
        (tmp_path / "last.ckpt").touch()
        (tmp_path / "best-e004.ckpt").touch()
        assert pick(tmp_path) == tmp_path / "best-e004.ckpt"
        (tmp_path / "best-e009.ckpt").touch()
        with pytest.raises(ValueError, match="best-e004.ckpt.*best-e009.ckpt"):
            pick(tmp_path)

    @pytest.mark.skipif(cuda_ops_available(), reason="checks the no-CUDA-ops exit")
    def test_train_exits_without_cuda_ops(self, irwm_env):
        env = {**os.environ, "WANDB_MODE": "disabled"}
        res = subprocess.run([sys.executable, str(PROJECT / "train.py"), "experiment=det_ir"], cwd=REPO, env=env,
                             capture_output=True, text=True, timeout=300)
        assert res.returncode != 0 and "cuda_ops_available" in res.stderr


# --------------------------------------------------------------------------- Group 8
def _fit(tmp_path: Path, max_steps: int, ckpt_path: str | None = None, callbacks: list | None = None):
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    dm = FlirDetectionDataModule(modality="thermal", batch_size=2, flip_p=0.5)
    dm.setup("fit")
    model = make_module(val_gt=dm.val_gt)
    model.metrics_dir = tmp_path / "val_metrics"
    ckpt_dir = tmp_path / "ckpt"
    best = ModelCheckpoint(dirpath=ckpt_dir, monitor="val/AP", mode="max", save_top_k=-1, save_last=True,
                           filename="best-e{epoch:03d}", auto_insert_metric_name=False)
    trainer = L.Trainer(accelerator="gpu", devices=1, max_steps=max_steps, precision="16-mixed",
                        limit_val_batches=1, check_val_every_n_epoch=1, log_every_n_steps=1,
                        gradient_clip_val=0.1, logger=CSVLogger(tmp_path, name="csv"),
                        callbacks=[best, *(callbacks or [])], default_root_dir=ckpt_dir,
                        enable_progress_bar=False)
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    return trainer, model, ckpt_dir


def _csv_rows(tmp_path: Path) -> list[dict]:
    import csv

    files = sorted((tmp_path / "csv").rglob("metrics.csv"))
    return [row for f in files for row in csv.DictReader(f.open())]


@pytest.mark.gpu
@requires_cuda_ops
class TestGpuTraining:
    def test_fit_resume_and_metrics(self, tiny_flir, tmp_path):
        import lightning as L

        trainer, _, ckpt_dir = _fit(tmp_path, max_steps=4)
        rows = _csv_rows(tmp_path)
        losses = [float(r["train/loss_step"]) for r in rows if r.get("train/loss_step")]
        assert losses and all(math.isfinite(v) for v in losses)
        assert "val/AP" in trainer.callback_metrics

        # perf/* and grad_scale once per epoch (amendment 2026-10-27); batch_idx 0 is timed (perf_every = 50)
        perf = [r for r in rows if r.get("perf/encode_share")]
        assert perf and [int(r["epoch"]) for r in perf][0] == 0
        assert float(perf[0]["perf/encode_ms"]) > 0 and float(perf[0]["perf/step_ms"]) > 0
        assert 0 < float(perf[0]["perf/encode_share"]) < 1
        assert any(r.get("train/grad_scale") and float(r["train/grad_scale"]) > 0 for r in rows)

        # FR6 lookup: each best checkpoint's epoch is the NNN of the JSON written in its validation
        bests = sorted(ckpt_dir.glob("best-e*.ckpt"))
        assert bests
        for p in bests:
            ck = torch.load(p, map_location="cpu")
            js = json.loads((tmp_path / "val_metrics" / f"epoch_{ck['epoch']:03d}.json").read_text())
            assert js["epoch"] == ck["epoch"] == int(p.stem.removeprefix("best-e"))
            assert "commit" in js["git"] and "AP" in js["metrics"]

        last = torch.load(ckpt_dir / "last.ckpt", map_location="cpu")
        assert last["global_step"] == 4
        assert not any(k.startswith("encoder.") for k in last["state_dict"])

        class StartStep(L.Callback):
            step = None

            def on_train_start(self, trainer, pl_module):
                StartStep.step = trainer.global_step

        trainer2, _, _ = _fit(tmp_path, max_steps=6, ckpt_path=str(ckpt_dir / "last.ckpt"), callbacks=[StartStep()])
        assert StartStep.step == 4 and trainer2.global_step == 6


# --------------------------------------------------------------------------- Group 9
@pytest.mark.data
@pytest.mark.gpu
@requires_cuda_ops
def test_real_encoder_thermal_val():
    from irwm.detection.lightning import DetectorModule
    from irwm.utils.paths import load_env

    load_env()
    split = FlirSplit.load("thermal", "val")
    ds = FlirDetectionDataset(split, image_ids=first_nonempty_ids(split, 2))
    batch = collate_detection([ds[0], ds[1]])
    module = DetectorModule(detector={}, encoder=ENCODER, optim=OPTIM).cuda()
    images = batch["images"].cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in batch["targets"]]
    with torch.autocast("cuda", dtype=torch.float16):
        feats = module.encode(images)
        assert feats.shape == (2, 1152, 32, 40) and torch.isfinite(feats).all()
        module.detector.train()
        losses = module.detector(feats, targets, image_size=POLICY_SIZE["native"])
        loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
