"""F18 aligned FLIR v1 training pipeline: coco_val label spaces, aligned data module, DetectorModule
(frozen-encoder and end-to-end ViTDet modes), frozen ViTDet export, configs and the SLURM launcher.

CPU tests use a tiny synthetic aligned tree (640×512 frames), a stub frozen encoder, a reduced DINO
head and a 2-block ViTDet (``init: none``); validation runs a real CPU ``Trainer.validate``. ``gpu``
tests run a real ``Trainer.fit`` in ``16-mixed`` (MSDA fallback allowed); ``data`` tests use real assets.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import nn

from irwm.data.aligned import (
    ALIGNED_CLASS_NAMES,
    ALIGNED_LABEL_TO_CATEGORY_ID,
    AlignedSplit,
    render_val_file,
)
from irwm.data.detection import AlignedDetectionDataModule, FlirDetectionDataset, collate_detection
from irwm.detection.coco_val import coco_map, to_flir_detections
from irwm.features.preprocess import POLICY_SIZE, Preprocess
from irwm.utils.paths import repo_root

REPO = repo_root()
PROJECT = REPO / "projects" / "detection"
HEADS = [f"aligned_{b}_{m}" for b in ("dinopca", "dino3072", "mae") for m in ("rgb", "ir")]
REDUCED = {"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1,
           "transformer.decoder.num_layers": 2}
DINOPCA = {"in_channels": 1152, "in_proj": None, "stem_dim": 768, "num_classes": 2, "overrides": REDUCED}
DINO3072 = {"in_channels": 3072, "in_proj": 1152, "stem_dim": 768, "num_classes": 2, "overrides": REDUCED}
VITDET = {"arch": "vitdet_mae", "num_classes": 2, "image_hw": [512, 640], "preprocess_policy": "native640",
          "use_act_checkpoint": True, "init": "none", "init_path": None,
          "optim": {"lr_schedule": "layer_decay", "layer_decay": 0.7, "backbone_lr_factor": 0.1},
          "overrides": {**REDUCED, "backbone.net.depth": 2, "backbone.net.window_block_indexes": [0]}}
ENC_DINO = {"policy": "native", "kind": "dinopca", "precision": "fp16", "hub_version": "dinof_release",
            "attention": "sdpa"}
ENC_MAE = {"policy": "native640", "kind": "mae", "encoder_tag": "mae_vitb16_512x640_vitdet-ft"}
OPTIM = {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 1e-4, "backbone_lr_factor": 1.0, "lr_drop_epoch": 14}
ALIGNED_KW = {"class_names": list(ALIGNED_CLASS_NAMES), "label_to_category_id": dict(ALIGNED_LABEL_TO_CATEGORY_ID)}

# --------------------------------------------------------------------------- tiny aligned tree
# 10 train ids → val = [40], purge band [20, 40) = 21, 34, 35, 36 → train = 2, 3, 5, 8, 13.
TRAIN_IDS = [2, 3, 5, 8, 13, 21, 34, 35, 36, 40]
TEST_IDS = [50, 51]
OBJECTS = {
    2: [("person", (10, 20, 40, 80)), ("bicycle", (0, 0, 5, 5)), ("car", (100, 100, 300, 250))],
    3: [("car", (400, 300, 600, 450))],
    5: [("bicycle", (1, 2, 30, 40))],  # background only: empty → dropped from train
    40: [("person", (50, 60, 90, 160)), ("car", (300, 200, 500, 330)), ("dog", (5, 5, 20, 20))],
}
DEFAULT = [("person", (200, 100, 240, 200))]


def _voc(objects) -> str:
    objs = "".join(f"<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
                   f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>" for n, b in objects)
    return f"<annotation><size><width>640</width><height>512</height></size>{objs}</annotation>"


@pytest.fixture
def tiny_aligned(irwm_env) -> dict:
    root = irwm_env["data"] / "aligned" / "align"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    stem = lambda i: f"FLIR_{i:05d}"  # noqa: E731
    (root / "align_train.txt").write_text("".join(f"{stem(i)}_PreviewData\n" for i in TRAIN_IDS))
    (root / "align_validation.txt").write_text("".join(f"{stem(i)}_PreviewData\n" for i in TEST_IDS))
    rng = np.random.default_rng(0)
    for i in TRAIN_IDS + TEST_IDS:
        (root / "Annotations" / f"{stem(i)}_PreviewData.xml").write_text(_voc(OBJECTS.get(i, DEFAULT)))
        Image.fromarray(rng.integers(0, 256, (512, 640), dtype=np.uint8)).save(
            root / "JPEGImages" / f"{stem(i)}_PreviewData.jpeg")
        Image.fromarray(rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)).save(
            root / "JPEGImages" / f"{stem(i)}_RGB.jpg")
    val_file = irwm_env["data"] / "val_ids.txt"
    val_file.write_text(render_val_file(root))
    return {"root": root, "val_file": val_file}


def _split(t, modality="thermal", split="train") -> AlignedSplit:
    return AlignedSplit.load(modality, split, t["root"], val_file=t["val_file"])


def _dm(t, **kw) -> AlignedDetectionDataModule:
    args = dict(modality="thermal", policy="native", batch_size=2, flip_p=0.0, root=t["root"],
                val_file=t["val_file"])
    args.update(kw)
    dm = AlignedDetectionDataModule(**args)
    dm.setup("fit")
    return dm


class StubEncoder(nn.Module):
    """Frozen-encoder interface: ``encode`` → avg-pool 14 → fixed ``Linear(3, C)`` → ``[B, C, H/14, W/14]``."""

    def __init__(self, channels: int = 1152) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3, channels)
        self.requires_grad_(False)
        self.out_channels = channels

    def encode_pca(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.proj(F.avg_pool2d(x, 14).permute(0, 2, 3, 1)).float()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_pca(x).permute(0, 3, 1, 2).contiguous()


def make_dino(detector=DINOPCA, channels=1152, **kw):
    from irwm.detection.lightning import DetectorModule

    args = dict(detector=detector, encoder=ENC_DINO, optim=OPTIM, encoder_module=StubEncoder(channels), **ALIGNED_KW)
    args.update(kw)
    m = DetectorModule(**args)
    m.detector.detr.select_box_nums_for_evaluation = 100  # reduced head: 100 queries x 2 classes < 300
    return m


def make_vitdet(**kw):
    from irwm.detection.lightning import DetectorModule

    args = dict(detector=VITDET, encoder=ENC_MAE, optim=OPTIM, **ALIGNED_KW)
    args.update(kw)
    m = DetectorModule(**args)
    m.detector.detr.select_box_nums_for_evaluation = 100  # reduced head: 100 queries x 2 classes
    return m


# --------------------------------------------------------------------------- coco_val
class TestCocoValLabelSpaces:
    def test_aligned_perfect(self, tiny_aligned):
        split = _split(tiny_aligned, split="val")
        gt = split.to_coco_gt()
        dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
                for a in gt["annotations"]]
        m = coco_map(gt, dets, class_names=ALIGNED_CLASS_NAMES)
        assert m["AP"] == pytest.approx(1.0) and m["AP50"] == pytest.approx(1.0)
        assert {k for k in m if k.startswith("AP_") or k.startswith("AP50_")} == {
            "AP_person", "AP_car", "AP50_person", "AP50_car"}
        assert m["AP_person"] == pytest.approx(1.0) and m["AP50_car"] == pytest.approx(1.0)

    def test_aligned_swapped_labels_score_zero(self, tiny_aligned):
        gt = _split(tiny_aligned, split="val").to_coco_gt()
        dets = [{"image_id": a["image_id"], "category_id": 3 - a["category_id"], "bbox": a["bbox"], "score": 1.0}
                for a in gt["annotations"]]
        m = coco_map(gt, dets, class_names=ALIGNED_CLASS_NAMES)
        assert m["AP_person"] == 0.0 and m["AP_car"] == 0.0

    def test_empty_and_absent(self, tiny_aligned):
        gt = _split(tiny_aligned, split="val").to_coco_gt()
        gt["annotations"] = [a for a in gt["annotations"] if a["category_id"] == 1]
        m = coco_map(gt, [], class_names=ALIGNED_CLASS_NAMES)
        assert m["AP"] == 0.0 and m["AP_person"] == 0.0 and math.isnan(m["AP_car"])

    def test_unknown_class(self, tiny_aligned):
        gt = _split(tiny_aligned, split="val").to_coco_gt()
        with pytest.raises(KeyError, match="bike"):
            coco_map(gt, [], class_names=["person", "bike"])

    def test_to_flir_detections_map(self):
        geom = Preprocess("native640", "thermal").geometry(640, 512)
        pred = {"boxes": torch.tensor([[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 6.0, 9.0]]),
                "scores": torch.tensor([0.9, 0.1]), "labels": torch.tensor([0, 1])}
        out = to_flir_detections([pred], [7], [geom], label_to_category_id=ALIGNED_LABEL_TO_CATEGORY_ID)
        assert [d["category_id"] for d in out] == [1, 2]
        assert out[0]["bbox"] == pytest.approx([0, 0, 10, 20])  # native640 on 640x512: identity
        # default: FLIR v2 map (label 1 = bike = category 2, label 0 = person = 1)
        assert [d["category_id"] for d in to_flir_detections([pred], [7], [geom])] == [1, 2]


# --------------------------------------------------------------------------- data
class TestAlignedData:
    @pytest.mark.parametrize("policy", ["native", "native640"])
    @pytest.mark.parametrize("modality", ["rgb", "thermal"])
    def test_dataset_items(self, tiny_aligned, policy, modality):
        ds = FlirDetectionDataset(_split(tiny_aligned, modality), policy=policy)
        assert ds._images.__class__.__name__ == "AlignedImageDataset"
        item = ds[0]
        h, w = POLICY_SIZE[policy]
        assert item["image"].shape == (3, h, w) and item["image_id"] == 2
        assert item["labels"].tolist() == [0, 1]  # bicycle dropped
        sx, sy = w / 640, h / 512
        assert item["boxes"][0].tolist() == pytest.approx([10 * sx, 20 * sy, 40 * sx, 80 * sy], rel=1e-5)

    def test_datamodule(self, tiny_aligned):
        dm = _dm(tiny_aligned)
        assert dm.train_set._image_ids() == [2, 3, 8, 13] and dm.train_set.n_dropped_empty == 1
        assert dm.val_set._image_ids() == [40]
        assert [c["name"] for c in dm.val_gt["categories"]] == ["person", "car"]
        assert {a["category_id"] for a in dm.val_gt["annotations"]} == {1, 2}
        assert not any(a["iscrowd"] for a in dm.val_gt["annotations"])
        batch = next(iter(dm.train_dataloader()))
        assert batch["images"].shape == (2, 3, 448, 560)

    def test_smoke_subset_val_on_train(self, tiny_aligned):
        dm = _dm(tiny_aligned, train_image_ids=[3, 8], val_split="train", val_image_ids=[3, 8], policy="native640")
        assert dm.train_set._image_ids() == [3, 8] and dm.val_set._image_ids() == [3, 8]
        assert sorted(i["id"] for i in dm.val_gt["images"]) == [3, 8]
        assert next(iter(dm.val_dataloader()))["images"].shape == (2, 3, 512, 640)


# --------------------------------------------------------------------------- DetectorModule
def _validate(model, dm, tmp_path):
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger

    from irwm.detection.detrex_adapter import select_msda

    select_msda("auto")
    model.metrics_dir = tmp_path / "val_metrics"
    trainer = L.Trainer(accelerator="cpu", devices=1, logger=CSVLogger(tmp_path, name="csv"),
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.validate(model, datamodule=dm)
    return trainer


class TestDinoModule:
    def test_dino3072_encode_and_groups(self):
        m = make_dino(DINO3072, channels=3072)
        assert m.arch == "dino" and not m.end_to_end and m.image_size == (448, 560)
        f = m.encode(torch.randn(1, 3, 448, 560))
        assert f.shape == (1, 3072, 32, 40)
        proj = m.detector.detr.backbone.net.proj  # learned Linear(3072 -> 1152), trained with the head
        opt = m.configure_optimizers()["optimizer"]
        ids = {id(p) for g in opt.param_groups for p in g["params"]}
        assert ids == {id(p) for p in m.detector.parameters()} and id(proj.weight) in ids
        assert not ids & {id(p) for p in m.encoder.parameters()}

    def test_encode_falls_back_to_encode_pca(self):
        class F8Stub(nn.Module):  # the F8 stub interface: encode_pca only
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def encode_pca(self, x):
                return self.inner.encode_pca(x)

        inner = StubEncoder()
        x = torch.randn(2, 3, 448, 560)
        expected = inner.encode_pca(x).permute(0, 3, 1, 2)
        assert torch.equal(make_dino(encoder_module=inner).encode(x), expected)
        assert torch.equal(make_dino(encoder_module=F8Stub(inner)).encode(x), expected)

    def test_schedule_20_epochs_drop_14(self):
        m = make_dino()
        cfg = m.configure_optimizers()
        opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
        lrs = []
        for _ in range(20):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        assert lrs[:14] == pytest.approx([1e-4] * 14) and lrs[14:] == pytest.approx([1e-5] * 6)

    def test_checkpoint_strips_encoder(self):
        m = make_dino()
        ckpt = {"state_dict": m.state_dict()}
        m.on_save_checkpoint(ckpt)
        assert not any(k.startswith("encoder.") for k in ckpt["state_dict"])
        assert ckpt["irwm"]["arch"] == "dino"

    def test_validate_aligned_metrics(self, tiny_aligned, tmp_path):
        dm = _dm(tiny_aligned, val_split="train", val_image_ids=[2, 3])
        m = make_dino(val_gt=dm.val_gt)
        trainer = _validate(m, dm, tmp_path)
        keys = set(trainer.callback_metrics)
        assert {"val/AP", "val/AP50", "val/AP_person", "val/AP_car", "val/AP50_car"} <= keys
        assert not any(k.startswith("val/AP_bike") for k in keys)
        rec = json.loads((tmp_path / "val_metrics" / "epoch_000.json").read_text())
        assert rec["gpu"]["name"] == "cpu" and rec["gpu"]["world_size"] == 1  # CPU validate
        assert set(rec["metrics"]) >= {"AP_person", "AP_car"}


@pytest.fixture(scope="module")
def vitdet_module():
    return make_vitdet()


class TestVitdetModule:
    def test_build(self, vitdet_module):
        m = vitdet_module
        assert m.arch == "vitdet_mae" and m.end_to_end and m.encoder is None
        assert m.image_size == (512, 640) == tuple(POLICY_SIZE["native640"])
        x = torch.randn(1, 3, 512, 640)
        assert m.encode(x) is x
        assert all(p.requires_grad for p in m.detector.parameters())

    def test_param_groups_layer_decay(self, vitdet_module):
        m = vitdet_module
        cfg = m.configure_optimizers()
        opt = cfg["optimizer"]
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in m.detector.parameters()}
        lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
        vit = m.detector.vit
        assert lr_of[id(vit.patch_embed.proj.weight)] == pytest.approx(1e-4 * 0.7 ** 3)  # depth 2
        assert lr_of[id(vit.blocks[1].mlp.fc1.weight)] == pytest.approx(1e-4 * 0.7)
        head = next(p for n, p in m.detector.named_parameters() if "class_embed" in n)
        assert lr_of[id(head)] == pytest.approx(1e-4)
        # MultiStepLR multiplies every group's own lr by 0.1 from epoch 14
        sched = cfg["lr_scheduler"]["scheduler"]
        before = [g["lr"] for g in opt.param_groups]
        for _ in range(14):
            opt.step()
            sched.step()
        assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.1 * b for b in before])

    def test_checkpoint_keeps_all_weights(self, vitdet_module):
        m = vitdet_module
        ckpt = {"state_dict": m.state_dict()}
        m.on_save_checkpoint(ckpt)
        assert set(ckpt["state_dict"]) == set(m.state_dict())
        assert any(k.startswith("detector.detr.backbone.net.blocks.") for k in ckpt["state_dict"])
        m.on_load_checkpoint(ckpt)  # no encoder to restore
        assert ckpt["irwm"]["arch"] == "vitdet_mae"

    def test_geometry_from_one_source(self):
        from irwm.detection.lightning import DetectorModule

        with pytest.raises(ValueError, match="image_hw"):
            DetectorModule(detector=VITDET, encoder={**ENC_MAE, "policy": "native"}, optim=OPTIM)
        with pytest.raises(ValueError, match="preprocess_policy"):
            DetectorModule(detector={**VITDET, "image_hw": None, "preprocess_policy": "native"},
                           encoder=ENC_MAE, optim=OPTIM)
        with pytest.raises(ValueError, match="arch"):
            DetectorModule(detector={**VITDET, "arch": "vitdet_x"}, encoder=ENC_MAE, optim=OPTIM)

    def test_validate_aligned_metrics(self, vitdet_module, tiny_aligned, tmp_path):
        dm = _dm(tiny_aligned, policy="native640", val_split="train", val_image_ids=[2, 3])
        vitdet_module.val_gt = dm.val_gt
        trainer = _validate(vitdet_module, dm, tmp_path)
        assert {"val/AP", "val/AP_person", "val/AP_car"} <= set(trainer.callback_metrics)


# --------------------------------------------------------------------------- frozen exports
def _fake_ckpt(irwm_env, module, detector, encoder, *, epoch=7, perturb=0.0, run="slurm-9") -> Path:
    metrics_dir = irwm_env["output"] / "detection" / run / "val_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    gpu = {"name": "NVIDIA L40S", "count": 1, "world_size": 1}
    (metrics_dir / f"epoch_{epoch:03d}.json").write_text(json.dumps({"metrics": {"AP": 0.3}, "epoch": epoch,
                                                                     "gpu": gpu}))
    state = {f"detector.{k}": v.clone() + perturb for k, v in module.detector.state_dict().items()}
    ckpt = {"state_dict": state, "epoch": epoch, "global_step": 99,
            "hyper_parameters": {"detector": detector, "encoder": encoder, "optim": OPTIM, **ALIGNED_KW},
            "irwm": {"metrics_dir": str(metrics_dir), "arch": detector.get("arch", "dino"), "gpu": gpu}}
    path = irwm_env["ckpt"] / "detection" / run / f"best-e{epoch:03d}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


def _load_entry(name: str):
    spec = importlib.util.spec_from_file_location(f"detection_{name}", PROJECT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFrozenVitdet:
    def test_export_and_load(self, irwm_env, vitdet_module):
        from irwm.detection.frozen import frozen_meta, load_frozen_detector
        from irwm.detection.frozen_vitdet import load_frozen_vitdet

        export_head = _load_entry("export").export_head
        out = export_head("aligned_mae_ir", _fake_ckpt(irwm_env, vitdet_module, VITDET, ENC_MAE))
        assert out == irwm_env["ckpt"] / "detection" / "frozen" / "aligned_mae_ir"
        assert {p.name for p in out.iterdir()} == {"head.pt", "meta.json"}
        meta = frozen_meta("aligned_mae_ir")
        assert meta["detector"]["arch"] == "vitdet_mae" and meta["dataset"] == "aligned_v1"
        assert meta["modality"] == "thermal" and meta["preprocess_policy"] == "native640"
        assert meta["image_size"] == [512, 640] and meta["encoder"]["encoder_tag"] == ENC_MAE["encoder_tag"]
        assert meta["gpu"]["name"] == "NVIDIA L40S" and meta["val_metrics"]["metrics"]["AP"] == 0.3
        assert meta["class_names"] == ["person", "car"] and meta["source_ckpt"].startswith("detection/slurm-9/")
        head = torch.load(out / "head.pt")
        assert any(k.startswith("detr.backbone.net.blocks.") for k in head)  # the ViT is kept

        model = load_frozen_vitdet("aligned_mae_ir")
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        src = vitdet_module.detector.state_dict()
        assert all(torch.equal(v, src[k]) for k, v in model.state_dict().items())
        with pytest.raises(ValueError, match="vitdet_mae"):
            load_frozen_detector("aligned_mae_ir")

    def test_write_once_and_sha(self, irwm_env, vitdet_module):
        from irwm.detection.frozen_vitdet import export_frozen_vitdet, load_frozen_vitdet

        out = export_frozen_vitdet("aligned_mae_rgb", _fake_ckpt(irwm_env, vitdet_module, VITDET, ENC_MAE))
        mtime = (out / "meta.json").stat().st_mtime_ns
        export_frozen_vitdet("aligned_mae_rgb", _fake_ckpt(irwm_env, vitdet_module, VITDET, ENC_MAE))
        assert (out / "meta.json").stat().st_mtime_ns == mtime
        other = _fake_ckpt(irwm_env, vitdet_module, VITDET, ENC_MAE, perturb=1e-3, run="slurm-10")
        with pytest.raises(FileExistsError):
            export_frozen_vitdet("aligned_mae_rgb", other)
        raw = bytearray((out / "head.pt").read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        (out / "head.pt").write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="SHA-256"):
            load_frozen_vitdet("aligned_mae_rgb")

    def test_rejects(self, irwm_env, vitdet_module):
        from irwm.detection.frozen_vitdet import export_frozen_vitdet

        ckpt = _fake_ckpt(irwm_env, vitdet_module, VITDET, ENC_MAE)
        with pytest.raises(ValueError):
            export_frozen_vitdet("aligned_dinopca_rgb", ckpt)
        dino_ckpt = _fake_ckpt(irwm_env, make_dino(), DINOPCA, ENC_DINO, run="slurm-11")
        with pytest.raises(ValueError, match="arch"):
            export_frozen_vitdet("aligned_mae_rgb", dino_ckpt)

    def test_dino_aligned_export(self, irwm_env):
        from irwm.detection.frozen import frozen_meta, load_frozen_detector

        m = make_dino(DINO3072, channels=3072)
        enc = {**ENC_DINO, "kind": "dino3072"}
        _load_entry("export").export_head("aligned_dino3072_rgb", _fake_ckpt(irwm_env, m, DINO3072, enc))
        meta = frozen_meta("aligned_dino3072_rgb")
        assert meta["dataset"] == "aligned_v1" and meta["image_size"] == [448, 560]
        assert meta["encoder"]["encoder_tag"].endswith("full3072") and meta["detector"]["in_proj"] == 1152
        det = load_frozen_detector("aligned_dino3072_rgb")
        assert all(torch.equal(v, m.detector.state_dict()[k]) for k, v in det.state_dict().items())


class TestAutoExport:
    def test_export_best_routes(self, irwm_env, vitdet_module):
        from irwm.detection.frozen import frozen_meta

        export_best = _load_entry("train_aligned").export_best
        out = export_best("aligned_mae_rgb", _fake_ckpt(irwm_env, vitdet_module, VITDET, ENC_MAE))
        assert out.name == "aligned_mae_rgb" and frozen_meta("aligned_mae_rgb")["detector"]["arch"] == "vitdet_mae"
        out = export_best("aligned_dinopca_ir", _fake_ckpt(irwm_env, make_dino(), DINOPCA, ENC_DINO, run="slurm-12"))
        meta = frozen_meta("aligned_dinopca_ir")
        assert meta["dataset"] == "aligned_v1" and meta["encoder"]["encoder_tag"].endswith("pca1152")
        with pytest.raises(FileExistsError):  # write-once
            export_best("aligned_dinopca_ir",
                        _fake_ckpt(irwm_env, make_dino(), DINOPCA, ENC_DINO, perturb=1e-3, run="slurm-13"))
        with pytest.raises(ValueError, match="aligned"):
            export_best("det_rgb", out)

    def test_fit_complete(self):
        from types import SimpleNamespace

        from lightning.pytorch.trainer.states import TrainerStatus

        fit_complete = _load_entry("train_aligned").fit_complete
        t = lambda status, epoch, interrupted=False: SimpleNamespace(  # noqa: E731
            state=SimpleNamespace(status=status), interrupted=interrupted, max_epochs=20, current_epoch=epoch)
        assert fit_complete(t(TrainerStatus.FINISHED, 20))
        assert not fit_complete(t(TrainerStatus.FINISHED, 13))  # requeue's partial run
        assert not fit_complete(t(TrainerStatus.INTERRUPTED, 20, True))

    def test_config(self, irwm_env):
        cfg = _compose("train_aligned", ["aligned=aligned_dinopca_rgb"])
        assert cfg.export.enabled is True and cfg.export.overwrite is False


# --------------------------------------------------------------------------- configs
def _compose(name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(PROJECT / "conf"), version_base="1.3"):
        return compose(name, overrides=overrides)


class TestConfigs:
    @pytest.mark.parametrize("head", HEADS)
    def test_heads(self, irwm_env, head):
        cfg = _compose("train_aligned", [f"aligned={head}"])
        kind, mod = head.split("_")[1], head.split("_")[2]
        assert cfg.name == head and cfg.dataset == "aligned_v1"
        assert cfg.modality == {"rgb": "rgb", "ir": "thermal"}[mod]
        assert cfg.trainer.max_epochs == 20 and cfg.optim.lr_drop_epoch == 14 and cfg.optim.lr == 1e-4
        assert cfg.trainer.check_val_every_n_epoch == 1 and cfg.trainer.precision == "16-mixed"
        assert cfg.trainer.gradient_clip_val == 0.1 and cfg.data.batch_size == 16 and cfg.data.flip_p == 0.5
        assert cfg.detector.num_classes == 2 and cfg.smoke.enabled is False
        assert head in cfg.wandb.tags
        size = tuple(POLICY_SIZE[cfg.encoder.policy])
        if kind == "mae":
            assert cfg.detector.arch == "vitdet_mae" and cfg.encoder.policy == "native640"
            assert size == (512, 640) and cfg.detector.preprocess_policy == "native640"
            assert list(cfg.detector.image_hw) == [512, 640] and cfg.detector.init == "mae"
            assert cfg.detector.optim.layer_decay == 0.7 and cfg.detector.use_act_checkpoint is True
        else:
            assert cfg.encoder.kind == kind and cfg.encoder.policy == "native" and size == (448, 560)
            assert cfg.encoder.attention == "sdpa" and cfg.detector.get("arch") is None
            assert cfg.detector.in_channels == (3072 if kind == "dino3072" else 1152)
            assert cfg.detector.in_proj == (1152 if kind == "dino3072" else None)
        _load_entry("train_aligned").check_config(cfg)

    def test_check_config_rejects(self, irwm_env):
        check = _load_entry("train_aligned").check_config
        with pytest.raises(ValueError, match="native640"):
            check(_compose("train_aligned", ["aligned=aligned_mae_rgb", "encoder.policy=native"]))
        with pytest.raises(ValueError, match="num_classes"):
            check(_compose("train_aligned", ["aligned=aligned_dinopca_rgb", "detector.num_classes=6"]))

    def test_aligned_mandatory_and_v2_untouched(self, irwm_env):
        from hydra.errors import ConfigCompositionException
        from omegaconf.errors import MissingMandatoryValue

        with pytest.raises((ConfigCompositionException, MissingMandatoryValue)):
            _compose("train_aligned", [])
        v2 = _compose("train", ["experiment=det_rgb"])
        assert v2.detector.num_classes == 6 and v2.trainer.max_epochs == 50 and v2.optim.lr_drop_epoch == 40

    def test_export_aligned(self, irwm_env):
        cfg = _compose("export_aligned", ["aligned=aligned_mae_ir", "source_run=slurm-3"])
        assert cfg.name == "aligned_mae_ir" and cfg.source_run == "slurm-3" and cfg.overwrite is False


# --------------------------------------------------------------------------- launcher
def _launch(irwm_env, *args, env=None):
    e = {**os.environ, **(env or {})}
    e.pop("MAE_GRES", None) if "MAE_GRES" not in (env or {}) else None
    return subprocess.run(["bash", str(REPO / "slurm" / "train_aligned.sh"), *args], cwd=REPO, env=e,
                          capture_output=True, text=True, timeout=120)


class TestLauncher:
    def test_dry_run_all(self, irwm_env):
        res = _launch(irwm_env, "ctcomp3", "--dry-run")
        assert res.returncode == 0, res.stderr
        lines = [ln for ln in res.stdout.splitlines() if ln.startswith("[dry-run ")]
        assert [re.match(r"\[dry-run (\S+)\]", ln)[1] for ln in lines] == HEADS
        assert "MAE_GRES is not set" in res.stderr
        for ln in lines:
            head = re.match(r"\[dry-run (\S+)\]", ln)[1]
            assert "IRWM_STAGE_DATA=1" in ln and "IRWM_STAGE_DATASETS=aligned/align" in ln
            assert f"aligned={head}" in ln and f"--job-name=irwm-{head} " in ln
            assert "projects/detection/train_aligned.py" in ln and "--gres=gpu:1 " in ln
            assert ("--time=360" if "_mae_" in head else "--time=210") in ln
            assert "--cpus-per-task=8" in ln and "smoke.enabled" not in ln

    def test_dry_run_smoke_subset_mae_gres(self, irwm_env):
        res = _launch(irwm_env, "ctcomp3", "--smoke", "--heads", "mae_ir,dinopca", "--dry-run", "--", "seed=3",
                      env={"MAE_GRES": "gpu:a100:1"})
        assert res.returncode == 0, res.stderr
        lines = [ln for ln in res.stdout.splitlines() if ln.startswith("[dry-run ")]
        assert [re.match(r"\[dry-run (\S+)\]", ln)[1] for ln in lines] == [
            "aligned_mae_ir", "aligned_dinopca_rgb", "aligned_dinopca_ir"]
        assert "--gres=gpu:a100:1" in lines[0] and "--mem=48G" in lines[0]
        assert all("--gres=gpu:1 " in ln for ln in lines[1:])
        assert all("smoke.enabled=true" in ln and "--time=45" in ln and "seed=3" in ln for ln in lines)
        assert "MAE_GRES is not set" not in res.stderr

    def test_submit_prints_ids_and_eval_command(self, irwm_env, tmp_path):
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        counter = tmp_path / "n"
        counter.write_text("100")
        (fake_bin / "sbatch").write_text(  # fake sbatch: logs its args, prints a job id
            "#!/usr/bin/env bash\n"
            f"n=$(( $(cat {counter}) + 1 )); echo $n > {counter}\n"
            f'echo "$@" >> {tmp_path / "calls"}\n'
            'echo "Submitted batch job $n"\n')
        (fake_bin / "sbatch").chmod(0o755)
        env = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "MAE_GRES": "gpu:h100:1"}
        res = _launch(irwm_env, "ctcomp3", "--heads", "dino3072,mae_rgb", env=env)
        assert res.returncode == 0, res.stderr
        assert "  aligned_dino3072_rgb 101\n  aligned_dino3072_ir 102\n  aligned_mae_rgb 103" in res.stdout
        assert ('SBATCH_EXTRA="--dependency=afterok:101:102:103" bash slurm/eval_aligned.sh ctcomp3 '
                "'heads=[aligned_dino3072_rgb,aligned_dino3072_ir,aligned_mae_rgb]'") in res.stdout
        calls = (tmp_path / "calls").read_text().splitlines()
        assert len(calls) == 3 and "--gres=gpu:h100:1" in calls[2] and "--gres=gpu:1 " in calls[0]

    def test_unknown_head(self, irwm_env):
        res = _launch(irwm_env, "ctcomp3", "--heads", "mae_xx", "--dry-run")
        assert res.returncode == 2 and "unknown head" in res.stderr


# --------------------------------------------------------------------------- GPU
@pytest.mark.gpu
class TestGpuFit:
    @pytest.mark.parametrize("arch", ["dino3072", "vitdet"])
    def test_fit_resume(self, tiny_aligned, tmp_path, arch):
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger

        from irwm.detection.detrex_adapter import select_msda

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        select_msda("auto")
        policy = "native640" if arch == "vitdet" else "native"
        dm = _dm(tiny_aligned, policy=policy, flip_p=0.5)

        def fit(max_steps, ckpt_path=None):
            m = make_vitdet(val_gt=dm.val_gt) if arch == "vitdet" else make_dino(DINO3072, 3072, val_gt=dm.val_gt)
            m.metrics_dir = tmp_path / "val_metrics"
            best = ModelCheckpoint(dirpath=tmp_path / "ckpt", monitor="val/AP", mode="max", save_top_k=1,
                                   save_last=True, filename="best-e{epoch:03d}", auto_insert_metric_name=False)
            tr = L.Trainer(accelerator="gpu", devices=1, max_steps=max_steps, precision="16-mixed",
                           check_val_every_n_epoch=1, gradient_clip_val=0.1, log_every_n_steps=1,
                           logger=CSVLogger(tmp_path, name="csv"), callbacks=[best], enable_progress_bar=False)
            tr.fit(m, datamodule=dm, ckpt_path=ckpt_path)
            return tr, m

        tr, m = fit(4)
        assert "val/AP_person" in tr.callback_metrics and m.gpu_info["name"] != "cpu"
        last = torch.load(tmp_path / "ckpt" / "last.ckpt", map_location="cpu", weights_only=False)
        assert last["global_step"] == 4 and last["irwm"]["gpu"]["name"] == torch.cuda.get_device_name()
        has_vit = any(k.startswith("detector.detr.backbone.net.blocks.") for k in last["state_dict"])
        assert has_vit == (arch == "vitdet") and not any(k.startswith("encoder.") for k in last["state_dict"])
        tr2, _ = fit(6, str(tmp_path / "ckpt" / "last.ckpt"))
        assert tr2.global_step == 6


# --------------------------------------------------------------------------- real assets
@pytest.mark.data
def test_real_dinopca_encode_unchanged():
    """``DetectorModule.encode`` through ``frozen_encoder_from_config`` == F8's ``encode_pca`` + permute."""
    from irwm.detection.lightning import DetectorModule
    from irwm.utils.paths import load_env

    load_env()
    m = DetectorModule(detector=DINOPCA, encoder={**ENC_DINO, "precision": "fp32", "attention": "plain"},
                       optim=OPTIM)
    x = torch.randn(1, 3, 448, 560)
    assert torch.equal(m.encode(x), m.encoder.encode_pca(x).permute(0, 3, 1, 2).contiguous())
