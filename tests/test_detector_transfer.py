"""F15 cross-modal detector transfer (spec/2026-10-28-cross-modal-detector-transfer, validation.md Groups 1-8).

CPU groups run on a tiny synthetic ``video_*_test`` tree with a stub encoder and F7's reduced detector,
in fp32. Fake frozen heads are written directly (``head.pt`` + ``meta.json``), not through
``export_frozen``. Group 7 (``gpu``) runs ``predict_split`` on CUDA under fp16 autocast; Group 8
(``data`` + ``gpu``) the real frozen heads and encoder on real thermal test frames.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
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
from hydra import compose, initialize_config_dir
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import nn
from torch.utils.data import DataLoader

from irwm.data.detection import FlirDetectionDataset, collate_detection
from irwm.data.flir import FLIR_DIRNAME, FlirSplit
from irwm.detection import DINODetector, cuda_ops_available
from irwm.detection.coco_val import coco_map
from irwm.detection import inference
from irwm.detection.inference import (check_encoder_config, check_outputs_free, detections_path, load_detections,
                                      predict_split, save_detections, shared_encoder_meta, with_person_car)
from irwm.utils.paths import repo_root

REPO = repo_root()
PROJECT = REPO / "projects" / "detection"
# F7 plan.md Step 7: reduced detector for fast tests; the 32x40 grid has >= 100 pyramid tokens.
REDUCED = {"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1,
           "transformer.decoder.num_layers": 2}
DETECTOR = {"in_channels": 1152, "stem_dim": 768, "num_classes": 6, "overrides": REDUCED}
TAG = "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152"
ENCODER = {"policy": "native", "precision": "fp16", "hub_version": "dinof_release", "attention": "sdpa",
           "encoder_tag": TAG}
OPTIM = {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 1e-4, "backbone_lr_factor": 1.0, "lr_drop_epoch": 40}
IMAGE_SIZE = (448, 560)
requires_cuda_ops = pytest.mark.skipif(not cuda_ops_available(), reason="needs CUDA and detrex's CUDA ops")

# --------------------------------------------------------------------------- tiny video test tree
# FLIR ids: 1 person, 2 bike, 3 car, 4 motor, 10 light, 12 sign; 8 is an ignore class.
CATEGORIES = [{"id": i, "name": n} for i, n in
              [(1, "person"), (2, "bike"), (3, "car"), (4, "motor"), (8, "truck"), (10, "light"), (12, "sign")]]
SIZES = {"thermal": (80, 64), "rgb": (100, 64)}  # (W, H); native crops RGB to x = 10..90, scale 7
IGNORE = [(8, [30, 20, 10, 10])]
# image_id -> (video_id, [(category_id, xywh)]); every box inside x = 10..90, so the same for both modalities.
FRAMES = {1: ("vidA", IGNORE),
          2: ("vidA", [(1, [20, 10, 8, 16]), (3, [40, 30, 20, 10])]),
          3: ("vidA", [(2, [50, 20, 8, 10])]),
          4: ("vidB", [(4, [15, 40, 12, 8]), (10, [60, 5, 4, 6])]),
          5: ("vidB", [(12, [70, 30, 6, 6]), (3, [30, 45, 15, 10]), (1, [50, 10, 6, 12])]),
          6: ("vidB", [])}
VIDEOS = {"videos": [{"id": "vidA", "tags": ["night", "rain"]}, {"id": "vidB", "tags": ["day"]}]}


def frame_index(image_id: int) -> int:
    return 10 * image_id


def write_tiny_video_test(flir_root: Path) -> None:
    """``coco.json``, ``index.json`` and noise JPEGs of ``video_{rgb,thermal}_test`` in FLIR's layout."""
    rng = np.random.default_rng(0)
    for modality, (w, h) in SIZES.items():
        sdir = flir_root / "FLIR_ADAS_v2" / f"video_{modality}_test"
        (sdir / "data").mkdir(parents=True)
        images, annotations = [], []
        for image_id, (vid, boxes) in FRAMES.items():
            fname = f"data/video-{vid}-frame-{frame_index(image_id):06d}-u{image_id}.jpg"
            shape = (h, w) if modality == "thermal" else (h, w, 3)
            Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(sdir / fname)
            images.append({"id": image_id, "file_name": fname, "width": w, "height": h,
                           "extra_info": {"video_id": vid}})
            for cat, bbox in boxes:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": cat,
                                    "bbox": [float(v) for v in bbox], "area": float(bbox[2] * bbox[3]),
                                    "iscrowd": 0})
        (sdir / "coco.json").write_text(json.dumps({"images": images, "annotations": annotations,
                                                    "categories": CATEGORIES}))
        (sdir / "index.json").write_text(json.dumps(VIDEOS))


@pytest.fixture
def tiny_flir(irwm_env) -> Path:
    root = irwm_env["data"] / FLIR_DIRNAME
    write_tiny_video_test(root)
    return root


def _split(modality: str) -> FlirSplit:
    return FlirSplit.load(modality, "test")  # type: ignore[arg-type]


class StubEncoder(nn.Module):
    """F8's stub (avg-pool 14 → fixed ``Linear(3, 1152)``), counting ``encode_pca`` calls."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3, 1152)
        self.requires_grad_(False)
        self.calls = 0

    def encode_pca(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        with torch.no_grad():
            return self.proj(F.avg_pool2d(x, 14).permute(0, 2, 3, 1)).float()


def reduced_detector(seed: int) -> DINODetector:
    torch.manual_seed(seed)
    return DINODetector(**DETECTOR).eval()


@pytest.fixture(scope="module")
def det_a() -> DINODetector:
    return reduced_detector(0)


@pytest.fixture(scope="module")
def det_b() -> DINODetector:
    return reduced_detector(1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fake_frozen(ckpt: Path, name: str, det: DINODetector, *, encoder: dict,
                      image_size: list[int] | None = None) -> Path:
    """``<ckpt>/detection/frozen/<name>/{head.pt, meta.json}`` with every key of F8 FR6."""
    out = ckpt / "detection" / "frozen" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(det.state_dict(), out / "head.pt")
    meta = {"name": name, "modality": {"det_rgb": "rgb", "det_ir": "thermal"}[name], "precision": "16-mixed",
            "source_ckpt": f"detection/slurm-1/best-e009-{name}.ckpt", "epoch": 9, "global_step": 1234,
            "val_metrics": {"metrics": {"AP": 0.3, "AP50": 0.6}}, "detector": DETECTOR, "encoder": dict(encoder),
            "image_size": list(image_size or IMAGE_SIZE), "head_sha256": _sha256(out / "head.pt"),
            "git": {"commit": "abc", "dirty": False}, "created_at": "2026-10-28T00:00:00+00:00"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


@pytest.fixture
def fake_heads(irwm_env, det_a, det_b) -> dict[str, Path]:
    return {"det_rgb": write_fake_frozen(irwm_env["ckpt"], "det_rgb", det_a, encoder=ENCODER),
            "det_ir": write_fake_frozen(irwm_env["ckpt"], "det_ir", det_b, encoder=ENCODER)}


# --------------------------------------------------------------------------- Group 1
class TestSharedEncoder:
    def test_equal(self, irwm_env, fake_heads):
        assert shared_encoder_meta(["det_rgb", "det_ir"]) == {"encoder": ENCODER, "image_size": (448, 560),
                                                              "precision": "16-mixed"}

    def test_attention_differs(self, irwm_env, det_a, det_b):
        write_fake_frozen(irwm_env["ckpt"], "det_rgb", det_a, encoder=ENCODER)
        write_fake_frozen(irwm_env["ckpt"], "det_ir", det_b, encoder=dict(ENCODER, attention="plain"))
        with pytest.raises(ValueError, match="attention") as e:
            shared_encoder_meta(["det_rgb", "det_ir"])
        assert "sdpa" in str(e.value) and "plain" in str(e.value)

    def test_image_size_differs(self, irwm_env, det_a, det_b):
        write_fake_frozen(irwm_env["ckpt"], "det_rgb", det_a, encoder=ENCODER)
        write_fake_frozen(irwm_env["ckpt"], "det_ir", det_b, encoder=ENCODER, image_size=[448, 896])
        with pytest.raises(ValueError, match="image_size"):
            shared_encoder_meta(["det_rgb", "det_ir"])

    def test_empty(self):
        with pytest.raises(ValueError):
            shared_encoder_meta([])


# --------------------------------------------------------------------------- Group 2
def _predict(detectors: dict, modality: str, stub: StubEncoder | None = None, **kwargs):
    return predict_split(detectors, _split(modality), stub or StubEncoder(), policy="native", image_size=IMAGE_SIZE,
                         batch_size=kwargs.pop("batch_size", 4), **kwargs)


def _per_image(dets: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for d in dets:
        out.setdefault(d["image_id"], []).append(d)
    return out


class TestPredictSplit:
    def test_two_heads(self, tiny_flir, det_a, det_b):
        stub = StubEncoder()
        split = _split("thermal")
        pred = _predict({"det_a": det_a, "det_b": det_b}, "thermal", stub)
        assert list(pred.detections) == ["det_a", "det_b"]
        assert pred.image_ids == [r.image_id for r in split.records] == [1, 2, 3, 4, 5, 6]
        assert stub.calls == math.ceil(6 / 4)  # once per batch, not once per head

        for dets in pred.detections.values():
            assert all(set(d) == {"image_id", "category_id", "bbox", "score"} for d in dets)
            assert {d["category_id"] for d in dets} <= {1, 2, 3, 4, 10, 12}
            assert all(len(v) <= 300 for v in _per_image(dets).values())
            assert [d["image_id"] for d in dets] == sorted(d["image_id"] for d in dets)
        assert [d["score"] for d in pred.detections["det_a"]] != [d["score"] for d in pred.detections["det_b"]]

        assert pred.encode_s >= 0
        assert set(pred.head_s) == {"det_a", "det_b"}
        assert all(isinstance(v, float) and v >= 0 for v in pred.head_s.values())

    def test_rgb_crop_offset(self, tiny_flir, det_a):
        dets = _predict({"det_a": det_a}, "rgb").detections["det_a"]
        assert dets
        for d in dets:
            x, _, w, _ = d["bbox"]
            assert x >= 10 - 1e-3 and x + w <= 90 + 1e-3

    def test_training_mode(self, tiny_flir):
        det = reduced_detector(2).train()
        stub = StubEncoder()
        with pytest.raises(ValueError, match="training"):
            _predict({"h": det}, "thermal", stub)
        assert stub.calls == 0

    def test_empty(self, tiny_flir):
        with pytest.raises(ValueError):
            _predict({}, "thermal")


# --------------------------------------------------------------------------- Group 3
@pytest.mark.parametrize("modality", ["thermal", "rgb"])
def test_equals_f8_validation(tiny_flir, det_a, modality):
    from irwm.detection.lightning import DetectorModule

    split = _split(modality)
    path_a = predict_split({"h": det_a}, split, StubEncoder(), policy="native", image_size=IMAGE_SIZE,
                           batch_size=4).detections["h"]

    module = DetectorModule(detector=DETECTOR, encoder=ENCODER, optim=OPTIM, encoder_module=StubEncoder())
    module.detector.load_state_dict(det_a.state_dict())
    module.eval()
    loader = DataLoader(FlirDetectionDataset(split, flip_p=0.0, filter_empty=False), batch_size=4,
                        collate_fn=collate_detection)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            module.validation_step(batch, i)
    path_b = [d for image_id in sorted(module._val_dets) for d in module._val_dets[image_id]]
    assert path_a and path_a == path_b  # same dicts, |Δ| = 0 on every bbox value and score


# --------------------------------------------------------------------------- Group 4
def _some_detections(split: FlirSplit) -> list[dict]:
    gt = split.to_coco_gt()
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
             "score": 0.5 + 0.01 * i} for i, a in enumerate(gt["annotations"]) if not a["iscrowd"]]


class TestDetectionsFile:
    def test_path(self, irwm_env, fake_heads):
        assert detections_path("det_rgb", "thermal") == (
            irwm_env["output"] / "detection" / "detections" / TAG / "det_rgb" / "thermal_test.json.gz")

    def test_round_trip(self, tiny_flir, fake_heads, tmp_path):
        split = _split("thermal")
        dets = _some_detections(split)
        path = tmp_path / "d" / "x.json.gz"
        sha = save_detections(path, dets, head="det_rgb", split=split)
        obj = load_detections(path)
        assert obj["detections"] == dets and obj["n_detections"] == len(dets) and obj["n_images"] == 6
        assert obj["format"] == "irwm.detections/1" and obj["head"] == "det_rgb" and obj["head_modality"] == "rgb"
        assert obj["modality"] == "thermal" and obj["split"] == "test" and obj["precision"] == "16-mixed"
        assert [im["image_id"] for im in obj["images"]] == [1, 2, 3, 4, 5, 6]
        for im in obj["images"]:
            assert im["video_id"] == FRAMES[im["image_id"]][0]
            assert im["frame_index"] == frame_index(im["image_id"])
            assert (im["width"], im["height"]) == SIZES["thermal"] and im["file_name"].startswith("data/video-")
        assert obj["video_tags"] == {"vidA": ["night", "rain"], "vidB": ["day"]}
        assert all(im["video_id"] in obj["video_tags"] for im in obj["images"])  # image_id → video_id → tags
        meta = json.loads((fake_heads["det_rgb"] / "meta.json").read_text())
        assert obj["head_sha256"] == meta["head_sha256"] and obj["encoder"] == ENCODER
        assert obj["image_size"] == [448, 560]
        assert "commit" in obj["git"] and obj["created_at"] and obj["device"]
        assert sha == _sha256(path)

    def test_deterministic_bytes(self, tiny_flir, fake_heads, tmp_path, monkeypatch):
        monkeypatch.setattr(inference, "_now", lambda: "2026-10-28T00:00:00+00:00")
        monkeypatch.setattr(inference, "git_info", lambda: {"commit": "abc", "dirty": False})
        monkeypatch.setattr(inference, "_device_name", lambda: "cpu")
        split = _split("thermal")
        dets = _some_detections(split)
        sha1 = save_detections(tmp_path / "a.json.gz", dets, head="det_ir", split=split)
        sha2 = save_detections(tmp_path / "b" / "a.json.gz", copy.deepcopy(dets), head="det_ir", split=split)
        assert sha1 == sha2 == _sha256(tmp_path / "a.json.gz") == _sha256(tmp_path / "b" / "a.json.gz")

    def test_write_once(self, tiny_flir, fake_heads, tmp_path):
        split = _split("thermal")
        dets = _some_detections(split)
        path = tmp_path / "x.json.gz"
        save_detections(path, dets, head="det_ir", split=split)
        before = path.read_bytes()
        with pytest.raises(FileExistsError):
            save_detections(path, dets[:1], head="det_ir", split=split)
        assert path.read_bytes() == before
        save_detections(path, dets[:1], head="det_ir", split=split, overwrite=True)
        assert load_detections(path)["n_detections"] == 1
        assert not list(tmp_path.glob("*.tmp"))

    def test_unknown_image_id(self, tiny_flir, fake_heads, tmp_path):
        split = _split("thermal")
        bad = _some_detections(split) + [{"image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.1}]
        out = tmp_path / "dets"
        with pytest.raises(ValueError, match="99"):
            save_detections(out / "x.json.gz", bad, head="det_ir", split=split)
        assert not out.exists()

    def test_check_outputs_free(self, tmp_path):
        p1, p2 = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
        check_outputs_free([p1, p2], overwrite=False)
        p2.touch()
        with pytest.raises(FileExistsError, match=str(p2)):
            check_outputs_free([p1, p2], overwrite=False)
        check_outputs_free([p1, p2], overwrite=True)


# --------------------------------------------------------------------------- Group 5
class TestPersonCar:
    BASE = {"AP": 0.3, "AP50": 0.6, "AP_person": 0.4, "AP_car": 0.6, "AP50_person": 0.8, "AP50_car": 0.6}

    def test_mean(self):
        m = with_person_car(self.BASE)
        assert m["AP_person_car"] == pytest.approx(0.5, abs=1e-15)
        assert m["AP50_person_car"] == pytest.approx(0.7, abs=1e-15)
        assert {k: m[k] for k in self.BASE} == self.BASE

    def test_nan(self):
        assert math.isnan(with_person_car(dict(self.BASE, AP_car=math.nan))["AP_person_car"])

    def test_restricted_cocoeval(self, tiny_flir):
        gt = _split("thermal").to_coco_gt("normalised")
        dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
                for a in gt["annotations"] if not a["iscrowd"]]
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = copy.deepcopy(gt)
            coco_gt.createIndex()
            E = COCOeval(coco_gt, coco_gt.loadRes(copy.deepcopy(dets)), iouType="bbox")
            E.params.catIds = [1, 3]
            E.evaluate()
            E.accumulate()
            E.summarize()
        assert with_person_car(coco_map(gt, dets))["AP_person_car"] == pytest.approx(E.stats[0], abs=1e-12)


# --------------------------------------------------------------------------- Group 6
def _compose(name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(PROJECT / "conf"), version_base="1.3"):
        return compose(name, overrides=overrides or [])


class TestConfig:
    def test_transfer_yaml(self, irwm_env):
        cfg = _compose("transfer")
        assert list(cfg.heads) == ["det_rgb", "det_ir"] and list(cfg.test_modalities) == ["rgb", "thermal"]
        assert cfg.split == "test" and cfg.batch_size == 16 and cfg.overwrite is False
        assert cfg.encoder.attention == "sdpa" and cfg.encoder.policy == "native"
        assert "transfer" in cfg.wandb.tags and cfg.project == "detection"

    def test_check_encoder_config(self, irwm_env, fake_heads):
        shared = shared_encoder_meta(["det_rgb", "det_ir"])
        cfg = {"policy": "native", "precision": "fp16", "hub_version": "dinof_release", "attention": "plain"}
        with pytest.raises(ValueError, match="attention"):
            check_encoder_config(cfg, shared)
        check_encoder_config(dict(cfg, attention="sdpa"), shared)
        check_encoder_config(_compose("transfer").encoder, shared)

    @pytest.mark.skipif(cuda_ops_available(), reason="checks the no-CUDA-ops exit")
    def test_exits_without_cuda_ops(self, irwm_env):
        env = {**os.environ, "WANDB_MODE": "disabled"}
        res = subprocess.run([sys.executable, str(PROJECT / "transfer.py")], cwd=REPO, env=env,
                             capture_output=True, text=True, timeout=300)
        assert res.returncode != 0 and "cuda_ops_available" in res.stderr


# --------------------------------------------------------------------------- Group 7
@pytest.mark.gpu
@requires_cuda_ops
def test_cuda_autocast(tiny_flir):
    det = reduced_detector(0).cuda()
    stub = StubEncoder().cuda()
    pred = _predict({"h": det}, "thermal", stub, device="cuda")
    assert pred.image_ids == [1, 2, 3, 4, 5, 6]
    dets = pred.detections["h"]
    assert dets and all(math.isfinite(d["score"]) and 0.0 <= d["score"] <= 1.0 for d in dets)
    assert all(len(v) <= 300 for v in _per_image(dets).values())
    assert all(p.dtype == torch.float32 for p in det.parameters())  # autocast does not cast weights


# --------------------------------------------------------------------------- Group 8
@pytest.mark.data
@pytest.mark.gpu
@requires_cuda_ops
def test_real_heads_thermal_test():
    from irwm.detection.frozen import frozen_dir, load_frozen_detector
    from irwm.features.encoder import DinoV2RegEncoder, set_attention
    from irwm.utils.paths import load_env

    load_env()
    if not (frozen_dir("det_rgb") / "meta.json").is_file():
        pytest.skip("frozen det_rgb not exported")
    shared = shared_encoder_meta(["det_rgb", "det_ir"])
    enc = shared["encoder"]
    assert enc["attention"] == "sdpa" and enc["policy"] == "native" and shared["image_size"] == (448, 560)

    full = _split("thermal")
    split = FlirSplit("thermal", "test", full.root, full.records[:4])
    encoder = DinoV2RegEncoder.from_assets(precision=enc["precision"], hub_version=enc["hub_version"], device="cuda")
    set_attention(encoder.backbone, enc["attention"])
    heads = {h: load_frozen_detector(h, device="cuda") for h in ("det_rgb", "det_ir")}
    pred = predict_split(heads, split, encoder, policy=enc["policy"], image_size=shared["image_size"], device="cuda")
    assert len(pred.detections) == 2
    for dets in pred.detections.values():
        per = _per_image(dets)
        assert len(per) == 4 and set(per) == {r.image_id for r in split.records}
        assert all(1 <= len(v) <= 300 for v in per.values())
        for d in dets:
            x, y, w, h = d["bbox"]
            assert x >= -1 and y >= -1 and x + w <= 640 + 1 and y + h <= 512 + 1
