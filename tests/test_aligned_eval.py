"""F19 aligned FLIR v1 evaluation (``irwm.eval.aligned_eval``, ``irwm.eval.aligned_table``,
the aligned generalisation of ``irwm.detection.inference``).

CPU only: a tiny synthetic ``aligned/align`` tree (4 test pairs, 640×512 noise), stub DINO encoders
(avg-pool 14 → fixed linear), F7's reduced ``DINODetector`` heads with 2 classes, and a fake image head
for the MAE ViTDet heads (one test also runs a reduced real ``ViTDetDINO`` through ``load_frozen_vitdet``).
Fake frozen heads are written directly (``head.pt`` + ``meta.json``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import nn

from irwm.data.aligned import ALIGNED_LABEL_TO_CATEGORY_ID, AlignedSplit
from irwm.detection import DINODetector
from irwm.detection.inference import (HeadGroup, check_same_dataset, detections_path, load_detections,
                                      predict_heads, predict_split, save_detections)
from irwm.eval import aligned_eval
from irwm.eval.aligned_eval import evaluate_aligned, merge_aligned, plan_groups
from irwm.eval.aligned_table import (ALL_HEADS, parse_aligned_head, render_table_b_markdown, rescore_detections,
                                     score_aligned, transfer_ratios)
from irwm.utils.paths import repo_root

PROJECT = repo_root() / "projects" / "detection"
# 150 queries x 2 classes >= DINO's 300 eval boxes; F7's reduced head otherwise.
REDUCED = {"num_queries": 150, "dn_number": 10, "transformer.encoder.num_layers": 1,
           "transformer.decoder.num_layers": 2}
DET = {"dinopca": {"in_channels": 1152, "stem_dim": 768, "num_classes": 2, "overrides": REDUCED},
       "dino3072": {"in_channels": 3072, "in_proj": 1152, "stem_dim": 768, "num_classes": 2, "overrides": REDUCED}}
TAGS = {"dinopca": "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152",
        "dino3072": "dinov2_vitb14reg_448x560_l3-6-9-12_full3072",
        "mae": "mae_vitb16_512x640_vitdet-ft"}


def enc_meta(kind: str) -> dict:
    if kind == "mae":
        return {"kind": "mae", "policy": "native640", "encoder_tag": TAGS["mae"], "precision": "fp16"}
    return {"kind": kind, "policy": "native", "precision": "fp16", "hub_version": "dinof_release",
            "attention": "sdpa", "encoder_tag": TAGS[kind]}


# --------------------------------------------------------------------------- tiny aligned tree
TEST_IDS = [8864, 8865, 8866, 8867]
OBJECTS = {8864: [("person", (100, 100, 130, 180)), ("car", (300, 250, 420, 330)), ("dog", (1, 1, 5, 5))],
           8865: [("car", (10, 300, 200, 400))],
           8866: [("person", (500, 200, 530, 260)), ("person", (540, 200, 570, 270))],
           8867: [("car", (200, 200, 260, 240)), ("bicycle", (0, 0, 10, 10))]}


def _voc(objects) -> str:
    objs = "".join(f"<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax>"
                   f"<ymax>{b[3]}</ymax></bndbox></object>" for n, b in objects)
    return f"<annotation><size><width>640</width><height>512</height><depth>3</depth></size>{objs}</annotation>"


@pytest.fixture
def tiny_aligned(irwm_env) -> Path:
    root = irwm_env["data"] / "aligned" / "align"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    (root / "align_train.txt").write_text("FLIR_00001_PreviewData\n")
    (root / "align_validation.txt").write_text("".join(f"FLIR_{i:05d}_PreviewData\n" for i in TEST_IDS))
    rng = np.random.default_rng(0)
    for i in TEST_IDS:
        (root / "Annotations" / f"FLIR_{i:05d}_PreviewData.xml").write_text(_voc(OBJECTS[i]))
        Image.fromarray(rng.integers(0, 256, (512, 640), dtype=np.uint8)).save(
            root / "JPEGImages" / f"FLIR_{i:05d}_PreviewData.jpeg")
        Image.fromarray(rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)).save(
            root / "JPEGImages" / f"FLIR_{i:05d}_RGB.jpg")
    return root


# --------------------------------------------------------------------------- fakes
class StubEncoder(nn.Module):
    """``FrozenEncoder`` stand-in: avg-pool 14 → fixed ``Linear(3, C)``, channels-first; counts calls."""

    def __init__(self, kind: str = "dinopca") -> None:
        super().__init__()
        self.kind, self.policy = kind, "native"
        self.out_channels = {"dinopca": 1152, "dino3072": 3072}[kind]
        torch.manual_seed(0)
        self.proj = nn.Linear(3, self.out_channels)
        self.requires_grad_(False)
        self.calls = 0

    def encode_pca(self, x: torch.Tensor) -> torch.Tensor:  # predict_split's interface (channels-last)
        with torch.no_grad():
            return self.proj(F.avg_pool2d(x, 14).permute(0, 2, 3, 1)).float()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.encode_pca(x).permute(0, 3, 1, 2).contiguous()


class FakeImageHead(nn.Module):
    """MAE ViTDet stand-in: takes the 512×640 images, returns 5 deterministic boxes per image."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.shapes: list[tuple] = []
        self.eval()

    def forward(self, images: torch.Tensor, image_size=None):
        self.shapes.append((tuple(images.shape), tuple(image_size)))
        out = []
        for img in images:
            m = float(img.mean())
            boxes = torch.tensor([[100 + 10 * k, 100, 160 + 10 * k, 200] for k in range(5)], dtype=torch.float32)
            scores = torch.linspace(0.9, 0.1, 5) * (0.5 + 0.1 * math.tanh(m + self.seed))
            out.append({"boxes": boxes, "scores": scores, "labels": torch.tensor([0, 1, 0, 1, 1])})
        return out


def reduced_detector(kind: str, seed: int) -> DINODetector:
    torch.manual_seed(seed)
    return DINODetector(**DET[kind]).eval()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fake_frozen(ckpt: Path, name: str, *, state: dict | None = None, detector: dict | None = None,
                      encoder: dict | None = None, dataset: str | None = "aligned_v1",
                      image_size: list[int] | None = None, modality: str | None = None) -> Path:
    backbone = name.split("_")[1] if name.startswith("aligned_") else "dinopca"
    out = ckpt / "detection" / "frozen" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state if state is not None else {"w": torch.zeros(1)}, out / "head.pt")
    if detector is None:
        detector = {"arch": "vitdet_mae", "num_classes": 2} if backbone == "mae" else DET[backbone]
    meta = {"name": name, "modality": modality or ("thermal" if name.endswith("_ir") else "rgb"),
            "precision": "16-mixed", "source_ckpt": f"detection/slurm-1/{name}.ckpt", "epoch": 14,
            "global_step": 99, "val_metrics": {"metrics": {"AP": 0.5}}, "detector": detector,
            "encoder": encoder or enc_meta(backbone),
            "image_size": image_size or ([512, 640] if backbone == "mae" else [448, 560]),
            "head_sha256": _sha256(out / "head.pt"), "git": {"commit": "abc", "dirty": False},
            "created_at": "2026-11-13T00:00:00+00:00"}
    if dataset is not None:
        meta["dataset"] = dataset
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


_DETS: dict[str, DINODetector] = {}


def cached_detector(name: str) -> DINODetector:
    if name not in _DETS:
        _DETS[name] = reduced_detector(parse_aligned_head(name)[0], ALL_HEADS.index(name))
    return _DETS[name]


@pytest.fixture
def six_heads(irwm_env) -> list[str]:
    for h in ALL_HEADS:
        state = None if h.startswith("aligned_mae") else cached_detector(h).state_dict()
        write_fake_frozen(irwm_env["ckpt"], h, state=state)
    return list(ALL_HEADS)


def fake_loader(name, device):
    """``head_loader`` for ``evaluate_aligned``: real ``load_frozen_detector`` for DINO heads, fake MAE."""
    if name.startswith("aligned_mae"):
        return FakeImageHead(seed=ALL_HEADS.index(name))
    return aligned_eval.load_aligned_head(name, device)


class EncoderSpy:
    def __init__(self) -> None:
        self.built: list[str] = []
        self.encoders: dict[str, StubEncoder] = {}

    def __call__(self, meta, device):
        self.built.append(meta["kind"])
        self.encoders[meta["kind"]] = StubEncoder(meta["kind"])
        return self.encoders[meta["kind"]]


def _split(modality: str) -> AlignedSplit:
    return AlignedSplit.load(modality, "test")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- Group 1: planning, R12
class TestPlan:
    def test_six_heads(self, six_heads):
        plan = plan_groups(six_heads)
        assert list(plan) == ["dinopca", "dino3072", "images_native640"]
        assert plan["dinopca"]["heads"] == ["aligned_dinopca_rgb", "aligned_dinopca_ir"]
        assert plan["dino3072"]["encoder"]["kind"] == "dino3072" and plan["dino3072"]["image_size"] == (448, 560)
        assert plan["images_native640"]["encoder"] is None and plan["images_native640"]["image_size"] == (512, 640)

    def test_subset(self, six_heads):
        assert list(plan_groups(["aligned_mae_rgb"])) == ["images_native640"]

    def test_r12_meta_dataset(self, irwm_env, six_heads):
        write_fake_frozen(irwm_env["ckpt"], "aligned_dinopca_ir", dataset="flir_v2")
        with pytest.raises(ValueError, match="R12"):
            plan_groups(["aligned_dinopca_rgb", "aligned_dinopca_ir"])

    def test_r12_v2_head_name(self, irwm_env, six_heads):
        write_fake_frozen(irwm_env["ckpt"], "det_rgb", dataset=None, encoder=enc_meta("dinopca"))
        with pytest.raises(ValueError, match="not an aligned head"):
            plan_groups(["det_rgb"])
        with pytest.raises(ValueError, match="R12"):
            check_same_dataset("det_rgb", "aligned_v1")  # metas without dataset are FLIR v2

    def test_group_disagrees(self, irwm_env, six_heads):
        write_fake_frozen(irwm_env["ckpt"], "aligned_dinopca_ir", encoder=dict(enc_meta("dinopca"), attention="plain"),
                          state=cached_detector("aligned_dinopca_ir").state_dict())
        with pytest.raises(ValueError, match="attention"):
            plan_groups(["aligned_dinopca_rgb", "aligned_dinopca_ir"])

    def test_repeated(self, six_heads):
        with pytest.raises(ValueError, match="repeated"):
            plan_groups(["aligned_mae_rgb", "aligned_mae_rgb"])


# --------------------------------------------------------------------------- Group 2: predict_heads
class TestPredictHeads:
    def test_groups_share_encodings(self, tiny_aligned):
        pca, full, img = StubEncoder("dinopca"), StubEncoder("dino3072"), FakeImageHead()
        groups = {"dinopca": HeadGroup({"a": reduced_detector("dinopca", 0), "b": reduced_detector("dinopca", 1)},
                                       "native", (448, 560), pca),
                  "dino3072": HeadGroup({"c": reduced_detector("dino3072", 2)}, "native", (448, 560), full),
                  "img": HeadGroup({"m": img}, "native640", (512, 640))}
        pred = predict_heads(groups, _split("thermal"), batch_size=3, label_to_category_id=ALIGNED_LABEL_TO_CATEGORY_ID)
        assert pred.image_ids == TEST_IDS and list(pred.detections) == ["a", "b", "c", "m"]
        assert pca.calls == full.calls == 2  # once per batch per encoder, not per head
        assert img.shapes == [((3, 3, 512, 640), (512, 640)), ((1, 3, 512, 640), (512, 640))]
        for dets in pred.detections.values():
            assert dets and {d["category_id"] for d in dets} <= {1, 2}
            assert [d["image_id"] for d in dets] == sorted(d["image_id"] for d in dets)
        # native640 on a 640x512 frame is the identity geometry: boxes come back unchanged.
        assert pred.detections["m"][0]["bbox"] == pytest.approx([100.0, 100.0, 60.0, 100.0])
        assert set(pred.encode_s) == {"dinopca", "dino3072", "img"} and set(pred.head_s) == {"a", "b", "c", "m"}

    def test_equals_predict_split(self, tiny_aligned):
        det = reduced_detector("dinopca", 0)
        split = _split("rgb")
        a = predict_split({"h": det}, split, StubEncoder(), policy="native", image_size=(448, 560),
                          batch_size=2).detections["h"]
        b = predict_heads({"g": HeadGroup({"h": det}, "native", (448, 560), StubEncoder())}, split, batch_size=2,
                          label_to_category_id=ALIGNED_LABEL_TO_CATEGORY_ID).detections["h"]
        assert a and a == b

    def test_rejects(self, tiny_aligned):
        with pytest.raises(ValueError):
            predict_heads({}, _split("rgb"))
        train = reduced_detector("dinopca", 0).train()
        with pytest.raises(ValueError, match="training"):
            predict_heads({"g": HeadGroup({"h": train}, "native", (448, 560), StubEncoder())}, _split("rgb"))
        with pytest.raises(ValueError, match="repeat"):
            predict_heads({"g1": HeadGroup({"h": FakeImageHead()}, "native640", (512, 640)),
                           "g2": HeadGroup({"h": FakeImageHead()}, "native640", (512, 640))}, _split("rgb"))


# --------------------------------------------------------------------------- Group 3: detections files
def _gt_dets(split) -> list[dict]:
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
            for a in split.to_coco_gt()["annotations"]]


F15_KEYS = {"format", "head", "head_sha256", "head_modality", "encoder", "image_size", "precision", "modality",
            "split", "n_images", "n_detections", "images", "video_tags", "detections", "git", "created_at", "device"}


class TestDetectionsFile:
    def test_aligned_round_trip(self, tiny_aligned, six_heads, tmp_path):
        split = _split("thermal")
        dets = _gt_dets(split)
        path = detections_path("aligned_dino3072_rgb", "thermal")
        assert path.parts[-3:] == (TAGS["dino3072"], "aligned_dino3072_rgb", "thermal_test.json.gz")
        sha = save_detections(path, dets, head="aligned_dino3072_rgb", split=split)
        obj = load_detections(path)
        assert set(obj) == (F15_KEYS - {"video_tags"}) | {"dataset"} and obj["dataset"] == "aligned_v1"
        assert obj["detections"] == dets and obj["head_modality"] == "rgb" and obj["modality"] == "thermal"
        assert [im["image_id"] for im in obj["images"]] == TEST_IDS
        assert all(set(im) == {"image_id", "file_name", "width", "height"} for im in obj["images"])
        assert sha == _sha256(path)
        assert rescore_detections(path) == score_aligned(split.to_coco_gt(), dets)
        assert rescore_detections(path)["AP_person_car"] == pytest.approx(1.0)

    def test_mae_tag_path(self, six_heads):
        assert detections_path("aligned_mae_ir", "rgb").parts[-3:] == (TAGS["mae"], "aligned_mae_ir",
                                                                       "rgb_test.json.gz")

    def test_r12_save(self, tiny_aligned, six_heads, irwm_env, tmp_path):
        write_fake_frozen(irwm_env["ckpt"], "det_ir", dataset=None, encoder=enc_meta("dinopca"), modality="thermal")
        split = _split("thermal")
        with pytest.raises(ValueError, match="R12"):
            save_detections(tmp_path / "out" / "x.json.gz", _gt_dets(split), head="det_ir", split=split)
        v2 = SimpleNamespace(modality="thermal", split="test", records=split.records)  # not an AlignedSplit
        with pytest.raises(ValueError, match="R12"):
            save_detections(tmp_path / "out" / "y.json.gz", [], head="aligned_mae_ir", split=v2)
        assert not (tmp_path / "out").exists()

    def test_flir_v2_keys_unchanged(self, irwm_env, tmp_path, monkeypatch):
        """A FLIR v2 split (records with video ids, ``video_tags``) gets F15's exact keys, no ``dataset``."""
        write_fake_frozen(irwm_env["ckpt"], "det_ir", dataset=None, encoder=enc_meta("dinopca"), modality="thermal")
        rec = SimpleNamespace(image_id=7, file_name="data/x.jpg", video_id="vidA", frame_index=3, width=640,
                              height=512)
        v2 = SimpleNamespace(modality="thermal", split="test", records=[rec], video_tags=lambda: {"vidA": ["day"]})
        save_detections(tmp_path / "v2.json.gz", [], head="det_ir", split=v2)
        obj = load_detections(tmp_path / "v2.json.gz")
        assert set(obj) == F15_KEYS and obj["video_tags"] == {"vidA": ["day"]}
        assert obj["images"] == [{"image_id": 7, "file_name": "data/x.jpg", "video_id": "vidA", "frame_index": 3,
                                  "width": 640, "height": 512}]

    def test_rescore_rejects_non_aligned(self, tmp_path):
        p = tmp_path / "v2.json.gz"
        with gzip.GzipFile(p, "wb", mtime=0) as f:
            f.write(json.dumps({"format": "irwm.detections/1", "modality": "rgb", "split": "test"}).encode())
        with pytest.raises(ValueError, match="R12"):
            rescore_detections(p)


# --------------------------------------------------------------------------- Group 4: end to end
class TestEvaluateAligned:
    def test_six_heads(self, tiny_aligned, six_heads, irwm_env):
        spy = EncoderSpy()
        res = evaluate_aligned(six_heads, device="cpu", batch_size=4, head_loader=fake_loader, encoder_builder=spy)
        assert spy.built == ["dinopca", "dino3072"]  # one encoder per kind, built once for both modalities
        assert spy.encoders["dinopca"].calls == spy.encoders["dino3072"].calls == 2  # 1 batch × 2 modalities
        assert set(res["cells"]) == set(ALL_HEADS) and all(set(c) == {"rgb", "thermal"} for c in res["cells"].values())
        for c in (c for per in res["cells"].values() for c in per.values()):
            assert {"AP", "AP50", "APs", "APm", "APl", "AP_person", "AP50_person", "AP_car", "AP50_car",
                    "AP_person_car", "AP50_person_car"} <= set(c)
        assert res["rescore_identical"] == {h: {"rgb": True, "thermal": True} for h in ALL_HEADS}
        assert res["n_images"] == {"rgb": 4, "thermal": 4}
        # transfer: cross ÷ same backbone's in-domain head on the same test modality
        r = res["transfer_ratio"]
        assert set(r) == set(ALL_HEADS) and all(len(v) == 1 for v in r.values())
        assert set(r["aligned_mae_rgb"]) == {"thermal"} and set(r["aligned_mae_ir"]) == {"rgb"}
        a, ref = res["cells"]["aligned_mae_rgb"]["thermal"], res["cells"]["aligned_mae_ir"]["thermal"]
        if ref["AP50_person_car"] > 0:
            assert r["aligned_mae_rgb"]["thermal"]["AP50_person_car"] == pytest.approx(
                a["AP50_person_car"] / ref["AP50_person_car"])
        for h in ALL_HEADS:
            for m in ("rgb", "thermal"):
                info = res["detections"][h][m]
                path = irwm_env["output"] / info["path"]
                assert path == detections_path(h, m) and info["sha256"] == _sha256(path)
                assert rescore_detections(path) == res["cells"][h][m]
        json.dumps(res)  # the run JSON is serialisable
        # write-once: a rerun fails before anything is loaded
        with pytest.raises(FileExistsError):
            evaluate_aligned(["aligned_mae_rgb"], device="cpu", head_loader=fake_loader, encoder_builder=spy)
        assert spy.built == ["dinopca", "dino3072"]

    def test_subset_and_smoke(self, tiny_aligned, six_heads, irwm_env):
        spy = EncoderSpy()
        res = evaluate_aligned(["aligned_mae_rgb"], device="cpu", head_loader=fake_loader, encoder_builder=spy,
                               test_modalities=["thermal"], max_images=2)
        assert spy.built == [] and list(res["groups"]) == ["images_native640"]
        assert res["n_images"] == {"thermal": 2} and res["transfer_ratio"] == {}
        assert res["detections"]["aligned_mae_rgb"]["thermal"]["path"].endswith("thermal_test-first2.json.gz")
        assert res["rescore_identical"] == {"aligned_mae_rgb": {"thermal": True}}
        assert not detections_path("aligned_mae_rgb", "thermal").exists()  # the real file is never written

    def test_real_vitdet_loader(self, tiny_aligned, irwm_env):
        """``load_aligned_head`` routes MAE metas to ``load_frozen_vitdet``; a reduced real ViTDetDINO runs."""
        from irwm.detection.vitdet import vitdet_from_config

        cfg = {"num_classes": 2, "image_hw": [512, 640], "use_act_checkpoint": False, "init": "none",
               "overrides": {"backbone.net.depth": 1, "backbone.net.window_block_indexes": [],
                             **REDUCED}}
        try:
            torch.manual_seed(0)
            model = vitdet_from_config(cfg)
        except (KeyError, ValueError) as e:  # the override keys belong to irwm.detection.vitdet
            pytest.skip(f"reduced ViTDetDINO config not accepted: {e}")
        detector = {"arch": "vitdet_mae", **{k: v for k, v in cfg.items() if k != "init"}}
        write_fake_frozen(irwm_env["ckpt"], "aligned_mae_ir", state=model.state_dict(), detector=detector)
        head = aligned_eval.load_aligned_head("aligned_mae_ir", "cpu")
        assert type(head).__name__ == "ViTDetDINO" and not head.training
        res = evaluate_aligned(["aligned_mae_ir"], device="cpu", test_modalities=["thermal"], max_images=1,
                               encoder_builder=EncoderSpy())
        dets = load_detections(irwm_env["output"] / res["detections"]["aligned_mae_ir"]["thermal"]["path"])
        assert dets["n_detections"] > 0 and {d["category_id"] for d in dets["detections"]} <= {1, 2}


# --------------------------------------------------------------------------- Group 4b: merge
MERGE_HEADS = ["aligned_dinopca_rgb", "aligned_dinopca_ir", "aligned_mae_rgb", "aligned_mae_ir"]


class TestMerge:
    def test_subset_runs_merge_to_full(self, tiny_aligned, six_heads, irwm_env):
        a = evaluate_aligned(MERGE_HEADS[:2], device="cpu", head_loader=fake_loader, encoder_builder=EncoderSpy())
        b = evaluate_aligned(MERGE_HEADS[2:], device="cpu", head_loader=fake_loader, encoder_builder=EncoderSpy())
        merged = merge_aligned(ALL_HEADS)
        assert merged["cells"] == {**a["cells"], **b["cells"]}  # re-scored from disk, identical
        assert set(merged["transfer_ratio"]) == set(MERGE_HEADS)  # ratios across the two runs
        assert merged["missing"] == [f"detection/detections/{TAGS['dino3072']}/{h}/{m}_test.json.gz"
                                     for h in ("aligned_dino3072_rgb", "aligned_dino3072_ir")
                                     for m in ("rgb", "thermal")]
        assert merged["detections"]["aligned_mae_ir"]["rgb"] == {
            **b["detections"]["aligned_mae_ir"]["rgb"],
            **{k: merged["detections"]["aligned_mae_ir"]["rgb"][k] for k in ("created_at", "device", "git")}}
        assert "dino3072" not in render_table_b_markdown(merged).split("**Per cell**")[0]

        # stale file: the head was re-exported after its detections were saved
        write_fake_frozen(irwm_env["ckpt"], "aligned_mae_ir", state={"w": torch.ones(1)})
        with pytest.raises(ValueError, match="stale"):
            merge_aligned(["aligned_mae_ir"])

    def test_nothing_saved(self, tiny_aligned, six_heads):
        with pytest.raises(FileNotFoundError):
            merge_aligned(["aligned_mae_rgb"])

    def test_script_merge(self, tiny_aligned, six_heads, irwm_env):
        evaluate_aligned(["aligned_mae_rgb", "aligned_mae_ir"], device="cpu", head_loader=fake_loader,
                         encoder_builder=EncoderSpy())
        env = {**os.environ, "WANDB_MODE": "disabled"}
        res = subprocess.run([sys.executable, str(PROJECT / "eval_aligned.py"), "mode=merge",
                              "heads=[aligned_mae_rgb,aligned_mae_ir]"], cwd=repo_root(), env=env,
                             capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, res.stderr[-2000:]
        (out,) = list((irwm_env["output"] / "detection").glob("*/*/eval_aligned.json"))
        obj = json.loads(out.read_text())
        assert obj["mode"] == "merge" and obj["msda"] is None and set(obj["cells"]) == {"aligned_mae_rgb",
                                                                                       "aligned_mae_ir"}
        assert obj["cells"] == merge_aligned(["aligned_mae_rgb", "aligned_mae_ir"])["cells"]
        assert (out.parent / "table_b.md").read_text() == render_table_b_markdown(obj)


# --------------------------------------------------------------------------- Group 5: Table B
def _metrics(v: float) -> dict:
    keys = ("AP", "AP50", "APs", "APm", "APl", "AP_person", "AP50_person", "AP_car", "AP50_car",
            "AP_person_car", "AP50_person_car")
    return {k: v for k in keys}


class TestTable:
    def test_parse(self):
        assert parse_aligned_head("aligned_dino3072_ir") == ("dino3072", "thermal")
        for bad in ("det_rgb", "aligned_vit_rgb", "aligned_mae_thermal"):
            with pytest.raises(ValueError):
                parse_aligned_head(bad)

    def test_ratios_and_markdown(self):
        cells = {"aligned_dinopca_rgb": {"rgb": _metrics(0.4), "thermal": _metrics(0.2)},
                 "aligned_dinopca_ir": {"rgb": _metrics(0.1), "thermal": _metrics(0.5)},
                 "aligned_mae_rgb": {"rgb": _metrics(0.6), "thermal": _metrics(0.3)}}
        r = transfer_ratios(cells)
        assert r["aligned_dinopca_rgb"]["thermal"]["AP_person_car"] == pytest.approx(0.4)
        assert r["aligned_dinopca_ir"]["rgb"]["AP50_person_car"] == pytest.approx(0.25)
        assert "aligned_mae_rgb" not in r  # its in-domain reference on IR (aligned_mae_ir) was not scored
        md = render_table_b_markdown({"cells": cells})
        assert "| dinopca | RGB | **0.400** | 0.200 |" in md and "| dinopca | IR | 0.100 | **0.500** |" in md
        assert "| mae | RGB | **0.600** | 0.300 |" in md
        assert "| dinopca | 0.400 | 0.500 | 0.100 |" in md  # paired H1: IR − RGB in-domain
        assert "| mae | 0.600 | – | – |" in md
        assert "`aligned_dinopca_rgb` on IR (÷ `aligned_dinopca_ir`) | 0.400 |" in md
        assert "dino3072" not in md.split("**Per cell**")[0]  # no rows for heads that were not run
        tiny = {"aligned_dinopca_rgb": {"rgb": _metrics(0.0002)}, "aligned_dinopca_ir": {"thermal": _metrics(0.0)}}
        md = render_table_b_markdown({"cells": tiny})
        assert "-0.000" not in md and "| dinopca | 0.000 | 0.000 | 0.000 |" in md


# --------------------------------------------------------------------------- Group 6: config
def test_config(irwm_env):
    with initialize_config_dir(config_dir=str(PROJECT / "conf"), version_base="1.3"):
        cfg = compose("eval_aligned")
    assert tuple(cfg.heads) == ALL_HEADS and list(cfg.test_modalities) == ["rgb", "thermal"]
    assert cfg.split == "test" and cfg.overwrite is False and cfg.verify_rescore is True and cfg.max_images is None
    assert cfg.msda == "cuda" and cfg.device == "cuda" and cfg.project == "detection" and cfg.mode == "run"
    assert "eval_aligned" in cfg.wandb.tags


def test_launcher_script():
    text = (repo_root() / "slurm" / "eval_aligned.sh").read_text()
    assert 'IRWM_STAGE_DATASETS="aligned/align"' in text and "IRWM_STAGE_DATA=1" in text
    assert "slurm/submit.sh" in text and "projects/detection/eval_aligned.py" in text and "SBATCH_EXTRA" in text
