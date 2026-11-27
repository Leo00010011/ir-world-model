"""KAIST-Paired detection (2026-11-27): readers and splits, COCO GT with ignore regions and occlusion
subsets, the KAIST data module (val + fixed train subset), DetectorModule extra evaluations, the
train_kaist configs and the SLURM launcher.

CPU tests use a tiny synthetic KAIST tree (640×512 frames, CRLF image lists, 3-channel lwir JPEGs),
a stub frozen encoder and a reduced DINO head; validation runs a real CPU ``Trainer.validate``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import nn

from irwm.data.detection import KaistDetectionDataModule
from irwm.data.kaist import (
    KAIST_CLASS_NAMES,
    KAIST_LABEL_TO_CATEGORY_ID,
    OCCLUSION_LEVELS,
    KaistSplit,
    image_id_of,
    parse_bbgt,
    split_frames,
    train_eval_ids,
)
from irwm.detection.coco_val import coco_map, fp_metrics
from irwm.utils.paths import repo_root

REPO = repo_root()
PROJECT = REPO / "projects" / "detection"
HEADER = "% bbGt version=3\n"


def _line(label, x, y, w, h, occ=0):
    return f"{label} {x} {y} {w} {h} {occ} 0 0 0 0 0 0\n"


# (set, video) → {frame: lwir lines}; the visible camera gets the same boxes shifted by +2 px in x.
FRAMES = {
    ("set00", "V000"): {3: [_line("person", 10, 20, 30, 60), _line("people", 300, 200, 50, 40)],
                        7: [],  # negative frame
                        11: [_line("cyclist", 100, 100, 40, 80), _line("person", 400, 150, 30, 70, occ=1)],
                        15: [_line("person", 200, 100, 30, 60)], 19: [_line("person", 50, 50, 30, 60)]},
    ("set03", "V000"): {3: [_line("person", 600, 300, 60, 100, occ=2)],  # clipped at x = 640
                        7: [_line("person?", 10, 10, 20, 40), _line("person", 700, 10, 20, 20)]},  # 2nd: off frame
    ("set02", "V000"): {3: [_line("person", 50, 60, 40, 100), _line("person", 300, 200, 30, 70, occ=1),
                            _line("person", 500, 100, 30, 70, occ=2), _line("cyclist", 150, 150, 40, 80)],
                        7: [_line("unpaired", 10, 10, 20, 40)], 11: [], 15: [_line("person", 80, 90, 30, 60)]},
    ("set05", "V000"): {3: [_line("person", 120, 130, 30, 60)]},
    ("set06", "V000"): {19: [_line("person", 120, 130, 30, 60)]},
}


@pytest.fixture
def tiny_kaist(irwm_env) -> dict:
    root = irwm_env["data"] / "kaist-cvpr15"
    paired = irwm_env["data"] / "kaist-paired" / "annotations"
    (root / "imageSets").mkdir(parents=True)
    train, test = [], []
    rng = np.random.default_rng(0)
    for (s, v), frames in FRAMES.items():
        for f, lines in frames.items():
            name = f"I{f:05d}"
            (test if s == "set06" else train).append(f"{s}/{v}/{name}")
            for cam, dx in (("lwir", 0), ("visible", 2)):
                d = paired / s / v / cam
                d.mkdir(parents=True, exist_ok=True)
                shifted = []
                for ln in lines:
                    lab, x, *rest = ln.split()
                    shifted.append(" ".join([lab, str(int(x) + dx), *rest]) + "\n")
                (d / f"{name}.txt").write_text(HEADER + "".join(shifted))
                img = root / "images" / s / v / cam
                img.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (512, 640, 3), dtype=np.uint8)).save(img / f"{name}.jpg")
    # CRLF, like the real KAIST lists
    (root / "imageSets" / "train-all-04.txt").write_bytes("".join(f"{x}\r\n" for x in train).encode())
    (root / "imageSets" / "test-all-20.txt").write_bytes("".join(f"{x}\r\n" for x in test).encode())
    return {"root": root, "paired": paired}


def _split(t, modality="thermal", split="train") -> KaistSplit:
    return KaistSplit.load(modality, split, t["root"], t["paired"])


def _dm(t, **kw) -> KaistDetectionDataModule:
    args = dict(modality="thermal", policy="native", batch_size=2, flip_p=0.0, root=t["root"],
                paired_root=t["paired"])
    args.update(kw)
    dm = KaistDetectionDataModule(**args)
    dm.setup("fit")
    return dm


def _perfect(gt: dict) -> list[dict]:
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
            for a in gt["annotations"] if not a["iscrowd"]]


# --------------------------------------------------------------------------- reader
class TestParse:
    def test_bbgt(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text(HEADER + _line("person", 10, 20, 30, 40, occ=2) + _line("people", 1, 2, 3, 4)
                     + _line("cyclist", 630, 500, 20, 20) + _line("person", -50, 0, 20, 20) + "\n")
        boxes, labels, occ, ignore, bad = parse_bbgt(p)
        assert boxes.tolist() == [[10, 20, 30, 40], [630, 500, 10, 12]]  # clipped to 640×512
        assert labels.tolist() == [0, 1] and occ.tolist() == [2, 0]
        assert ignore.tolist() == [[1, 2, 3, 4]] and bad == 1  # the off-frame box

    def test_negative_and_errors(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text(HEADER)
        boxes, labels, occ, ignore, bad = parse_bbgt(p)
        assert boxes.shape == (0, 4) and labels.shape == (0,) and ignore.shape == (0, 4) and bad == 0
        p.write_text(HEADER + _line("dog", 1, 2, 3, 4))
        with pytest.raises(ValueError, match="unknown class"):
            parse_bbgt(p)
        p.write_text("person 1 2 3 4 0\n")
        with pytest.raises(ValueError, match="header"):
            parse_bbgt(p)
        with pytest.raises(FileNotFoundError):
            parse_bbgt(tmp_path / "missing.txt")

    def test_image_id(self):
        assert image_id_of("set02", "V001", "I00144") == 20_100_144
        assert image_id_of("set11", "V008", "I04400") == 110_804_400


class TestSplits:
    def test_sets_disjoint(self, tiny_kaist):
        train = {f[0] for f in split_frames("train", tiny_kaist["root"])}
        val = {f[0] for f in split_frames("val", tiny_kaist["root"])}
        test = {f[0] for f in split_frames("test", tiny_kaist["root"])}
        assert train == {"set00", "set03"} and val == {"set02", "set05"} and test == {"set06"}

    def test_records(self, tiny_kaist):
        tr, va = _split(tiny_kaist), _split(tiny_kaist, split="val")
        assert not set(tr.image_ids()) & set(va.image_ids())
        assert len(tr) == 7 and len(va) == 5 and tr.n_degenerate == 1
        r = tr.by_image_id(image_id_of("set00", "V000", "I00003"))
        assert r.file_name == "images/set00/V000/lwir/I00003.jpg" and r.path.is_file()
        assert r.boxes.tolist() == [[10, 20, 30, 60]] and r.ignore_boxes.tolist() == [[300, 200, 50, 40]]
        assert r.time_of_day == "day" and tr.by_image_id(image_id_of("set03", "V000", "I00003")).time_of_day == "night"
        assert tr.videos() == ["set00/V000", "set03/V000"]

    def test_camera(self, tiny_kaist):
        rid = image_id_of("set00", "V000", "I00003")
        rgb = _split(tiny_kaist, "rgb").by_image_id(rid)
        assert rgb.file_name == "images/set00/V000/visible/I00003.jpg"
        assert rgb.boxes.tolist() == [[12, 20, 30, 60]]  # the visible camera's own boxes

    def test_image_dataset(self, tiny_kaist):
        th = _split(tiny_kaist).image_dataset()[0]
        rgb = _split(tiny_kaist, "rgb").image_dataset()[0]
        assert th["image"].shape == (512, 640) and th["image"].dtype == np.uint8  # 3-ch lwir → L
        assert rgb["image"].shape == (512, 640, 3)

    def test_train_eval_ids(self, tiny_kaist):
        tr = _split(tiny_kaist)
        ids = train_eval_ids(tr, 2)
        assert ids == sorted([image_id_of("set00", "V000", f"I{f:05d}") for f in (3, 11, 19)]
                             + [image_id_of("set03", "V000", "I00003")])
        assert train_eval_ids(tr, 1) == list(tr.image_ids())
        with pytest.raises(ValueError):
            train_eval_ids(tr, 0)


# --------------------------------------------------------------------------- COCO GT
class TestCocoGt:
    def test_ignore_as_crowd(self, tiny_kaist):
        gt = _split(tiny_kaist).to_coco_gt()
        assert [c["name"] for c in gt["categories"]] == ["person", "cyclist"]
        crowd = [a for a in gt["annotations"] if a["iscrowd"]]
        assert len(crowd) == 2 * 2  # people + person?, once per category
        ids = [a["id"] for a in gt["annotations"]]
        assert len(ids) == len(set(ids))

    def test_perfect_and_ignored_detections(self, tiny_kaist):
        va = _split(tiny_kaist, split="val")
        gt = va.to_coco_gt()
        dets = _perfect(gt)
        assert coco_map(gt, dets, class_names=KAIST_CLASS_NAMES)["AP"] == pytest.approx(1.0)
        # a detection on the 'unpaired' ignore box is neither TP nor FP
        unpaired = image_id_of("set02", "V000", "I00007")
        extra = dets + [{"image_id": unpaired, "category_id": 1, "bbox": [10, 10, 20, 40], "score": 0.9}]
        assert coco_map(gt, extra, class_names=KAIST_CLASS_NAMES)["AP"] == pytest.approx(1.0)

    def test_occlusion_subsets(self, tiny_kaist):
        va = _split(tiny_kaist, split="val")
        full = va.to_coco_gt()
        for name, levels in OCCLUSION_LEVELS.items():
            gt = va.to_coco_gt(occlusion=levels)
            kept = [a for a in gt["annotations"] if not a["iscrowd"]]
            assert {a["category_id"] for a in kept} == {1}
            n = {"occ_partial": 1, "occ_heavy": 1, "occ_any": 2}[name]
            assert len(kept) == n
            # detecting every box (occluded or not, cyclist included) costs nothing on the subset
            m = coco_map(gt, _perfect(full), class_names=KAIST_CLASS_NAMES)
            assert m["AP"] == pytest.approx(1.0) and math.isnan(m["AP_cyclist"])
        # missing the heavy box drops occ_heavy to 0 and leaves occ_partial at 1
        heavy = [a for a in full["annotations"] if a["bbox"] == [500.0, 100.0, 30.0, 70.0]]
        dets = [d for d in _perfect(full) if d["bbox"] != heavy[0]["bbox"]]
        assert coco_map(va.to_coco_gt(occlusion=(2,)), dets, class_names=KAIST_CLASS_NAMES)["AP"] == 0.0
        assert coco_map(va.to_coco_gt(occlusion=(1,)), dets, class_names=KAIST_CLASS_NAMES)["AP"] == pytest.approx(1.0)


class TestFpMetrics:
    def test_rates(self):
        cats = {"person": 1, "cyclist": 2}
        dets = [{"image_id": 1, "category_id": 1, "score": 0.9}, {"image_id": 1, "category_id": 2, "score": 0.4},
                {"image_id": 2, "category_id": 1, "score": 0.35}, {"image_id": 9, "category_id": 1, "score": 0.99}]
        m = fp_metrics(dets, [1, 2, 3, 4], cats, thresholds=(0.3, 0.5))
        assert m["n_images"] == 4
        assert m["FPPI@0.3"] == 3 / 4 and m["frac_fp@0.3"] == 2 / 4  # image 9 is not an empty frame
        assert m["FPPI@0.5"] == 1 / 4 and m["frac_fp@0.5"] == 1 / 4
        assert m["FPPI@0.3_person"] == 2 / 4 and m["FPPI@0.3_cyclist"] == 1 / 4 and m["frac_fp@0.5_cyclist"] == 0.0
        assert math.isnan(fp_metrics(dets, [], cats)["FPPI@0.5"])


# --------------------------------------------------------------------------- data module
class TestDataModule:
    def test_sets(self, tiny_kaist):
        dm = _dm(tiny_kaist, val_stride=2, train_eval_stride=2, empty_ratio=0)
        assert len(dm.train_set) == 5 and dm.train_set.n_dropped_empty == 2  # negative + ignore-only frame
        assert dm.train_sampler is None and dm.train_epoch_size == 5
        assert len(dm.val_set) == 3  # set02 frames 3, 11 + set05 frame 3
        assert len(dm.train_eval_set) == 4 and dm.train_eval_set.flip_p == 0.0
        assert sorted(dm.val_variants) == sorted(OCCLUSION_LEVELS)
        assert sorted(i["id"] for i in dm.val_gt["images"]) == sorted(dm.val_set._image_ids())
        assert len(dm.val_dataloader()) == 2
        (e,) = dm.extra_eval()
        assert e["prefix"] == "train_eval" and e["gt"] is dm.train_eval_gt

    def test_frame_subsets(self, tiny_kaist):
        dm = _dm(tiny_kaist, train_eval_stride=1)
        sid = lambda s, f: image_id_of(s, "V000", f"I{f:05d}")  # noqa: E731
        # val: set02 3, 15 + set05 3 have boxes; set02 11 is empty; set02 7 has only an 'unpaired' ignore box
        assert sorted(dm.val_subsets["nonempty"]["image_ids"]) == sorted([sid("set02", 3), sid("set02", 15),
                                                                            sid("set05", 3)])
        assert dm.val_negatives["empty"] == [sid("set02", 11)]
        assert sorted(i["id"] for i in dm.val_subsets["nonempty"]["gt"]["images"]) == \
            sorted(dm.val_subsets["nonempty"]["image_ids"])
        assert dm.train_eval_negatives["empty"] == [sid("set00", 7)]  # set03 7 is ignore-only: in neither
        assert len(dm.train_eval_subsets["nonempty"]["image_ids"]) == 5
        (e,) = dm.extra_eval()
        assert e["subsets"] is dm.train_eval_subsets and e["negatives"] is dm.train_eval_negatives

    def test_no_train_eval(self, tiny_kaist):
        dm = _dm(tiny_kaist)
        assert dm.train_eval_set is None and dm.extra_eval() == []
        assert len(dm.val_set) == 5
        assert not isinstance(dm.val_dataloader(), list)

    def test_empty_ratio(self, tiny_kaist):
        neg = image_id_of("set00", "V000", "I00007")  # no box at all
        ign = image_id_of("set03", "V000", "I00007")  # only an ignore box (person?) → never a negative
        dm = _dm(tiny_kaist, empty_ratio=0.25)
        assert (dm.n_positive, dm.n_empty_pool, dm.n_ignore_only, dm.n_empty_per_epoch) == (5, 1, 1, 1)
        ids = dm.train_set._image_ids()
        assert neg in ids and ign not in ids and len(dm.train_set) == 6 == dm.train_epoch_size
        loader = dm.train_dataloader()
        assert loader.sampler is dm.train_sampler and len(list(dm.train_sampler)) == 6
        all_ = _dm(tiny_kaist, empty_ratio="all")
        assert all_.n_empty_per_epoch == 1
        with pytest.raises(ValueError, match="empty_ratio"):
            _dm(tiny_kaist, empty_ratio=-1)

    def test_smoke_mode(self, tiny_kaist):
        ids = [image_id_of("set00", "V000", "I00003"), image_id_of("set00", "V000", "I00011")]
        dm = _dm(tiny_kaist, train_image_ids=ids, val_split="train", val_image_ids=ids, train_eval_stride=2)
        assert dm.val_set._image_ids() == ids and set(dm.train_eval_set._image_ids()) <= set(ids)


class TestEmptyMixSampler:
    def test_draw(self):
        from irwm.data.detection import EmptyMixSampler, n_empty_per_epoch

        pos, emp = list(range(10)), list(range(10, 50))
        s = EmptyMixSampler(pos, emp, n_empty=8, seed=3)
        draws = []
        for epoch in range(3):
            s.set_epoch(epoch)
            idx = list(s)
            assert len(idx) == len(s) == 18 and len(set(idx)) == 18
            assert set(pos) <= set(idx)  # every positive, every epoch
            draws.append(frozenset(idx) - set(pos))
        assert len(set(draws)) == 3  # a fresh empty draw per epoch
        s.set_epoch(1)
        assert frozenset(s) - set(pos) == draws[1]  # (seed, epoch) → same frames, e.g. after a resume
        assert n_empty_per_epoch(0.25, 3397, 6000) == 849 and n_empty_per_epoch("all", 3397, 6000) == 6000
        assert n_empty_per_epoch(5.0, 10, 7) == 7 and n_empty_per_epoch(0, 10, 7) == 0
        with pytest.raises(ValueError):
            EmptyMixSampler(pos, emp, n_empty=41)


# --------------------------------------------------------------------------- DetectorModule
class StubEncoder(nn.Module):
    """Frozen-encoder interface: avg-pool 14 → fixed ``Linear(3, C)`` → ``[B, C, H/14, W/14]``."""

    def __init__(self, channels: int = 1152) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(3, channels)
        self.requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.proj(F.avg_pool2d(x, 14).permute(0, 2, 3, 1)).float().permute(0, 3, 1, 2).contiguous()


REDUCED = {"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1,
           "transformer.decoder.num_layers": 2}
DET = {"in_channels": 1152, "in_proj": None, "stem_dim": 768, "num_classes": 2, "overrides": REDUCED}
ENC = {"policy": "native", "kind": "dinopca", "precision": "fp16", "hub_version": "dinof_release", "attention": "sdpa"}
OPTIM = {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 1e-4, "backbone_lr_factor": 1.0, "lr_drop_epoch": 15}


class TestDetectorModule:
    def test_score_eval_sets(self, tiny_kaist):
        from irwm.detection.lightning import DetectorModule

        dm = _dm(tiny_kaist, train_eval_stride=2)
        ns = SimpleNamespace(val_gt=dm.val_gt, val_variants=dm.val_variants, extra_eval=dm.extra_eval(),
                             val_subsets=dm.val_subsets, val_negatives=dm.val_negatives,
                             fp_thresholds=(0.3, 0.5), class_names=list(KAIST_CLASS_NAMES))

        def by_image(gt):
            out: dict[int, list[dict]] = {}
            for d in _perfect(gt):
                out.setdefault(d["image_id"], []).append(d)
            return out

        val_dets = {i: [{**d, "score": 0.5} for d in ds] for i, ds in by_image(dm.val_gt).items()}
        empty_id = dm.val_negatives["empty"][0]
        val_dets[empty_id] = [{"image_id": empty_id, "category_id": 1, "bbox": [5, 5, 20, 40], "score": 0.6}]
        scored = DetectorModule.score_eval_sets(ns, [val_dets, by_image(dm.train_eval_gt)])
        assert set(scored) == {"val", "train_eval", "val_nonempty", "val_empty", "train_eval_nonempty",
                               "train_eval_empty"}
        # the false positive on the empty frame costs AP on all frames, not on the non-empty ones
        assert scored["val"]["AP"] < 1.0 and scored["val_nonempty"]["AP"] == pytest.approx(1.0)
        assert scored["val_empty"]["FPPI@0.5"] == 1.0 and scored["val_empty"]["frac_fp@0.3_person"] == 1.0
        assert scored["val_empty"]["FPPI@0.5_cyclist"] == 0.0 and scored["val_empty"]["n_images"] == 1
        assert scored["val_nonempty"]["n_images"] == 3 and "AP50_cyclist" in scored["val_nonempty"]
        for prefix in ("val", "train_eval"):
            m = scored[prefix]
            for k in ("AP", "AP50", "AP_person", "AP50_person", "AP_cyclist", "AP50_cyclist",
                      "AP_occ_partial", "AP50_occ_partial", "AP_occ_heavy", "AP50_occ_heavy",
                      "AP_occ_any", "AP50_occ_any"):
                assert k in m, (prefix, k)
        assert scored["train_eval"]["AP"] == pytest.approx(1.0)
        # occlusion metrics stay on all frames, so the empty-frame false positive costs them too
        assert scored["val"]["AP_occ_heavy"] < 1.0
        # a subset with no GT box logs nan, not COCOeval's -1
        empty = SimpleNamespace(**{**ns.__dict__, "extra_eval": [], "val_subsets": {}, "val_negatives": {},
                                   "val_variants": {"empty": {**dm.val_gt, "annotations": []}}})
        m = DetectorModule.score_eval_sets(empty, [by_image(dm.val_gt)])["val"]
        assert math.isnan(m["AP_empty"]) and math.isnan(m["AP50_empty"])

    def test_reserved_prefix(self):
        from irwm.detection.lightning import DetectorModule

        with pytest.raises(ValueError, match="reserved"):
            DetectorModule(detector=DET, encoder=ENC, optim=OPTIM, encoder_module=StubEncoder(),
                           extra_eval=[{"prefix": "val", "gt": {}}])

    def test_validate_two_loaders(self, tiny_kaist, tmp_path):
        import lightning as L
        from lightning.pytorch.loggers import CSVLogger

        from irwm.detection.detrex_adapter import select_msda
        from irwm.detection.lightning import DetectorModule

        select_msda("auto")
        dm = _dm(tiny_kaist, val_stride=2, train_eval_stride=2)
        m = DetectorModule(detector=DET, encoder=ENC, optim=OPTIM, encoder_module=StubEncoder(),
                           val_gt=dm.val_gt, class_names=list(KAIST_CLASS_NAMES),
                           label_to_category_id=dict(KAIST_LABEL_TO_CATEGORY_ID),
                           val_variants=dm.val_variants, extra_eval=dm.extra_eval(),
                           val_subsets=dm.val_subsets, val_negatives=dm.val_negatives)
        m.detector.detr.select_box_nums_for_evaluation = 100
        m.metrics_dir = tmp_path / "val_metrics"
        trainer = L.Trainer(accelerator="cpu", devices=1, logger=CSVLogger(tmp_path, name="csv"),
                            enable_progress_bar=False, enable_model_summary=False)
        trainer.validate(m, datamodule=dm)
        keys = set(trainer.callback_metrics)
        for prefix in ("val", "train_eval"):
            assert {f"{prefix}/AP", f"{prefix}/AP50", f"{prefix}/AP_person", f"{prefix}/AP50_cyclist",
                    f"{prefix}/AP_occ_partial", f"{prefix}/AP50_occ_heavy", f"{prefix}/AP_occ_any"} <= keys
            assert {f"{prefix}_nonempty/AP", f"{prefix}_nonempty/AP50_person", f"{prefix}_empty/FPPI@0.3",
                    f"{prefix}_empty/frac_fp@0.5", f"{prefix}_empty/FPPI@0.5_cyclist"} <= keys
        assert not any("dataloader_idx" in k for k in keys)
        rec = json.loads((tmp_path / "val_metrics" / "epoch_000.json").read_text())
        assert set(rec["eval"]) == {"val", "train_eval", "val_nonempty", "val_empty", "train_eval_nonempty",
                                    "train_eval_empty"} and rec["metrics"] == rec["eval"]["val"]


# --------------------------------------------------------------------------- configs + launcher
def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=str(PROJECT / "conf"), version_base="1.3"):
        return compose("train_kaist", overrides=overrides)


def _entry():
    spec = importlib.util.spec_from_file_location("detection_train_kaist", PROJECT / "train_kaist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestConfigs:
    @pytest.mark.parametrize("head,modality", [("kaist_dinopca_rgb", "rgb"), ("kaist_dinopca_ir", "thermal")])
    def test_heads(self, irwm_env, head, modality):
        cfg = _compose([f"kaist={head}"])
        _entry().check_config(cfg)
        assert cfg.name == head and cfg.modality == modality and cfg.dataset == "kaist"
        assert cfg.trainer.max_epochs == 20 and cfg.optim.lr_drop_epoch == 15
        assert cfg.trainer.check_val_every_n_epoch == 2 and cfg.detector.num_classes == 2
        assert cfg.encoder.kind == "dinopca" and cfg.encoder.policy == "native" and cfg.encoder.attention == "sdpa"
        assert cfg.data.empty_ratio == 0.25 and list(cfg.kaist.fp_thresholds) == [0.3, 0.5]

    def test_bad_config(self, irwm_env):
        check = _entry().check_config
        with pytest.raises(ValueError, match="num_classes"):
            check(_compose(["kaist=kaist_dinopca_ir", "detector.num_classes=1"]))
        with pytest.raises(ValueError, match="modality"):
            check(_compose(["kaist=kaist_dinopca_ir", "modality=rgb"]))
        with pytest.raises(ValueError, match="empty_ratio"):
            check(_compose(["kaist=kaist_dinopca_ir", "data.empty_ratio=some"]))
        with pytest.raises(ValueError, match="not a KAIST head"):
            check(_compose(["kaist=kaist_dinopca_ir", "name=aligned_dinopca_ir"]))

    def test_frozen_names(self, irwm_env):
        from irwm.detection.frozen import FROZEN_DATASET, FROZEN_MODALITY, frozen_dir

        assert FROZEN_DATASET["kaist_dinopca_ir"] == "kaist" and FROZEN_MODALITY["kaist_dinopca_ir"] == "thermal"
        assert FROZEN_DATASET["aligned_dinopca_ir"] == "aligned_v1" and FROZEN_DATASET["det_rgb"] == "flir_v2"
        assert frozen_dir("kaist_dinopca_rgb").name == "kaist_dinopca_rgb"


class TestLauncher:
    def test_dry_run(self):
        out = subprocess.run(["bash", "slurm/train_kaist.sh", "ctcomp3", "--dry-run"], cwd=REPO,
                             capture_output=True, text=True, env={**os.environ}, check=True).stdout
        lines = [ln for ln in out.splitlines() if ln.startswith("[dry-run")]
        assert len(lines) == 2
        rgb, ir = lines
        assert "kaist=kaist_dinopca_rgb" in rgb and "IRWM_STAGE_KAIST_CAMERAS=visible" in rgb
        assert "kaist=kaist_dinopca_ir" in ir and "IRWM_STAGE_KAIST_CAMERAS=lwir" in ir
        for ln in lines:
            assert "IRWM_STAGE_DATA=1" in ln and "IRWM_STAGE_KAIST_LIST=train-all-04.txt" in ln
            assert "kaist-cvpr15" in ln and "kaist-paired" in ln and "--time=270" in ln


@pytest.mark.data
class TestRealData:
    def test_splits(self):
        from irwm.utils.paths import load_env

        load_env()
        for modality in ("thermal", "rgb"):
            tr, va = KaistSplit.load(modality, "train"), KaistSplit.load(modality, "val")
            assert len(tr) == 9843 and len(va) == 2695
            assert not set(tr.image_ids()) & set(va.image_ids())
            assert {r.set_name for r in va} == {"set02", "set05"}
            for occ in (None, *OCCLUSION_LEVELS.values()):
                gt = va.to_coco_gt(occlusion=occ)
                assert coco_map(gt, _perfect(va.to_coco_gt()), class_names=KAIST_CLASS_NAMES)["AP"] == pytest.approx(1.0)
