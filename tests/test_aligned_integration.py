"""Integration tests for Phase 2b (aligned FLIR v1 + backbone ablation, F16-F19).

The per-component suites (test_aligned_data, test_backbone_dino3072, test_backbone_mae,
test_aligned_training, test_aligned_eval) test each part on its own. These tests check the seams
between parts that were written separately:

- one head list everywhere (frozen names, experiment files, launcher, eval config);
- label map, box geometry and scoring agree end to end: the reader's own boxes, pushed through the
  training dataset's encoder-frame mapping, ``to_flir_detections`` and the scoring used by training
  (``coco_map``) and by evaluation (``score_aligned``), must score AP = 1 for both input policies;
- the real chain on a local GPU: train each of the 6 heads briefly -> auto-export -> F19 eval ->
  re-score, in subprocesses exactly as the SLURM jobs run them (``-m "data and gpu"``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
CONF = REPO / "projects" / "detection" / "conf"
HEADS = [f"aligned_{k}_{m}" for k in ("dinopca", "dino3072", "mae") for m in ("rgb", "ir")]


# --------------------------------------------------------------------------- one head list
class TestHeadNames:
    def test_same_heads_everywhere(self):
        from irwm.detection.frozen import FROZEN_MODALITY, FROZEN_NAMES

        experiments = sorted(p.stem for p in (CONF / "aligned").glob("aligned_*.yaml"))
        eval_heads = yaml.safe_load((CONF / "eval_aligned.yaml").read_text())["heads"]
        launcher = (REPO / "slurm" / "train_aligned.sh").read_text()
        launcher_heads = re.search(r"^ALL_HEADS=\(([^)]*)\)", launcher, re.M)[1].split()

        assert experiments == sorted(HEADS)
        assert eval_heads == HEADS
        assert launcher_heads == HEADS
        assert set(HEADS) <= set(FROZEN_NAMES)
        for h in HEADS:
            assert FROZEN_MODALITY[h] == ("rgb" if h.endswith("_rgb") else "thermal")

    @pytest.mark.parametrize("head", HEADS)
    def test_experiment_matches_frozen_modality_and_label_space(self, head):
        from irwm.data.aligned import ALIGNED_NUM_CLASSES
        from irwm.detection.frozen import FROZEN_MODALITY

        exp = yaml.safe_load((CONF / "aligned" / f"{head}.yaml").read_text())
        assert exp["name"] == head and exp["modality"] == FROZEN_MODALITY[head]
        train = yaml.safe_load((CONF / "train_aligned.yaml").read_text())
        assert train["detector"]["num_classes"] == ALIGNED_NUM_CLASSES == 2


# --------------------------------------------------------------------------- encoder <-> head
class TestEncoderHeadContract:
    @pytest.mark.parametrize("kind,policy", [("dinopca", "native"), ("dino3072", "native")])
    def test_encoder_channels_match_detector_input(self, kind, policy):
        from irwm.features.encoder import ENCODER_CHANNELS

        exp = yaml.safe_load((CONF / "aligned" / f"aligned_{kind}_rgb.yaml").read_text())
        det = {"in_channels": 1152, "in_proj": None, **exp.get("detector", {})}
        assert det["in_channels"] == ENCODER_CHANNELS[kind]
        # Both DINO variants hand the same 1152-d map to the stem (user decision 2026-11-13).
        assert (det["in_proj"] or det["in_channels"]) == 1152

    def test_mae_policy_is_dino_incompatible_and_dino_policy_is_mae_incompatible(self):
        from irwm.features.encoder import build_frozen_encoder
        from irwm.features.preprocess import POLICY_SIZE

        assert POLICY_SIZE["native640"] == (512, 640) and all(s % 64 == 0 for s in POLICY_SIZE["native640"])
        assert POLICY_SIZE["native"] == (448, 560) and POLICY_SIZE["native"][1] % 64 != 0  # ViTDet needs %64
        with pytest.raises(ValueError, match="native640"):
            build_frozen_encoder("dino3072", policy="native640")


# --------------------------------------------------------------------------- label map + geometry + scoring
def _oracle_detections(split, policy):
    """The reader's GT, mapped to the encoder frame by the training dataset, back through scoring."""
    from irwm.data.aligned import ALIGNED_LABEL_TO_CATEGORY_ID
    from irwm.data.detection import FlirDetectionDataset
    from irwm.detection.coco_val import to_flir_detections

    ds = FlirDetectionDataset(split, policy=policy, filter_empty=False)
    dets: list[dict] = []
    for pos, (boxes, labels) in zip(ds._positions, ds._targets):
        r = split.records[pos]
        geom = ds.preprocess.geometry(r.width, r.height)
        pred = {"boxes": torch.from_numpy(boxes), "scores": torch.ones(len(boxes)),
                "labels": torch.from_numpy(labels)}
        dets += to_flir_detections([pred], [r.image_id], [geom], label_to_category_id=ALIGNED_LABEL_TO_CATEGORY_ID)
    return ds, dets


@pytest.fixture
def real_env():
    from irwm.data.aligned import aligned_root
    from irwm.utils.paths import load_env

    load_env()
    if not os.environ.get("IRWM_DATA_ROOT") or not aligned_root().is_dir():
        pytest.skip("aligned FLIR v1 not found under $IRWM_DATA_ROOT")


@pytest.mark.data
@pytest.mark.usefixtures("real_env")
class TestOracleRoundTrip:
    """Real aligned data. Perfect detections must score AP = 1 through both training and eval scoring."""

    @pytest.mark.parametrize("policy", ["native", "native640"])
    @pytest.mark.parametrize("modality", ["rgb", "thermal"])
    @pytest.mark.parametrize("split_name", ["val", "test"])
    def test_gt_as_detections_scores_one(self, policy, modality, split_name):
        from irwm.data.aligned import ALIGNED_CLASS_NAMES, AlignedSplit
        from irwm.detection.coco_val import coco_map
        from irwm.eval.aligned_table import score_aligned

        split = AlignedSplit.load(modality, split_name)
        gt = split.to_coco_gt("normalised")
        ds, dets = _oracle_detections(split, policy)
        n_gt = sum(len(r.labels) for r in split.records)
        assert len(dets) == n_gt  # aligned frames are 640x512: neither policy crops any box away

        train_side = coco_map(gt, dets, class_names=ALIGNED_CLASS_NAMES)
        eval_side = score_aligned(gt, dets)
        for key in ("AP", "AP50", "AP_person", "AP_car"):
            assert train_side[key] == pytest.approx(1.0, abs=1e-6), (key, train_side[key])
            assert eval_side[key] == train_side[key]

    def test_modalities_share_gt_boxes(self):
        from irwm.data.aligned import AlignedSplit

        a, b = (AlignedSplit.load(m, "test").to_coco_gt() for m in ("rgb", "thermal"))
        strip = lambda g: sorted((x["image_id"], x["category_id"], tuple(x["bbox"])) for x in g["annotations"])
        assert strip(a) == strip(b) and a["categories"] == b["categories"]

    def test_label_map_mixups(self):
        """What happens when the label maps are mixed up (R12 protection is NOT the category ids).

        - A 2-way aligned head scored with the *v2 default* map: labels 0/1 -> ids 1/2, which are
          exactly the aligned ids, so forgetting ``label_to_category_id`` is harmless (AP stays 1).
        - A *6-class v2* detection file (car = label 2 -> id 3) on aligned GT: cars are never matched.
          Nothing in the ids stops this; the eval's dataset checks (meta["dataset"]) do.
        """
        from irwm.data.aligned import ALIGNED_CLASS_NAMES, AlignedSplit
        from irwm.data.classes import CATEGORY_ID_TO_LABEL
        from irwm.detection.coco_val import coco_map, to_flir_detections

        split = AlignedSplit.load("thermal", "val")
        gt = split.to_coco_gt()
        ds, _ = _oracle_detections(split, "native")
        preds, ids, geoms = [], [], []
        for pos, (boxes, labels) in zip(ds._positions, ds._targets):
            r = split.records[pos]
            preds.append({"boxes": torch.from_numpy(boxes), "scores": torch.ones(len(boxes)),
                          "labels": torch.from_numpy(labels)})
            ids.append(r.image_id)
            geoms.append(ds.preprocess.geometry(r.width, r.height))

        forgot_map = to_flir_detections(preds, ids, geoms)  # v2 default map on 2-way labels
        assert coco_map(gt, forgot_map, class_names=ALIGNED_CLASS_NAMES)["AP"] == pytest.approx(1.0)

        v2_car = CATEGORY_ID_TO_LABEL[3]  # what a v2 head calls "car"
        v2_preds = [{**p, "labels": torch.where(p["labels"] == 1, v2_car, 0)} for p in preds]
        v2_file = to_flir_detections(v2_preds, ids, geoms)
        m = coco_map(gt, v2_file, class_names=ALIGNED_CLASS_NAMES)
        assert m["AP_person"] == pytest.approx(1.0) and m["AP_car"] == 0.0


# --------------------------------------------------------------------------- real chain on a local GPU
def _ckpt_root_with_weights(tmp: Path) -> Path:
    """A fresh IRWM_CKPT_ROOT that links the real encoder / MAE weights but has an empty detection/."""
    src = Path(os.environ.get("IRWM_INTEGRATION_WEIGHTS", REPO / "checkpoints"))
    root = tmp / "ckpt"
    root.mkdir()
    for name in ("torch_hub", "dino_foresight", "pretrained"):
        if not (src / name).exists():
            pytest.skip(f"no {src / name}; set IRWM_INTEGRATION_WEIGHTS to a checkpoint root with the weights")
        (root / name).symlink_to(src / name)
    return root


def _run(args: list[str], env: dict, cwd: Path, timeout: int = 1800) -> subprocess.CompletedProcess:
    res = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise AssertionError(f"{' '.join(args)} failed ({res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-6000:]}")
    return res


@pytest.mark.data
@pytest.mark.gpu
def test_train_export_eval_chain(tmp_path):
    """6 heads x (1 short epoch -> auto-export) -> F19 eval on the first test pairs -> re-score -> merge."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    env = {**os.environ, "IRWM_CKPT_ROOT": str(_ckpt_root_with_weights(tmp_path)),
           "IRWM_OUTPUT_ROOT": str(tmp_path / "out"), "IRWM_CACHE_ROOT": str(tmp_path / "cache"),
           "WANDB_MODE": "disabled", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    env.pop("SLURM_JOB_ID", None)
    small_gpu = torch.cuda.get_device_properties(0).total_memory < 12 * 2**30
    bs = 1 if small_gpu else 4

    for i, head in enumerate(HEADS):
        _run(["projects/detection/train_aligned.py", f"aligned={head}", "msda=auto", f"run_name=itest-{i}",
              "trainer.max_epochs=1", "+trainer.limit_train_batches=3", "+trainer.limit_val_batches=4",
              f"data.batch_size={bs}", "data.num_workers=2", "cluster.num_workers=2"], env, REPO)
        frozen = Path(env["IRWM_CKPT_ROOT"]) / "detection" / "frozen" / head
        meta = json.loads((frozen / "meta.json").read_text())
        assert meta["dataset"] == "aligned_v1" and (frozen / "head.pt").is_file()
        summary = sorted((tmp_path / "out").rglob("train_summary.json"))[-1]
        s = json.loads(summary.read_text())
        assert s["fit_complete"] is True and s["export"]["path"] and not s["export"].get("error")
        # The training checkpoints (MAE: full model + AdamW state, ~1.3 GB each) are not needed after export.
        shutil.rmtree(Path(env["IRWM_CKPT_ROOT"]) / "detection" / f"itest-{i}", ignore_errors=True)

    res = _run(["projects/detection/eval_aligned.py", "msda=auto", "max_images=12", f"batch_size={bs}",
                "num_workers=2", "cluster.num_workers=2"], env, REPO)
    out = sorted((tmp_path / "out").rglob("eval_aligned.json"))[-1]
    result = json.loads(out.read_text())
    cells = result["cells"] if "cells" in result else result["metrics"]
    assert sorted(cells) == sorted(HEADS)
    for head in HEADS:
        assert sorted(cells[head]) == ["rgb", "thermal"]
        for m in cells[head].values():
            assert 0.0 <= m["AP"] <= 1.0 and np.isfinite(m["AP50"])
    table = out.with_name("table_b.md").read_text()
    assert all(h.removeprefix("aligned_") in table or h in table for h in HEADS)
    assert "Traceback" not in res.stderr

    # R12: the v2 heads are refused by the aligned evaluation, before any model is loaded.
    bad = subprocess.run([sys.executable, "projects/detection/eval_aligned.py", "heads=[det_ir]", "msda=auto",
                          "max_images=2"], cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert bad.returncode != 0


# --------------------------------------------------------------------------- parallel SLURM jobs
def test_parallel_slurm_jobs_get_distinct_run_dirs(irwm_env, monkeypatch):
    """Two jobs starting in the same second must not share a Hydra run dir (2026-11-13 smoke collision)."""
    from hydra import compose, initialize_config_dir

    dirs = []
    for job in ("5531159", "5531160"):
        monkeypatch.setenv("SLURM_JOB_ID", job)
        with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
            cfg = compose("train_aligned", overrides=["aligned=aligned_dino3072_rgb", "cluster=ctcomp3"],
                          return_hydra_config=True)
        dirs.append(str(cfg.hydra.run.dir).rsplit("/", 1)[-1])
    assert dirs[0] != dirs[1] and dirs[0].endswith("slurm-5531159") and dirs[1].endswith("slurm-5531160")
