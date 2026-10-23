"""F7 · DINO-DETR head port. See spec/2026-10-23-dino-detr-head-port/validation.md.

Groups 1-6 run on CPU without data or weights. Groups 7-8 are ``gpu`` tests (ctcomp3, plan.md Step 9).
"""

from __future__ import annotations

import importlib
import inspect
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from detectron2.config import instantiate
from detectron2.layers import ShapeSpec
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torchvision.ops import box_iou

from irwm.detection import DINODetector, FeatureDINO, FeatureStem, build_dino_config, cuda_ops_available
from irwm.detection import detrex_adapter
from irwm.detection.detrex_adapter import (DETECTRON2_COMMIT, DETREX_COMMIT, detrex_root, ensure_detrex_projects,
                                           load_dino_vitdet_config)
from irwm.utils.paths import repo_root

REPO = repo_root()
DETREX = REPO / "third_party" / "detrex"
CONF = REPO / "lib" / "irwm" / "conf"
# plan.md Step 7: reduced config for fast CPU tests, on an 8x16 grid (680 tokens >= 100 queries).
REDUCED = {"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1,
           "transformer.decoder.num_layers": 2}


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def default_det() -> DINODetector:
    return DINODetector().eval()


@pytest.fixture(scope="module")
def reduced_det() -> DINODetector:
    torch.manual_seed(0)
    return DINODetector(overrides=REDUCED).eval()


@pytest.fixture(scope="module")
def eval_runs(default_det):
    """Full default config, eval, on 32x64 and 32x40 inputs, with the neck outputs captured."""
    captured: list = []
    hook = default_det.detr.neck.register_forward_hook(lambda m, i, o: captured.append([t.shape for t in o]))
    torch.manual_seed(1)
    runs = {}
    try:
        with torch.no_grad():
            for grid in [(32, 64), (32, 40)]:
                feats = torch.randn(1, 1152, *grid)
                runs[grid] = {"feats": feats, "out": default_det(feats), "neck": captured[-1]}
            runs["explicit"] = default_det(runs[(32, 40)]["feats"], image_size=(448, 560))
    finally:
        hook.remove()
    return runs


# --------------------------------------------------------------------------- Group 1
class TestSubmoduleAndBuild:
    def test_pinned_commits(self):
        status = _git("submodule", "status", "third_party/detrex")
        assert not status.startswith(("+", "-", "U")), status
        assert status.strip().startswith(DETREX_COMMIT), status
        nested = _git("submodule", "status", "detectron2", cwd=DETREX)
        assert not nested.startswith(("+", "-", "U")), nested
        assert nested.strip().startswith(DETECTRON2_COMMIT), nested

    def test_trees_clean(self):
        assert _git("status", "--porcelain", cwd=DETREX) == ""
        assert _git("status", "--porcelain", cwd=DETREX / "detectron2") == ""

    def test_imports(self):
        import detectron2
        import detrex  # noqa: F401
        import pkg_resources  # noqa: F401
        from detrex import _C  # noqa: F401

        assert detectron2.__version__ == "0.6"


# --------------------------------------------------------------------------- Group 2
class TestDetrexAdapter:
    def test_root_and_commits(self):
        assert detrex_root() == repo_root() / "third_party" / "detrex"
        assert _git("rev-parse", "HEAD", cwd=DETREX).strip() == DETREX_COMMIT
        assert _git("rev-parse", "HEAD", cwd=DETREX / "detectron2").strip() == DETECTRON2_COMMIT

    def test_missing_submodule(self, tmp_path, monkeypatch):
        monkeypatch.setattr(detrex_adapter, "repo_root", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="git submodule update --init --recursive"):
            detrex_root()

    def test_sys_path_appended_once(self):
        ensure_detrex_projects()
        ensure_detrex_projects()
        root = str(detrex_root())
        assert sys.path.count(root) == 1
        assert sys.path.index(root) != 0

    def test_resolved_locations(self):
        import detrex

        mod = importlib.import_module("projects.dino.modeling")
        assert Path(mod.__file__).resolve().is_relative_to(DETREX / "projects" / "dino" / "modeling")
        assert Path(detrex.__file__).resolve().is_relative_to(DETREX / "detrex")

    def test_config_from_detrex_file(self):
        cfg = load_dino_vitdet_config()
        assert cfg.model.num_queries == 900
        assert cfg.model.backbone.square_pad == 1024

    def test_cuda_ops_unavailable_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert cuda_ops_available() is False


# --------------------------------------------------------------------------- Group 3
class TestFeatureStem:
    def test_shapes(self):
        assert FeatureStem()(torch.randn(2, 1152, 32, 64))["last_feat"].shape == (2, 768, 32, 64)
        assert FeatureStem(in_channels=3072)(torch.randn(1, 3072, 32, 40))["last_feat"].shape == (1, 768, 32, 40)

    def test_output_shape(self):
        assert FeatureStem().output_shape() == {"last_feat": ShapeSpec(channels=768, stride=16)}

    def test_manual_formula(self):
        stem = FeatureStem()
        torch.nn.init.normal_(stem.norm.weight)
        torch.nn.init.normal_(stem.norm.bias)
        x = torch.randn(2, 1152, 4, 6)
        ref = F.linear(F.layer_norm(x.permute(0, 2, 3, 1), (1152,), stem.norm.weight, stem.norm.bias, 1e-6),
                       stem.proj.weight, stem.proj.bias).permute(0, 3, 1, 2)
        with torch.no_grad():
            assert torch.equal(stem(x)["last_feat"], ref)

    @pytest.mark.parametrize("shape", [(2, 1024, 32, 64), (1152, 32, 64)])
    def test_bad_input(self, shape):
        with pytest.raises(ValueError, match="1152") as exc:
            FeatureStem()(torch.randn(*shape))
        assert "4-D" in str(exc.value)


# --------------------------------------------------------------------------- Group 4
class TestConfig:
    def test_set_by_arguments(self):
        m = build_dino_config()
        assert m._target_ is FeatureDINO
        assert m.backbone.net._target_ is FeatureStem
        assert (m.backbone.net.in_channels, m.backbone.net.embed_dim, m.backbone.net.out_feature) == \
            (1152, 768, "last_feat")
        assert m.num_classes == 6 and m.device == "cpu"

    def test_hyperparameter_table(self):
        m = build_dino_config()
        sfp = m.backbone
        assert sfp.in_feature == "last_feat"
        assert sfp.out_channels == 256 and tuple(sfp.scale_factors) == (2.0, 1.0, 0.5) and sfp.norm == "LN"
        assert sfp.square_pad == 1024
        assert list(m.neck.in_features) == ["p3", "p4", "p5", "p6"]
        assert all(s.channels == 256 for s in m.neck.input_shapes.values())
        assert (m.neck.num_outs, m.neck.out_channels, m.neck.kernel_size) == (4, 256, 1)
        assert (m.neck.norm_layer.num_groups, m.neck.norm_layer.num_channels) == (32, 256)
        pe = m.position_embedding
        assert (pe.num_pos_feats, pe.temperature, pe.normalize, pe.offset) == (128, 10000, True, -0.5)
        for part in (m.transformer.encoder, m.transformer.decoder):
            assert (part.num_layers, part.embed_dim, part.num_heads, part.feedforward_dim) == (6, 256, 8, 2048)
            assert (part.attn_dropout, part.ffn_dropout, part.use_checkpoint) == (0.0, 0.0, False)
            assert part.num_feature_levels == 4
        assert m.transformer.encoder.post_norm is False
        assert m.transformer.decoder.return_intermediate is True
        assert m.transformer.num_feature_levels == 4 and m.transformer.two_stage_num_proposals == 900
        assert (m.embed_dim, m.num_queries, m.aux_loss) == (256, 900, True)
        mt = m.criterion.matcher
        assert (mt.cost_class, mt.cost_bbox, mt.cost_giou) == (2.0, 5.0, 2.0)
        assert (mt.cost_class_type, mt.alpha, mt.gamma) == ("focal_loss_cost", 0.25, 2.0)
        w = m.criterion.weight_dict
        for suffix in ("", "_dn"):
            assert (w[f"loss_class{suffix}"], w[f"loss_bbox{suffix}"], w[f"loss_giou{suffix}"]) == (1, 5.0, 2.0)
        c = m.criterion
        assert (c.loss_class_type, c.alpha, c.gamma, c.two_stage_binary_cls) == ("focal_loss", 0.25, 2.0, False)
        assert c.num_classes == 6
        assert (m.dn_number, m.label_noise_ratio, m.box_noise_scale) == (100, 0.5, 1.0)
        assert m.vis_period == 0

    def test_weight_dict_equals_detrex(self):
        ours = OmegaConf.to_container(build_dino_config().criterion.weight_dict)
        assert ours == OmegaConf.to_container(load_dino_vitdet_config().model.criterion.weight_dict)

    def test_structure_match_and_param_counts(self, default_det):
        stock_cfg = load_dino_vitdet_config().model
        stock_cfg.num_classes, stock_cfg.device = 6, "cpu"
        stock = {k: v.shape for k, v in instantiate(stock_cfg).state_dict().items()}
        ours = {k: v.shape for k, v in default_det.detr.state_dict().items()}
        stem = "backbone.net."
        assert {k: v for k, v in stock.items() if not k.startswith(stem)} == \
            {k: v for k, v in ours.items() if not k.startswith(stem)}
        assert {k: tuple(v) for k, v in ours.items() if k.startswith(stem)} == {
            "backbone.net.norm.weight": (1152,), "backbone.net.norm.bias": (1152,),
            "backbone.net.proj.weight": (768, 1152), "backbone.net.proj.bias": (768,)}
        assert all(k.startswith("detr.") for k in default_det.state_dict())
        assert sum(p.numel() for p in default_det.parameters()) == 23_001_286
        assert sum(p.numel() for p in DINODetector(in_channels=3072).parameters()) == 24_479_686

    def test_overrides(self):
        w = build_dino_config(overrides={"transformer.decoder.num_layers": 3}).criterion.weight_dict
        suffixes = {k[len("loss_class"):] for k in w if k.startswith("loss_class") and "_dn" not in k}
        assert suffixes == {"", "_enc", "_0", "_1"}
        with pytest.raises(KeyError):
            build_dino_config(overrides={"no.such.key": 1})
        for bad in ({"num_classes": 3}, {"backbone.net.embed_dim": 512}):
            with pytest.raises(ValueError):
                build_dino_config(overrides=bad)

    def test_hydra_group(self, irwm_env):
        with initialize_config_dir(version_base="1.3", config_dir=str(CONF)):
            cfg = compose(config_name="irwm_base")
        det = OmegaConf.to_container(cfg.detector)
        assert det == {"in_channels": 1152, "stem_dim": 768, "num_classes": 6, "overrides": {}}
        defaults = {k: p.default for k, p in inspect.signature(DINODetector.__init__).parameters.items()
                    if k != "self"}
        assert defaults == {**det, "overrides": None}  # overrides: None in code ≡ {} in YAML
        assert isinstance(DINODetector(**det), DINODetector)


# --------------------------------------------------------------------------- Group 5
class TestWrapperCPU:
    def test_eval_default_grid(self, eval_runs):
        (out,) = eval_runs[(32, 64)]["out"]
        boxes, scores, labels = out["boxes"], out["scores"], out["labels"]
        assert boxes.dtype == torch.float32 and boxes.ndim == 2 and boxes.shape[1] == 4
        assert 0 < boxes.shape[0] <= 300
        x0, y0, x1, y1 = boxes.unbind(1)
        assert bool(((0 <= x0) & (x0 <= x1) & (x1 <= 896)).all())
        assert bool(((0 <= y0) & (y0 <= y1) & (y1 <= 448)).all())
        assert bool(((0 <= scores) & (scores <= 1)).all())
        assert labels.dtype == torch.int64 and bool(((0 <= labels) & (labels <= 5)).all())

    def test_eval_native_grid(self, eval_runs):
        (out,) = eval_runs[(32, 40)]["out"]
        assert float(out["boxes"][:, 0::2].max()) <= 560 and float(out["boxes"][:, 1::2].max()) <= 448
        (explicit,) = eval_runs["explicit"]
        for k in ("boxes", "scores", "labels"):
            assert torch.equal(out[k], explicit[k]), k

    def test_pyramid(self, eval_runs):
        assert [tuple(s) for s in eval_runs[(32, 64)]["neck"]] == \
            [(1, 256, 64, 128), (1, 256, 32, 64), (1, 256, 16, 32), (1, 256, 8, 16)]
        assert [tuple(s[-2:]) for s in eval_runs[(32, 40)]["neck"]] == [(64, 80), (32, 40), (16, 20), (8, 10)]

    def test_deterministic_and_batched(self, reduced_det):
        feats = torch.randn(3, 1152, 8, 16)
        with torch.no_grad():
            a, b = reduced_det(feats), reduced_det(feats)
        assert len(a) == 3
        for da, db in zip(a, b):
            for k in ("boxes", "scores", "labels"):
                assert torch.equal(da[k], db[k]), k

    def test_train_errors(self):
        det = DINODetector(overrides=REDUCED).train()
        feats = torch.randn(3, 1152, 8, 16)
        targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 40.0]]), "labels": torch.tensor([1])}] * 3
        with pytest.raises(RuntimeError, match=r"\.cuda\(\)"):
            det(feats, targets)
        with pytest.raises(ValueError):
            det(feats, None)
        with pytest.raises(ValueError):
            det(feats, targets[:2])

    def test_too_few_tokens(self, default_det):
        with pytest.raises(ValueError, match="170") as exc:
            default_det(torch.randn(1, 1152, 4, 8))
        assert "900" in str(exc.value)

    def test_preprocess_image(self, reduced_det):
        xs = [torch.randn(1152, 8, 16) for _ in range(2)]
        images = reduced_det.detr.preprocess_image([{"image": x} for x in xs])
        assert images.tensor.shape == (2, 1152, 8, 16)
        assert torch.equal(images.tensor, torch.stack(xs))

    def test_feature_dino_overrides_only_preprocess(self):
        own = set(vars(FeatureDINO)) - set(vars(object)) - {"__module__", "__doc__", "__qualname__"}
        assert own == {"preprocess_image"}


# --------------------------------------------------------------------------- Group 6
GT_BOXES = torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.5, 0.1, 0.3], [0.8, 0.7, 0.15, 0.1]])  # cxcywh
GT_LABELS = torch.tensor([1, 4, 2])
MATCHED_QUERIES = (7, 2, 5)  # query q_i predicts GT box i


def _perfect_outputs(wrong_class: bool = False) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    logits = torch.full((1, 10, 6), -10.0)
    boxes = torch.rand(1, 10, 4, generator=g) * 0.4 + 0.1
    for i, q in enumerate(MATCHED_QUERIES):
        label = (int(GT_LABELS[i]) + 1) % 6 if wrong_class else int(GT_LABELS[i])
        logits[0, q, label] = 10.0
        boxes[0, q] = GT_BOXES[i]
    return {"pred_logits": logits, "pred_boxes": boxes}


class TestMatcherAndCriterion:
    def test_matcher(self, reduced_det):
        matcher = reduced_det.detr.criterion.matcher
        ((q, t),) = matcher(_perfect_outputs(), [{"labels": GT_LABELS, "boxes": GT_BOXES}])
        assert q.tolist() == [2, 5, 7] and t.tolist() == [1, 2, 0]

    def test_matcher_empty_target(self, reduced_det):
        matcher = reduced_det.detr.criterion.matcher
        ((q, t),) = matcher(_perfect_outputs(),
                            [{"labels": torch.zeros(0, dtype=torch.int64), "boxes": torch.zeros(0, 4)}])
        assert q.dtype == t.dtype == torch.int64 and q.numel() == t.numel() == 0

    def test_set_criterion(self, reduced_det):
        crit = reduced_det.detr.criterion
        two_stage = importlib.import_module("projects.dino.modeling.two_stage_criterion").TwoStageCriterion
        assert isinstance(crit, two_stage)
        targets = [{"labels": GT_LABELS, "boxes": GT_BOXES}]
        good = two_stage.forward(crit, _perfect_outputs(), targets)
        assert set(good) == {"loss_class", "loss_bbox", "loss_giou"}
        assert float(good["loss_bbox"]) == pytest.approx(0, abs=1e-6)
        # detrex's box_iou divides by union + 1e-6 (box_ops.py:81): a perfect box keeps 1e-6/(A+1e-6).
        areas = GT_BOXES[:, 2] * GT_BOXES[:, 3]
        assert float(good["loss_giou"]) == pytest.approx(float((1e-6 / (areas + 1e-6)).mean()), rel=1e-3)
        assert float(good["loss_class"]) < 1e-3
        bad = two_stage.forward(crit, _perfect_outputs(wrong_class=True), targets)
        assert float(bad["loss_class"]) >= 10 * float(good["loss_class"])


# --------------------------------------------------------------------------- Groups 7-8 (GPU)
gpu = [pytest.mark.gpu,
       pytest.mark.skipif(not cuda_ops_available(), reason="needs CUDA and detrex built with CUDA ops")]

LOSS_KEYS = {f"{loss}{s}" for loss in ("loss_class", "loss_bbox", "loss_giou")
             for s in ["", "_enc", "_dn"] + [f"_{i}" for i in range(5)] + [f"_dn_{i}" for i in range(5)]}


def _targets(counts: list[int], gen: torch.Generator, size: tuple[int, int] = (448, 896)) -> list[dict]:
    """plan.md Step 7 boxes: centres uniform in the frame, w/h uniform in [32, 200] px, clipped."""
    h, w = size
    out = []
    for n in counts:
        c = torch.rand(n, 2, generator=gen) * torch.tensor([w, h])
        wh = torch.rand(n, 2, generator=gen) * 168 + 32
        boxes = torch.cat([c - wh / 2, c + wh / 2], 1)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        out.append({"boxes": boxes.cuda(), "labels": torch.randint(0, 6, (n,), generator=gen).cuda()})
    return out


@pytest.fixture(scope="module")
def cuda_det() -> DINODetector:
    torch.manual_seed(0)
    return DINODetector().cuda()


class TestCudaOps:
    pytestmark = gpu

    def test_cuda_ops_available(self):
        assert cuda_ops_available() is True

    def test_msda_cuda_matches_pytorch(self):
        from detrex.layers.multi_scale_deform_attn import (MultiScaleDeformableAttnFunction,
                                                           multi_scale_deformable_attn_pytorch)

        torch.manual_seed(0)
        shapes = torch.tensor([[16, 32], [8, 16], [4, 8], [2, 4]], device="cuda")
        start = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
        bs, nq, heads, dims, levels, points = 2, 30, 8, 32, 4, 4
        value = torch.randn(bs, int((shapes[:, 0] * shapes[:, 1]).sum()), heads, dims, device="cuda")
        loc = torch.rand(bs, nq, heads, levels, points, 2, device="cuda")
        attn = torch.rand(bs, nq, heads, levels * points, device="cuda").softmax(-1).view(
            bs, nq, heads, levels, points)
        grad_out = torch.randn(bs, nq, heads * dims, device="cuda")
        results = []
        for fn in ("cuda", "pytorch"):
            v, l, a = (t.clone().requires_grad_(True) for t in (value, loc, attn))
            if fn == "cuda":
                out = MultiScaleDeformableAttnFunction.apply(v, shapes, start, l, a, 64)
            else:
                out = multi_scale_deformable_attn_pytorch(v, shapes, l, a)
            out.backward(grad_out)
            results.append((out.detach(), v.grad, l.grad, a.grad))
        (o1, *g1), (o2, *g2) = results
        torch.testing.assert_close(o1, o2, rtol=1e-4, atol=1e-5)
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)

    def test_train_losses(self, cuda_det):
        cuda_det.train()
        cuda_det.zero_grad(set_to_none=True)
        feats = torch.randn(2, 1152, 32, 64, device="cuda")
        losses = cuda_det(feats, _targets([3, 0], torch.Generator().manual_seed(0)))
        assert set(losses) == LOSS_KEYS and len(losses) == 39
        assert all(torch.isfinite(v).all() for v in losses.values())
        sum(losses.values()).backward()
        grad = cuda_det.detr.backbone.net.proj.weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0

    def test_train_losses_all_empty(self, cuda_det):
        cuda_det.train()
        losses = cuda_det(torch.randn(2, 1152, 32, 64, device="cuda"), _targets([0, 0], torch.Generator()))
        assert set(losses) == LOSS_KEYS
        assert all(torch.isfinite(v).all() for v in losses.values())
        assert all(float(v) == 0 for k, v in losses.items() if "_dn" in k)

    def test_fp16_autocast(self, cuda_det):
        feats = torch.randn(2, 1152, 32, 64, device="cuda")
        targets = _targets([3, 0], torch.Generator().manual_seed(0))
        with torch.autocast("cuda", dtype=torch.float16):
            cuda_det.train()
            losses = cuda_det(feats, targets)
            assert all(torch.isfinite(v).all() for v in losses.values())
            cuda_det.eval()
            with torch.no_grad():
                outs = cuda_det(feats)
        assert all(torch.isfinite(o["boxes"]).all() for o in outs)


@pytest.mark.gpu
@pytest.mark.skipif(not cuda_ops_available(), reason="needs CUDA and detrex built with CUDA ops")
def test_overfit_synthetic():
    """Group 8: 4 synthetic maps, 11 boxes, full default config, fp32, detrex 12ep optimiser."""
    gen = torch.Generator().manual_seed(0)
    feats = torch.randn(4, 1152, 32, 64, generator=gen).cuda()
    targets = _targets([1, 2, 3, 5], gen)
    torch.manual_seed(0)
    det = DINODetector().cuda()
    opt = torch.optim.AdamW(det.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    loss0, passed, summary = None, False, {}
    for it in range(1, 1501):
        det.train()
        loss = sum(det(feats, targets).values())
        loss0 = float(loss) if loss0 is None else loss0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(det.parameters(), max_norm=0.1, norm_type=2)
        opt.step()
        if it % 100:
            continue
        det.eval()
        with torch.no_grad():
            outs = det(feats)
        ious, scores = [], []
        for out, tgt in zip(outs, targets):
            iou = box_iou(tgt["boxes"], out["boxes"])  # [N_gt, K]
            same = tgt["labels"][:, None] == out["labels"][None, :]
            confident = same & (out["scores"][None, :] >= 0.5)
            overlapping = same & (iou >= 0.8)
            ious += torch.where(confident, iou, torch.zeros_like(iou)).max(1).values.tolist()
            scores += torch.where(overlapping, out["scores"][None, :].expand_as(iou),
                                  torch.zeros_like(iou)).max(1).values.tolist()
        ratio = float(loss) / loss0
        summary = {"iterations": it, "loss_ratio": round(ratio, 4), "min_iou": round(min(ious), 4),
                   "min_score": round(min(scores), 4), "seconds": round(time.perf_counter() - t0, 1),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(f"[overfit] {summary}", flush=True)
        if min(ious) >= 0.8 and min(scores) >= 0.5 and ratio <= 0.1:
            passed = True
            break
    assert math.isfinite(loss0)
    assert passed, f"overfit bar not met within 1500 iterations: {summary}"
