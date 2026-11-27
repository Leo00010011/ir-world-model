"""F17 · ``dino3072`` backbone: frozen DINOv2-Reg 3072-d features + learned Linear(3072 → 1152) in the head.

CPU tests use a stub ViT and the reduced detector config (no data, no weights). ``data`` tests run on
the real DINOv2 weights (``IRWM_CKPT_ROOT``) and the exported ``det_ir`` head, on CUDA if available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from irwm.detection import DINODetector, FeatureStem
from irwm.detection import frozen as F
from irwm.features import encoder as E
from irwm.features.encoder import (ENCODER_CHANNELS, DinoV2RegEncoder, FrozenEncoder, build_frozen_encoder,
                                   encoder_tag, frozen_encoder_from_config)
from irwm.features.pca import PCABasis
from irwm.utils.paths import repo_root

CONF = repo_root() / "lib" / "irwm" / "conf"
REDUCED = {"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1,
           "transformer.decoder.num_layers": 2}
NEW_KEYS = {"detr.backbone.net.in_proj.weight", "detr.backbone.net.in_proj.bias"}
PROJ_PARAMS = 3072 * 1152 + 1152


class StubViT(nn.Module):
    """``get_intermediate_layers(reshape=False)`` stand-in: per-patch mean pixel through a Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 768)
        self.drop = nn.Dropout(0.5)  # would change outputs if train mode leaked through

    def get_intermediate_layers(self, x, n, reshape, norm):
        b = x.shape[0]
        p = x.mean(1).unfold(1, 14, 14).unfold(2, 14, 14).mean((-1, -2)).reshape(b, -1, 1)
        return tuple(self.drop(self.proj(p)) * (k + 1) for k in n)


def _basis() -> PCABasis:
    g = torch.Generator().manual_seed(0)
    return PCABasis(torch.randn(3072, generator=g), torch.rand(3072, generator=g) + 0.5,
                    0.1 * torch.randn(3072, generator=g), torch.randn(1152, 3072, generator=g))


def _x(b: int = 2, h: int = 14 * 4, w: int = 14 * 5) -> torch.Tensor:
    return torch.randn(b, 3, h, w, generator=torch.Generator().manual_seed(1))


# --------------------------------------------------------------------------- encoder
class TestEncoder:
    def test_tags(self):
        assert encoder_tag("native") == "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152"
        assert encoder_tag("native", "dinopca") == encoder_tag("native")
        assert encoder_tag("native", "dino3072") == "dinov2_vitb14reg_448x560_l3-6-9-12_full3072"
        assert encoder_tag("stretch", "dino3072") == "dinov2_vitb14reg_448x896_l3-6-9-12_full3072"
        with pytest.raises(ValueError, match="encoder kind"):
            encoder_tag("native", "mae")  # type: ignore[arg-type]
        assert ENCODER_CHANNELS == {"dinopca": 1152, "dino3072": 3072}

    def test_dino3072_encode(self):
        enc = DinoV2RegEncoder(StubViT(), None, precision="fp32", policy="native")
        assert enc.kind == "dino3072" and enc.out_channels == 3072 and enc.pca is None
        assert enc.tag == "dinov2_vitb14reg_448x560_l3-6-9-12_full3072"
        assert isinstance(enc, FrozenEncoder)
        enc.train()
        assert not enc.training and all(not p.requires_grad for p in enc.parameters())
        x = _x()
        out = enc.encode(x)
        assert out.shape == (2, 3072, 4, 5) and out.dtype == torch.float32 and out.is_contiguous()
        assert not out.requires_grad
        assert torch.equal(out, enc.encode_full(x).permute(0, 3, 1, 2))
        assert not any(k.startswith("pca.") for k in enc.state_dict())
        with pytest.raises(RuntimeError, match="no PCA basis"):
            enc.encode_pca(x)
        with pytest.raises(RuntimeError, match="no PCA basis"):
            enc.pca_inverse(torch.zeros(1152))

    def test_dinopca_encode_unchanged(self):
        enc = DinoV2RegEncoder(StubViT(), _basis(), precision="fp32")
        assert enc.kind == "dinopca" and enc.out_channels == 1152 and enc.policy is None
        assert {k.split(".")[0] for k in enc.state_dict()} == {"backbone", "pca"}
        x = _x()
        assert torch.equal(enc.encode(x), enc.encode_pca(x).permute(0, 3, 1, 2))
        with pytest.raises(ValueError, match="no policy"):
            _ = enc.tag

    def test_kind_validation(self):
        with pytest.raises(ValueError, match="needs a PCABasis"):
            DinoV2RegEncoder(StubViT(), None, kind="dinopca")
        with pytest.raises(ValueError, match="encoder kind"):
            DinoV2RegEncoder(StubViT(), None, kind="mae")  # type: ignore[arg-type]
        # a PCA-capable encoder can still be asked for the full features
        enc = DinoV2RegEncoder(StubViT(), _basis(), precision="fp32", kind="dino3072")
        assert enc.encode(_x()).shape[1] == 3072

    def test_from_assets_pca_flag(self, irwm_env, monkeypatch):
        import irwm.worldmodel.reference as R

        with pytest.raises(FileNotFoundError, match="download.py"):
            DinoV2RegEncoder.from_assets(pca=False)  # hub code / weights still required
        for path in (E.dinov2_code_dir(E.DINOV2_HUB_PINNED) / "hubconf.py", E.dinov2_weights_path()):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        monkeypatch.setattr(R, "load_dinov2_backbone", lambda version: StubViT())
        with pytest.raises(FileNotFoundError, match=E.PCA_EXPORT):
            DinoV2RegEncoder.from_assets()  # dinopca still needs the PCA file
        enc = DinoV2RegEncoder.from_assets(pca=False)
        assert enc.kind == "dino3072" and enc.pca is None

        calls = []
        monkeypatch.setattr(E, "set_attention", lambda bb, impl: calls.append(impl) or 1)
        enc = frozen_encoder_from_config({"kind": "dino3072", "policy": "native", "precision": "fp32",
                                          "hub_version": "dinof_release", "attention": "sdpa"})
        assert enc.kind == "dino3072" and enc.policy == "native" and calls == ["sdpa"]
        assert enc.tag.endswith("_full3072") and not enc.training
        with pytest.raises(FileNotFoundError):  # no kind -> dinopca -> PCA file required
            frozen_encoder_from_config({"policy": "native"})
        with pytest.raises(ValueError, match="(?i)policy"):
            build_frozen_encoder("dino3072", policy="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize("kind", ["dinopca", "dino3072"])
    def test_native640_rejected(self, irwm_env, kind):
        # MAE-only policy (512x640, 512 % 14 != 0): rejected before any asset is touched
        with pytest.raises(ValueError, match="native640.*not valid for the DINOv2 encoder"):
            encoder_tag("native640", kind)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="native640.*not valid for the DINOv2 encoder"):
            build_frozen_encoder(kind, policy="native640")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="native640.*not valid for the DINOv2 encoder"):
            frozen_encoder_from_config({"kind": kind, "policy": "native640"})
        with pytest.raises(ValueError, match="native640.*not valid for the DINOv2 encoder"):
            DinoV2RegEncoder(StubViT(), None, policy="native640")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- stem / detector
class TestHead:
    def test_stem_default_keys_unchanged(self):
        assert set(FeatureStem().state_dict()) == {"norm.weight", "norm.bias", "proj.weight", "proj.bias"}
        assert set(FeatureStem(in_channels=3072).state_dict()) == {"norm.weight", "norm.bias",
                                                                   "proj.weight", "proj.bias"}

    def test_stem_projection(self):
        stem = FeatureStem(in_channels=3072, proj_channels=1152)
        sd = stem.state_dict()
        assert set(sd) == {"in_proj.weight", "in_proj.bias", "norm.weight", "norm.bias", "proj.weight", "proj.bias"}
        assert sd["in_proj.weight"].shape == (1152, 3072) and sd["norm.weight"].shape == (1152,)
        assert sd["proj.weight"].shape == (768, 1152)
        x = torch.randn(2, 3072, 4, 5)
        y = stem(x)["last_feat"]
        assert y.shape == (2, 768, 4, 5)
        y.sum().backward()
        assert stem.in_proj.weight.grad is not None and stem.in_proj.weight.grad.abs().sum() > 0
        with pytest.raises(ValueError, match="3072"):
            stem(torch.randn(1, 1152, 4, 5))

    def test_detector_keys(self):
        base = DINODetector(overrides=REDUCED, num_classes=2).state_dict()
        new = DINODetector(in_channels=3072, in_proj=1152, overrides=REDUCED, num_classes=2).state_dict()
        assert not any("net.in_proj" in k for k in base)
        assert set(new) == set(base) | NEW_KEYS
        for k, v in base.items():
            assert new[k].shape == v.shape, k
        assert sum(new[k].numel() for k in NEW_KEYS) == PROJ_PARAMS
        n = lambda sd: sum(v.numel() for v in sd.values())  # noqa: E731
        assert n(new) - n(base) == PROJ_PARAMS

    def test_detector_full_config_keys(self):
        base = DINODetector()
        assert base.in_proj is None and not any("net.in_proj" in k for k in base.state_dict())
        new = DINODetector(in_channels=3072, in_proj=1152)
        assert set(new.state_dict()) == set(base.state_dict()) | NEW_KEYS

    def test_eval_forward_two_classes(self):
        torch.manual_seed(0)
        # detrex keeps the top select_box_nums_for_evaluation (300) of num_queries x num_classes scores
        det = DINODetector(in_channels=3072, in_proj=1152, num_classes=2,
                           overrides={**REDUCED, "num_queries": 200}).eval()
        with torch.no_grad():
            out = det(torch.randn(1, 3072, 8, 16))
        assert len(out) == 1 and out[0]["boxes"].shape[1] == 4
        assert int(out[0]["labels"].max()) < 2
        with pytest.raises(ValueError, match="3072"):
            det(torch.randn(1, 1152, 8, 16))

    def test_config_defaults(self):
        with initialize_config_dir(config_dir=str(CONF / "encoder"), version_base=None):
            enc = compose(config_name="default")
        with initialize_config_dir(config_dir=str(CONF / "detector"), version_base=None):
            det = compose(config_name="default")
        assert enc.kind == "dinopca" and det.in_proj is None and det.in_channels == 1152


# --------------------------------------------------------------------------- frozen heads
DET3072 = {"in_channels": 3072, "in_proj": 1152, "stem_dim": 768, "num_classes": 2, "overrides": REDUCED}
ENC3072 = {"kind": "dino3072", "policy": "native", "precision": "fp16", "hub_version": "dinof_release",
           "attention": "sdpa"}


def _ckpt(irwm_env, detector: dict, encoder: dict, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    det = DINODetector(**detector)
    state = {f"detector.{k}": v for k, v in det.state_dict().items()}
    ckpt = {"state_dict": state, "epoch": 4, "global_step": 10,
            "hyper_parameters": {"detector": detector, "encoder": encoder}, "irwm": {"metrics_dir": None}}
    path = irwm_env["ckpt"] / "detection" / "run" / f"best-{seed}.ckpt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    return path


class TestFrozen:
    def test_names(self):
        assert F.FROZEN_NAMES[:2] == ("det_rgb", "det_ir")
        assert F.FROZEN_MODALITY["det_rgb"] == "rgb" and F.FROZEN_MODALITY["det_ir"] == "thermal"
        for b in ("dinopca", "dino3072", "mae"):
            assert F.FROZEN_MODALITY[f"aligned_{b}_rgb"] == "rgb"
            assert F.FROZEN_MODALITY[f"aligned_{b}_ir"] == "thermal"
            assert F.FROZEN_DATASET[f"aligned_{b}_ir"] == "aligned_v1"
        assert F.FROZEN_DATASET["det_ir"] == "flir_v2" and len(F.FROZEN_NAMES) == 10
        with pytest.raises(ValueError, match="Unknown frozen detector"):
            F.frozen_dir("aligned_resnet_ir")

    def test_export_load_dino3072(self, irwm_env):
        path = _ckpt(irwm_env, DET3072, ENC3072)
        out = F.export_frozen("aligned_dino3072_ir", path)
        meta = F.frozen_meta("aligned_dino3072_ir")
        assert meta["dataset"] == "aligned_v1" and meta["modality"] == "thermal"
        assert meta["encoder"]["encoder_tag"] == "dinov2_vitb14reg_448x560_l3-6-9-12_full3072"
        assert meta["detector"]["in_proj"] == 1152
        head = torch.load(out / "head.pt")
        assert NEW_KEYS <= set(head)
        det = F.load_frozen_detector("aligned_dino3072_ir")
        assert det.in_proj == 1152 and not det.training and all(not p.requires_grad for p in det.parameters())
        for k, v in det.state_dict().items():
            assert torch.equal(v, head[k]), k

    def test_dataset_arg(self, irwm_env):
        path = _ckpt(irwm_env, DET3072, ENC3072)
        with pytest.raises(ValueError, match="belongs to dataset"):
            F.export_frozen("aligned_dino3072_rgb", path, dataset="flir_v2")
        with pytest.raises(ValueError, match="Unknown dataset"):
            F.export_frozen("aligned_dino3072_rgb", path, dataset="coco")
        F.export_frozen("aligned_dino3072_rgb", path, dataset="aligned_v1")
        assert F.frozen_meta("aligned_dino3072_rgb")["dataset"] == "aligned_v1"

    def test_old_meta_backward_compatible(self, irwm_env):
        det_cfg = {"in_channels": 1152, "stem_dim": 768, "num_classes": 6, "overrides": REDUCED}
        enc_cfg = {"policy": "native", "precision": "fp16", "hub_version": "dinof_release", "attention": "sdpa"}
        out = F.export_frozen("det_rgb", _ckpt(irwm_env, det_cfg, enc_cfg))
        meta = json.loads((out / "meta.json").read_text())
        assert meta["encoder"]["encoder_tag"] == "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152"
        meta.pop("dataset")  # as written before the field existed
        (out / "meta.json").write_text(json.dumps(meta))
        assert F.frozen_meta("det_rgb")["dataset"] == "flir_v2"
        det = F.load_frozen_detector("det_rgb")
        assert det.in_proj is None and not any("net.in_proj" in k for k in det.state_dict())

    def test_mae_meta_refused(self, irwm_env):
        out = F.frozen_dir("aligned_mae_ir")
        out.mkdir(parents=True)
        (out / "meta.json").write_text(json.dumps({"detector": {"arch": "vitdet_mae"}, "head_sha256": "0"}))
        with pytest.raises(ValueError, match="vitdet_mae"):
            F.load_frozen_detector("aligned_mae_ir")


# --------------------------------------------------------------------------- real weights
@pytest.fixture
def real_ckpt(monkeypatch):
    root = repo_root() / "checkpoints"
    if not (root / "torch_hub").is_dir():
        pytest.skip("real checkpoints not found")
    monkeypatch.setenv("IRWM_CKPT_ROOT", str(root))
    return root


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.data
class TestRealWeights:
    def test_dino3072_real(self, real_ckpt):
        dev = _device()
        enc = build_frozen_encoder("dino3072", policy="native", precision="fp16", attention="sdpa", device=dev)
        assert enc.pca is None and enc.tag.endswith("_full3072")
        x = torch.randn(1, 3, 448, 560, generator=torch.Generator().manual_seed(0)).to(dev)
        out = enc.encode(x)
        assert out.shape == (1, 3072, 32, 40) and out.dtype == torch.float32 and torch.isfinite(out).all()
        assert torch.equal(out, enc.encode_full(x).permute(0, 3, 1, 2))

        pca = build_frozen_encoder("dinopca", policy="native", precision="fp16", attention="sdpa", device=dev)
        z = pca.encode(x)
        assert z.shape == (1, 1152, 32, 40) and torch.equal(z, pca.encode_pca(x).permute(0, 3, 1, 2))
        # the two kinds share the backbone: PCA of the 3072-d output equals the dinopca output (fp16 tolerance)
        ref = pca.pca.transform(out.permute(0, 2, 3, 1).float())
        assert torch.allclose(ref.permute(0, 3, 1, 2), z, atol=0.05 * float(z.abs().max()))

    def test_existing_head_loads_strict(self, real_ckpt):
        if not (real_ckpt / "detection" / "frozen" / "det_ir" / "head.pt").is_file():
            pytest.skip("det_ir head not exported here")
        det = F.load_frozen_detector("det_ir")
        assert det.in_proj is None and not any("net.in_proj" in k for k in det.state_dict())
        assert F.frozen_meta("det_ir")["dataset"] == "flir_v2"
