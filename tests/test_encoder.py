"""F4 · DINOv2-Reg encoder wrapper. See spec/2026-10-16-dinov2-reg-encoder-wrapper/validation.md."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn

from irwm.features.encoder import DinoV2RegEncoder, encoder_tag
from irwm.features.pca import PCABasis
from irwm.features.preprocess import POLICY_SIZE, Preprocess, ResizeGeometry
from irwm.utils.paths import repo_root
from irwm.worldmodel import reference as R

REPO = repo_root()
MEAN = torch.tensor(R.IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(R.IMAGENET_STD).view(3, 1, 1)
# Every source size in F2 stats.md, (W, H) → expected native 5:4 crop box.
NATIVE_BOXES = {
    (640, 512): (0, 0, 640, 512),
    (1280, 1024): (0, 0, 1280, 1024),
    (1224, 1024): (0, 22, 1224, 979),
    (1024, 768): (32, 0, 960, 768),
    (1800, 1600): (0, 80, 1800, 1440),
    (2048, 1536): (64, 0, 1920, 1536),
    (720, 480): (60, 0, 600, 480),
}


def _img(h: int, w: int, rgb: bool, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3) if rgb else (h, w), dtype=np.uint8)


def _geom(policy: str, w: int, h: int) -> ResizeGeometry:
    rgb = (w, h) != (640, 512)
    return Preprocess(policy, "rgb" if rgb else "thermal")(_img(h, w, rgb))[1]


# --------------------------------------------------------------------------- Group 1
class TestPreprocess:
    @pytest.mark.parametrize("policy", ["stretch", "crop", "native"])
    def test_output_shapes(self, policy):
        h, w = POLICY_SIZE[policy]
        for img, mod in [(_img(512, 640, False), "thermal"), (_img(1024, 1224, True), "rgb"),
                         (_img(768, 1024, True), "rgb")]:
            x, geom = Preprocess(policy, mod)(img)
            assert x.dtype == torch.float32 and tuple(x.shape) == (3, h, w)
            assert geom.out_size == (w, h)

    def test_thermal_replication(self):
        x, _ = Preprocess("stretch", "thermal")(_img(512, 640, False))
        raw = x * STD + MEAN
        # Undo Normalize, then back to the 8-bit grid ToTensor started from.
        q = torch.round(raw * 255)
        assert torch.equal(q[0], q[1]) and torch.equal(q[0], q[2])
        assert float((raw[0] - raw[1]).abs().max()) < 1e-6 and float((raw[0] - raw[2]).abs().max()) < 1e-6

    def test_normalisation(self):
        x, _ = Preprocess("stretch", "rgb")(np.full((512, 640, 3), 255, dtype=np.uint8))
        for c in range(3):
            expected = (1 - R.IMAGENET_MEAN[c]) / R.IMAGENET_STD[c]
            assert torch.allclose(x[c], torch.full_like(x[c], expected), atol=1e-6)

    @staticmethod
    def _demo_arrays() -> list[np.ndarray]:
        return [np.array(Image.open(p).convert("RGB"), dtype=np.uint8) for p in R.demo_frame_paths()]

    def test_stretch_equals_reference_transform(self):
        ours = torch.stack([Preprocess("stretch", "rgb")(a)[0] for a in self._demo_arrays()])
        assert torch.equal(ours, R.load_demo_frames((448, 896))[0])

    def test_crop_equals_f1_probe_transform(self):
        ours = torch.stack([Preprocess("crop", "rgb")(a)[0] for a in self._demo_arrays()])
        assert torch.equal(ours, R.load_demo_frames((448, 896), (2, 1))[0])

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            Preprocess("stretch", "thermal")(_img(64, 80, True))
        with pytest.raises(ValueError):
            Preprocess("stretch", "rgb")(_img(64, 80, False))
        with pytest.raises(ValueError):
            Preprocess("stretch", "rgb")(_img(64, 80, True).astype(np.float32))
        with pytest.raises(ValueError):
            Preprocess("windows", "rgb")
        with pytest.raises(ValueError):
            Preprocess("stretch", "depth")

    def test_picklable(self):
        p = Preprocess("native", "thermal")
        img = _img(512, 640, False, seed=3)
        q = pickle.loads(pickle.dumps(p))
        assert torch.equal(p(img)[0], q(img)[0])


# --------------------------------------------------------------------------- Group 2
class TestResizeGeometry:
    def test_stretch_scaling(self):
        out, keep = _geom("stretch", 640, 512).apply_boxes(np.array([[64, 51.2, 64, 51.2]]))
        assert np.allclose(out, [[89.6, 44.8, 89.6, 44.8]], atol=1e-4) and keep.tolist() == [True]

    def test_crop_offset(self):
        g = _geom("crop", 640, 512)
        assert g.crop_box == (0, 96, 640, 320)
        out, keep = g.apply_boxes(np.array([[100, 200, 50, 40]]))
        assert np.allclose(out, [[140, 145.6, 70, 56]], atol=1e-4) and keep.all()

    def test_crop_clipping_and_drop(self):
        out, keep = _geom("crop", 640, 512).apply_boxes(np.array([[10, 80, 20, 30], [10, 10, 20, 30]]))
        assert np.allclose(out[0], [14, 0, 28, 19.6], atol=1e-4)
        assert keep.tolist() == [True, False] and (out[1] == 0).all()

    def test_native_crop_boxes(self):
        for (w, h), box in NATIVE_BOXES.items():
            g = _geom("native", w, h)
            assert g.crop_box == box, (w, h)
            assert g.out_size == (560, 448)
            assert g.src_size == (w, h)

    def test_native_thermal_untouched(self):
        img = _img(512, 640, False, seed=5)
        x, g = Preprocess("native", "thermal")(img)
        tf = T.Compose([T.Resize((448, 560)), T.ToTensor(), T.Normalize(R.IMAGENET_MEAN, R.IMAGENET_STD)])
        assert torch.equal(x, tf(Image.fromarray(img).convert("RGB")))
        out, keep = g.apply_boxes(np.array([[64, 51.2, 64, 51.2]]))
        assert np.allclose(out, [[56, 44.8, 56, 44.8]], atol=1e-4) and keep.all()

    def test_native_rgb_4_3(self):
        out, keep = _geom("native", 1024, 768).apply_boxes(np.array([[0, 0, 1024, 768], [0, 0, 20, 20]]))
        assert np.allclose(out[0], [0, 0, 560, 448], atol=1e-3)
        assert keep.tolist() == [True, False]

    def test_native_aspect_preserved(self):
        for w, h in NATIVE_BOXES:
            _, _, cw, ch = _geom("native", w, h).crop_box
            assert abs(cw / ch / 1.25 - 1) < 1e-3, (w, h)

    @pytest.mark.parametrize("policy", ["stretch", "crop", "native"])
    def test_round_trip(self, policy):
        rng = np.random.default_rng(0)
        for w, h in NATIVE_BOXES:
            g = _geom(policy, w, h)
            left, top, cw, ch = g.crop_box
            x0 = left + rng.uniform(0, cw - 1, 100)
            y0 = top + rng.uniform(0, ch - 1, 100)
            bw = rng.uniform(0.5, 1, 100) * (left + cw - x0)
            bh = rng.uniform(0.5, 1, 100) * (top + ch - y0)
            b = np.stack([x0, y0, bw, bh], axis=1)
            out, keep = g.apply_boxes(b)
            assert keep.all()
            assert np.allclose(g.invert_boxes(out), b, atol=1e-3), (policy, w, h)

    def test_empty(self):
        out, keep = _geom("crop", 640, 512).apply_boxes(np.zeros((0, 4), dtype=np.float32))
        assert out.shape == (0, 4) and keep.shape == (0,)
        assert _geom("crop", 640, 512).invert_boxes(np.zeros((0, 4))).shape == (0, 4)


# --------------------------------------------------------------------------- Group 3
class StubBackbone(nn.Module):
    """Mimics ``DinoVisionTransformer.get_intermediate_layers(reshape=False)``: patch tokens only.

    ``mode``: ``"pixels"`` (per-patch mean pixel × (k+1), through a Linear so it has a
    parameter), ``"const"`` (block k → constant k+1) or ``"index"`` (token i → value i).
    """

    def __init__(self, mode: str = "pixels") -> None:
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(1, 768)
        self.drop = nn.Dropout(0.5)  # would change outputs if train mode ever leaked through
        self.calls: list[dict] = []

    def get_intermediate_layers(self, x, n, reshape, norm):
        self.calls.append({"n": n, "reshape": reshape, "norm": norm})
        b, _, h, w = x.shape
        patches = x.mean(1).unfold(1, 14, 14).unfold(2, 14, 14).mean((-1, -2)).reshape(b, -1, 1)
        out = []
        for k in n:
            if self.mode == "const":
                t = torch.full((b, patches.shape[1], 768), float(k + 1), device=x.device)
            elif self.mode == "index":
                t = torch.arange(patches.shape[1], dtype=x.dtype, device=x.device).view(1, -1, 1).expand(b, -1, 768)
            else:
                t = self.drop(self.proj(patches)) * (k + 1)
            out.append(t)
        return tuple(out)


def _basis(seed: int = 0) -> PCABasis:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3072, 3072, generator=g))
    return PCABasis(torch.randn(3072, generator=g), torch.rand(3072, generator=g) + 0.5,
                    0.1 * torch.randn(3072, generator=g), q[:1152].contiguous())


@pytest.fixture(scope="module")
def basis() -> PCABasis:
    return _basis()


def _x(b: int = 2, h: int = 448, w: int = 896) -> torch.Tensor:
    return torch.randn(b, 3, h, w, generator=torch.Generator().manual_seed(1))


class TestEncoderStub:
    def test_call_contract(self, basis):
        stub = StubBackbone()
        y = DinoV2RegEncoder(stub, basis, precision="fp32").encode_full(_x())
        assert stub.calls == [{"n": [2, 5, 8, 11], "reshape": False, "norm": True}]
        assert tuple(y.shape) == (2, 32, 64, 3072) and y.dtype == torch.float32

    def test_block_order(self, basis):
        y = DinoV2RegEncoder(StubBackbone("const"), basis, precision="fp32").encode_full(_x(1))
        for i, v in enumerate([3, 6, 9, 12]):
            assert bool((y[..., i * 768:(i + 1) * 768] == v).all())

    def test_row_major_layout(self, basis):
        y = DinoV2RegEncoder(StubBackbone("index"), basis, precision="fp32").encode_full(_x(1))
        r, c = torch.meshgrid(torch.arange(32), torch.arange(64), indexing="ij")
        assert torch.equal(y[0, :, :, 0], (r * 64 + c).float())

    def test_pca_path(self, basis):
        enc = DinoV2RegEncoder(StubBackbone(), basis, precision="fp32")
        x = _x()
        z = enc.encode_pca(x)
        assert tuple(z.shape) == (2, 32, 64, 1152)
        assert float((z - basis.transform(enc.encode_full(x))).abs().max()) <= 1e-5
        assert tuple(enc.encode_pca(_x(1, 448, 560)).shape) == (1, 32, 40, 1152)

    def test_inverse(self, basis):
        enc = DinoV2RegEncoder(StubBackbone(), basis, precision="fp32")
        x = _x()
        rec = enc.pca_inverse(enc.encode_pca(x))
        assert tuple(rec.shape) == (2, 32, 64, 3072)
        f = enc.encode_full(x)
        assert torch.allclose(rec, basis.inverse(basis.transform(f)), atol=1e-5)

    def test_frozen(self, basis):
        enc = DinoV2RegEncoder(StubBackbone(), basis)
        assert all(not p.requires_grad for p in enc.parameters()) and enc.training is False
        assert enc.train() is enc and enc.training is False and enc.backbone.training is False
        enc.train(True)
        assert enc.training is False and enc.backbone.training is False

    def test_no_graph_usable_downstream(self, basis):
        enc = DinoV2RegEncoder(StubBackbone(), basis, precision="fp32")
        y = enc.encode_pca(_x(1).requires_grad_())
        assert y.requires_grad is False
        nn.Linear(1152, 4)(y).sum().backward()  # fails on an inference tensor

    def test_input_validation(self, basis):
        enc = DinoV2RegEncoder(StubBackbone(), basis, precision="fp32")
        for bad in [torch.zeros(3, 448, 896), torch.zeros(1, 1, 448, 896), torch.zeros(1, 3, 450, 896)]:
            with pytest.raises(ValueError):
                enc.encode_full(bad)
        with pytest.raises(ValueError):
            DinoV2RegEncoder(StubBackbone(), basis, precision="bf16")

    def test_cpu_fp16_fallback(self, basis):
        stub = StubBackbone()
        enc16 = DinoV2RegEncoder(stub, basis, precision="fp16")
        enc32 = DinoV2RegEncoder(stub, basis, precision="fp32")
        x = _x(1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = enc16.encode_pca(x)
            enc16.encode_pca(x)
        assert sum(issubclass(i.category, UserWarning) for i in w) == 1
        assert torch.equal(a, enc32.encode_pca(x))

    def test_fp32_ignores_outer_autocast(self, basis):
        enc = DinoV2RegEncoder(StubBackbone(), basis, precision="fp32")
        x = _x(1)
        plain = enc.encode_pca(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inside = enc.encode_pca(x)
        assert torch.equal(plain, inside)

    def test_from_assets_without_weights(self, irwm_env):
        with pytest.raises(FileNotFoundError, match="download.py"):
            DinoV2RegEncoder.from_assets()


# --------------------------------------------------------------------------- Group 4
class TestConfigAndRefactor:
    def test_encoder_tags(self):
        assert encoder_tag("stretch") == "dinov2_vitb14reg_448x896_l3-6-9-12_pca1152"
        assert encoder_tag("crop") == "dinov2_vitb14reg_448x896crop_l3-6-9-12_pca1152"
        assert encoder_tag("native") == "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152"

    def test_hydra_group(self, irwm_env):
        from hydra import compose, initialize_config_module
        from omegaconf import OmegaConf

        with initialize_config_module(config_module="irwm.conf", version_base="1.3"):
            cfg = compose("irwm_base")
        assert OmegaConf.to_container(cfg.encoder) == {"policy": "native", "precision": "fp16",
                                                       "hub_version": "dinof_release"}

    def test_moved_helpers(self):
        from irwm.worldmodel.reference import one_step, pca_metrics  # noqa: F401

        audit = (REPO / "projects" / "dinof_audit" / "audit.py").read_text(encoding="utf-8")
        assert "def one_step" not in audit and "def pca_metrics" not in audit
        gt = torch.randn(4, 8, 16)
        m = pca_metrics(gt + 1, gt, beta=1.0)
        assert m["smoothl1"] == pytest.approx(0.5) and m["mse"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- Group 5 (real weights)
def _weights_present() -> bool:
    from irwm.utils.paths import load_env
    from irwm.worldmodel import assets as A

    load_env()
    try:
        paths = [A.dinof_dir() / A.DINOF_HIGHRES, A.dinof_dir() / A.DINOF_PCA, A.dinof_dir() / A.PCA_EXPORT,
                 A.dinov2_weights_path(), A.dinov2_code_dir() / "hubconf.py"]
    except KeyError:  # IRWM_CKPT_ROOT unset
        return False
    return all(Path(p).is_file() for p in paths)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


needs_weights = pytest.mark.skipif("not _weights_present()", reason="DINO-Foresight / DINOv2 weights not found")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def real():
    """Reference ``Dino_f`` + our fp32/fp16 encoders on one device, and the flattened demo clip."""
    from irwm.worldmodel import assets as A

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, _ = R.load_reference(A.dinof_dir() / A.DINOF_HIGHRES, pca_ckpt=A.dinof_dir() / A.DINOF_PCA,
                                device=device)
    enc32 = DinoV2RegEncoder.from_assets(precision="fp32", device=device)
    enc16 = DinoV2RegEncoder(enc32.backbone, enc32.pca, precision="fp16")
    frames = R.load_demo_frames((448, 896)).flatten(end_dim=1).to(device)  # [5, 3, 448, 896], never split
    return {"model": model, "enc32": enc32, "enc16": enc16, "x": frames, "device": device}


def _reference(model, x):
    with torch.no_grad():
        f = model.extract_features(x)
        z = model.pca_transform(f)
    return f.reshape(5, 32, 64, -1), z.reshape(5, 32, 64, -1)


@pytest.mark.data
@needs_weights
class TestRealWeights:
    def test_a_fp32_full_and_pca(self, real):
        f_ref, z_ref = _reference(real["model"], real["x"])
        d_full = _maxdiff(real["enc32"].encode_full(real["x"]), f_ref)
        d_pca = _maxdiff(real["enc32"].encode_pca(real["x"]), z_ref)
        print(f"\n[F4 Test A] device={real['device']} fp32 max|d| full={d_full:.3e} pca={d_pca:.3e}")
        assert d_full == 0 and d_pca == 0

    @needs_cuda
    def test_b_fp16_full_and_pca(self, real):
        with torch.autocast("cuda", dtype=torch.float16):
            f_ref, z_ref = _reference(real["model"], real["x"])
        d_full = _maxdiff(real["enc16"].encode_full(real["x"]), f_ref)
        d_pca = _maxdiff(real["enc16"].encode_pca(real["x"]), z_ref)
        print(f"\n[F4 Test B] gpu={torch.cuda.get_device_name(0)} fp16 max|d| full={d_full:.3e} pca={d_pca:.3e}")
        # Spec FR8: 0 required; <= 1e-2 accepted only with the differing op named in results.md.
        assert d_full <= 1e-2 and d_pca <= 1e-2
        if d_full or d_pca:
            pytest.xfail(f"Test B not bit-exact (full {d_full:.3e}, pca {d_pca:.3e}); name the op in results.md")

    @needs_cuda
    def test_c_fp16_drift(self, real):
        z16 = real["enc16"].encode_pca(real["x"])
        z32 = real["enc32"].encode_pca(real["x"])
        d = (z16 - z32).abs()
        cos = float(torch.nn.functional.cosine_similarity(z16, z32, dim=-1).mean())
        print(f"\n[F4 Test C] fp16 vs fp32 pca: max|d|={float(d.max()):.3e} mean|d|={float(d.mean()):.3e} "
              f"mean token cosine={cos:.6f}")

    @needs_cuda
    def test_determinism_fp16(self, real):
        a = real["enc16"].encode_pca(real["x"])
        b = real["enc16"].encode_pca(real["x"])
        print(f"\n[F4 determinism] fp16 max|d|={_maxdiff(a, b):.3e}")
        assert torch.equal(a, b)

    def test_native_grid_end_to_end(self, real):
        x, _ = Preprocess("native", "thermal")(_img(512, 640, False))
        z = real["enc16"].encode_pca(x.unsqueeze(0).to(real["device"]))
        assert tuple(z.shape) == (1, 32, 40, 1152) and bool(torch.isfinite(z).all())

    def test_assets_offline(self, real):
        assert real["enc32"].pca.sha256 != ""

    def test_register_variant(self, real):
        bb = real["enc32"].backbone
        assert bb.num_register_tokens == 4 and bb.embed_dim == 768
        with torch.no_grad():
            out = bb.get_intermediate_layers(real["x"][:1], n=[2, 5, 8, 11], reshape=False)
        assert all(o.shape[1] == 2048 for o in out)
