"""F1 · DINO-Foresight audit. See spec/2026-10-02-dino-foresight-audit/validation.md."""

from __future__ import annotations

import ast
import hashlib
import re
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from irwm.features.pca import PCABasis
from irwm.utils.paths import repo_root
from irwm.worldmodel import assets as A
from irwm.worldmodel import reference as R

REPO = repo_root()


# --------------------------------------------------------------------------- Group 1
class TestAssets:
    def test_paths(self, irwm_env):
        ck = irwm_env["ckpt"].resolve()
        assert A.dinof_dir() == ck / "dino_foresight" / "9174a31e"
        assert A.dinov2_code_dir("dinof_release") == ck / "torch_hub" / "dinov2@85a24602"
        assert A.dinov2_weights_path().name == "dinov2_vitb14_reg4_pretrain.pth"
        assert A.dinov2_weights_path().parent == ck / "torch_hub" / "checkpoints"
        with pytest.raises(KeyError):
            A.dinov2_code_dir("nope")

    def test_constants(self):
        assert A.DINOV2_HUB_PINNED in A.DINOV2_HUB_COMMITS
        for sha in [*A.DINOV2_HUB_COMMITS.values(), A.DINOF_HF_REVISION]:
            assert re.fullmatch(r"[0-9a-f]{40}", sha)
        assert len(A.DINOF_FILES) == 6 and A.DINOF_HIGHRES in A.DINOF_FILES and A.DINOF_PCA in A.DINOF_FILES

    def test_sha256_multi_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(A, "_CHUNK", 1000)
        data = bytes(range(256)) * 50  # 12.8 kB > several chunks
        p = tmp_path / "f.bin"
        p.write_bytes(data)
        assert A.sha256_file(p) == hashlib.sha256(data).hexdigest()

    def test_manifest_roundtrip_and_tamper(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"hello")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("world")
        A.write_manifest(tmp_path, [{"file": "a.bin", "source": "x"}, {"file": "sub/b.txt"}])
        assert A.verify_manifest(tmp_path) == []
        assert A.read_manifest(tmp_path)["a.bin"]["source"] == "x"
        (tmp_path / "a.bin").write_bytes(b"hellO")  # same size, different content
        assert A.verify_manifest(tmp_path) == ["a.bin"]
        (tmp_path / "sub" / "b.txt").unlink()
        assert A.verify_manifest(tmp_path) == ["a.bin", "sub/b.txt"]

    def test_missing_manifest(self, tmp_path):
        assert A.verify_manifest(tmp_path) == [A.MANIFEST]


# --------------------------------------------------------------------------- Group 2
def _synthetic_basis(d: int, k: int, seed: int = 0) -> tuple[PCABasis, dict]:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    t = {"mean": torch.randn(d, generator=g), "std": torch.rand(d, generator=g) + 0.5,
         "pca_mean": 0.1 * torch.randn(d, generator=g), "components": q[:k].contiguous()}
    return PCABasis(**t), t


class TestPCABasis:
    def test_transform_matches_formula(self):
        basis, t = _synthetic_basis(3072, 1152)
        for shape in [(2, 4, 3072), (3072,)]:
            x = torch.randn(*shape)
            expected = ((x - t["mean"]) / t["std"] - t["pca_mean"]) @ t["components"].T
            assert torch.allclose(basis.transform(x), expected, atol=1e-5)
        assert basis.dim == 3072 and basis.n_components == 1152

    def test_roundtrip_in_span(self):
        basis, _ = _synthetic_basis(3072, 1152)
        x = basis.inverse(torch.randn(5, 1152))  # lies in the affine span by construction
        assert torch.allclose(basis.inverse(basis.transform(x)), x, atol=1e-4)

    def test_full_rank_roundtrip_exact(self):
        basis, _ = _synthetic_basis(64, 64)
        x = torch.randn(10, 64)
        assert torch.allclose(basis.inverse(basis.transform(x)), x, atol=1e-5)

    def test_shape_validation_and_frozen(self):
        basis, _ = _synthetic_basis(64, 16)
        with pytest.raises(ValueError):
            basis.transform(torch.randn(3, 63))
        with pytest.raises(ValueError):
            basis.inverse(torch.randn(3, 17))
        assert all(not b.requires_grad for b in basis.buffers())
        assert list(basis.parameters()) == []
        with pytest.raises(ValueError):
            PCABasis(torch.zeros(5), torch.ones(64), torch.zeros(64), torch.zeros(16, 64))

    def test_load_roundtrip(self, tmp_path):
        basis, t = _synthetic_basis(64, 16)
        torch.save({**t, "source_sha256": "abc123"}, tmp_path / "pca.pt")
        loaded = PCABasis.load(tmp_path / "pca.pt")
        assert loaded.sha256 == "abc123"
        x = torch.randn(4, 64)
        assert torch.equal(loaded.transform(x), basis.transform(x))
        torch.save({"mean": t["mean"]}, tmp_path / "bad.pt")
        with pytest.raises(KeyError):
            PCABasis.load(tmp_path / "bad.pt")


# --------------------------------------------------------------------------- Group 3
def _saved_args() -> Namespace:
    return Namespace(img_size=(448, 896), eval_mode=True, eval_modality="segm", head_ckpt="/x/h.ckpt",
                     evaluate=True, vis_attn=False, pca_ckpt="/old/pca.pth", high_res_adapt=True,
                     d_layers=[2, 5, 8, 11], sequence_length=5)


class TestInferenceArgs:
    def test_train_py_defaults_parsed_from_source(self):
        d = R.train_py_defaults()
        # Values as written in third_party/DINO-Foresight/train.py
        assert d["sliding_window_inference"] is False and d["crop_feats"] is False
        assert d["sequence_length"] == 5 and d["beta_smoothl1"] == 0.1 and d["d_layers"] == [2, 5, 8, 11]
        assert d["img_size"] == (224, 448) and d["masking"] == "half_half" and d["pca_ckpt"] is None

    def test_missing_fields_filled_and_reported(self):
        saved = _saved_args()  # lacks e.g. sliding_window_inference, like the released checkpoint
        args, diff = R.inference_args(saved, pca_ckpt="p")
        assert args.sliding_window_inference is False
        assert diff["sliding_window_inference"] == (R.MISSING, False)
        assert args.sequence_length == 5 and "sequence_length" not in diff  # present: kept as saved
        assert not hasattr(saved, "sliding_window_inference")

    def test_overrides_and_no_mutation(self):
        saved = _saved_args()
        args, diff = R.inference_args(saved, pca_ckpt=Path("/new/pca.pth"), defaults={})
        assert (args.eval_mode, args.eval_modality, args.head_ckpt, args.evaluate, args.vis_attn) == \
            (False, None, None, False, False)
        assert args.pca_ckpt == str(Path("/new/pca.pth")) and args.high_res_adapt is False
        assert args.img_size == (448, 896) and args.d_layers == [2, 5, 8, 11]
        assert saved.eval_mode is True and saved.pca_ckpt == "/old/pca.pth"  # untouched
        assert set(diff) == {"eval_mode", "eval_modality", "head_ckpt", "evaluate", "pca_ckpt", "high_res_adapt"}
        assert diff["eval_mode"] == (True, False)

    def test_img_size_toggles_high_res_adapt(self):
        args, diff = R.inference_args(_saved_args(), pca_ckpt="p", img_size=(448, 560))
        assert args.img_size == (448, 560) and args.high_res_adapt is True
        assert diff["img_size"] == ((448, 896), (448, 560))
        args, _ = R.inference_args(_saved_args(), pca_ckpt="p", img_size=(448, 896))
        assert args.high_res_adapt is False

    def test_extra_overrides(self):
        args, diff = R.inference_args(_saved_args(), pca_ckpt="p", extra={"num_classes": 19}, defaults={})
        assert args.num_classes == 19 and diff["num_classes"] == (R.MISSING, 19)


# --------------------------------------------------------------------------- Group 4
class TestDemoInputs:
    def test_center_crop_box(self):
        assert R.center_crop_box(2048, 1024, (5, 4)) == (384, 0, 1280, 1024)
        assert R.center_crop_box(1280, 1024, (2, 1)) == (0, 192, 1280, 640)
        assert R.center_crop_box(2048, 1024, (2, 1)) == (0, 0, 2048, 1024)

    def test_demo_frames(self):
        paths = R.demo_frame_paths()
        assert [p.name.split("_")[2] for p in paths] == ["017089", "017092", "017095", "017098", "017101"]
        x = R.load_demo_frames((448, 896))
        assert x.shape == (1, 5, 3, 448, 896) and x.dtype == torch.float32
        ch_mean = x.mean(dim=(0, 1, 3, 4))
        assert bool((ch_mean.abs() < 2.5).all())

    def test_demo_frames_cropped(self):
        assert R.load_demo_frames((448, 560), crop_aspect=(5, 4)).shape == (1, 5, 3, 448, 560)
        # A chain of crops: 5:4 then 2:1 (R5 policy b) differs from the uncropped 2:1 frame.
        b = R.load_demo_frames((448, 896), crop_aspect=[(5, 4), (2, 1)])
        ref = R.load_demo_frames((448, 896))
        assert b.shape == ref.shape and not torch.allclose(b, ref)

    def test_no_heavy_toplevel_imports_in_lib(self):
        heavy = {"timm", "einops", "sklearn", "pytorch_lightning", "huggingface_hub"}
        for f in (REPO / "lib" / "irwm").rglob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only; function-level imports are lazy
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert not {n.split(".")[0] for n in names} & heavy, f"{f}: {names}"


# --------------------------------------------------------------------------- Group 5 (real weights)
@pytest.mark.data
class TestRealWeights:
    def test_manifests(self):
        from irwm.utils.paths import load_env

        load_env()
        assert A.verify_manifest(A.dinof_dir()) == []
        assert A.verify_manifest(A.dinov2_hub_dir()) == []

    def test_checkpoint_args(self):
        args = R.checkpoint_args(A.dinof_dir() / A.DINOF_HIGHRES)
        assert tuple(args.img_size) == (448, 896) and list(args.d_layers) == [2, 5, 8, 11]
        assert args.sequence_length == 5 and args.hidden_dim == 1152 and args.layers == 12
        assert args.masking == "simple_replace" and args.seperable_attention is True

    def test_pcabasis_matches_reference(self):
        model, _ = R.load_reference(A.dinof_dir() / A.DINOF_HIGHRES, pca_ckpt=A.dinof_dir() / A.DINOF_PCA)
        basis = PCABasis.load(A.dinof_dir() / A.PCA_EXPORT)
        x = torch.randn(2, 2048, 3072)
        z = torch.randn(1, 1, 32, 64, 1152)
        with torch.no_grad():
            assert torch.allclose(basis.transform(x), model.pca_transform(x), atol=1e-5)
            assert torch.allclose(basis.inverse(z), model.pca_inverse_transform(z), atol=1e-5)
