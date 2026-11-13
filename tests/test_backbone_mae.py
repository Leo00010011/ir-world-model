"""``mae`` backbone variant: end-to-end DINO-ViTDet from MAE ViT-B (irwm.detection.vitdet).

Default tests run on CPU with a tiny ViT (depth 2, width 64) and a reduced DINO head.
``data``: the real pinned MAE checkpoint (``$IRWM_CKPT_ROOT/pretrained/mae``).
``gpu``: one real-size 512×640 train step with activation checkpointing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from irwm.detection import mae_assets
from irwm.detection.detrex_adapter import select_msda
from irwm.detection.vitdet import (VITDET_META_ARCH, ViTDetDINO, build_vitdet_config, load_mae_weights,
                                   map_mae_state_dict, vitdet_from_config, vitdet_param_groups)
from irwm.features.preprocess import POLICY_SIZE, Preprocess
from irwm.utils.paths import repo_root
from irwm.worldmodel.reference import IMAGENET_MEAN, IMAGENET_STD

CONF = repo_root() / "lib" / "irwm" / "conf"
TINY = {"backbone.net.depth": 2, "backbone.net.embed_dim": 64, "backbone.net.num_heads": 2,
        "backbone.net.window_size": 4, "backbone.net.window_block_indexes": [0],
        "backbone.net.drop_path_rate": 0.0, "num_queries": 150, "dn_number": 5,
        "transformer.encoder.num_layers": 1, "transformer.decoder.num_layers": 2}
HW = (128, 192)


def _tiny(**kw) -> ViTDetDINO:
    torch.manual_seed(0)
    kw.setdefault("use_act_checkpoint", False)
    return ViTDetDINO(image_hw=HW, overrides=TINY, **kw)


def _fake_mae(vit: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A synthetic MAE-style checkpoint for ``vit``: its own keys minus rel-pos, plus MAE extras."""
    state = {k: torch.randn_like(v) for k, v in vit.state_dict().items() if "rel_pos" not in k}
    dim = state["pos_embed"].shape[-1]
    state.update({"cls_token": torch.zeros(1, 1, dim), "norm.weight": torch.ones(dim),
                  "norm.bias": torch.zeros(dim)})
    return state


@pytest.fixture(scope="module")
def tiny() -> ViTDetDINO:
    return _tiny().eval()


# --------------------------------------------------------------------------- config and geometry
class TestConfig:
    def test_stock_deviations(self):
        cfg = build_vitdet_config()
        assert cfg.num_classes == 2 and cfg.backbone.square_pad == 0
        assert cfg.backbone.net.img_size == 640 and cfg.backbone.net.use_act_checkpoint is True
        assert cfg.backbone.net.patch_size == 16 and cfg.backbone.net.depth == 12
        assert list(cfg.backbone.net.window_block_indexes) == [0, 1, 3, 4, 6, 7, 9, 10]
        assert cfg.backbone.net.use_rel_pos is True and cfg.backbone.net.window_size == 14
        assert "loss_class_4" in cfg.criterion.weight_dict  # 6 decoder layers → aux 0..4 + enc
        assert "loss_class_1" not in build_vitdet_config(
            overrides={"transformer.decoder.num_layers": 2}).criterion.weight_dict

    def test_override_errors(self):
        with pytest.raises(KeyError):
            build_vitdet_config(overrides={"backbone.net.not_a_key": 1})
        for key in ("num_classes", "backbone.square_pad", "backbone.net.img_size"):
            with pytest.raises(ValueError):
                build_vitdet_config(overrides={key: 1})

    def test_global_rel_pos_matches_grid(self, tiny):
        gh, gw = HW[0] // 16, HW[1] // 16
        for i, block in enumerate(tiny.vit.blocks):
            if i in TINY["backbone.net.window_block_indexes"]:
                assert block.attn.rel_pos_h.shape[0] == 2 * 4 - 1  # window 4
            else:
                assert block.attn.rel_pos_h.shape[0] == 2 * gh - 1
                assert block.attn.rel_pos_w.shape[0] == 2 * gw - 1

    def test_full_size_param_count(self):
        m = ViTDetDINO(use_act_checkpoint=False)
        vit = sum(p.numel() for p in m.vit.parameters())
        total = sum(p.numel() for p in m.parameters())
        assert 85.7e6 < vit < 85.9e6 and 107.8e6 < total < 108.0e6
        assert m.vit.blocks[2].attn.rel_pos_h.shape == (63, 64)
        assert m.vit.blocks[2].attn.rel_pos_w.shape == (79, 64)
        assert m.vit.pos_embed.shape == (1, 197, 768)

    def test_bad_sizes(self, tiny):
        with pytest.raises(ValueError):
            ViTDetDINO(image_hw=(500, 640), overrides=TINY)
        with pytest.raises(ValueError):
            tiny(torch.zeros(1, 3, 128, 200))
        with pytest.raises(ValueError):
            tiny(torch.zeros(1, 1, 128, 192))


# --------------------------------------------------------------------------- forward contract
class TestForward:
    def test_eval_contract(self, tiny):
        with torch.no_grad():
            out = tiny(torch.randn(2, 3, *HW))
        assert len(out) == 2
        for o in out:
            assert set(o) == {"boxes", "scores", "labels"}
            assert o["boxes"].shape == (300, 4) and o["scores"].shape == (300,)
            assert o["labels"].dtype == torch.int64 and int(o["labels"].max()) < 2
            assert float(o["boxes"][:, 2].max()) <= HW[1] + 1e-3 and float(o["boxes"][:, 3].max()) <= HW[0] + 1e-3

    def test_train_needs_cuda_and_targets(self, tiny):
        tiny.train()
        try:
            with pytest.raises(ValueError):
                tiny(torch.randn(1, 3, *HW))
            if not torch.cuda.is_available():
                t = [{"boxes": torch.tensor([[1.0, 1.0, 20.0, 20.0]]), "labels": torch.tensor([0])}]
                with pytest.raises(RuntimeError):
                    tiny(torch.randn(1, 3, *HW), t)
        finally:
            tiny.eval()

    def test_act_checkpoint_keeps_keys_and_outputs(self, tiny):
        ckpt = _tiny(use_act_checkpoint=True).eval()
        assert list(ckpt.state_dict()) == list(tiny.state_dict())
        ckpt.load_state_dict(tiny.state_dict())
        x = torch.randn(1, 3, *HW)
        with torch.no_grad():
            a, b = tiny(x)[0], ckpt(x)[0]
        assert torch.allclose(a["scores"], b["scores"], atol=1e-5)

    def test_state_dict_roundtrip(self, tiny, tmp_path):
        assert all(k.startswith("detr.") for k in tiny.state_dict())
        assert any(k.startswith("detr.backbone.net.blocks.") for k in tiny.state_dict())
        torch.save(tiny.state_dict(), tmp_path / "m.pt")
        other = _tiny()
        other.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
        other.eval()
        x = torch.randn(1, 3, *HW)
        with torch.no_grad():
            assert torch.equal(tiny(x)[0]["boxes"], other(x)[0]["boxes"])


# --------------------------------------------------------------------------- optimizer groups
class TestParamGroups:
    @staticmethod
    def _by_param(groups):
        out = {}
        for g in groups:
            for p in g["params"]:
                assert id(p) not in out, "parameter in two groups"
                out[id(p)] = (g["lr"], g["weight_decay"])
        return out

    def test_layer_decay(self, tiny):
        groups = vitdet_param_groups(tiny, base_lr=1e-4, weight_decay=1e-4)
        by = self._by_param(groups)
        names = dict(tiny.detr.named_parameters())
        assert set(by) == {id(p) for p in names.values()}
        d = tiny.depth
        expect_lr = {"backbone.net.patch_embed.proj.weight": 0.7 ** (d + 1),
                     "backbone.net.pos_embed": 0.7 ** (d + 1),
                     "backbone.net.blocks.0.attn.qkv.weight": 0.7 ** d,
                     "backbone.net.blocks.1.mlp.fc2.weight": 0.7 ** (d - 1)}
        for n, f in expect_lr.items():
            assert by[id(names[n])][0] == pytest.approx(1e-4 * f)
        head = [n for n in names if not n.startswith("backbone.net.")]
        assert all(by[id(names[n])][0] == pytest.approx(1e-4) for n in head)  # pyramid + DINO head
        # weight decay: 0 on torch norms and pos_embed; biases and rel-pos keep it (detectron2)
        assert by[id(names["backbone.net.pos_embed"])][1] == 0.0
        assert by[id(names["backbone.net.blocks.0.norm1.weight"])][1] == 0.0
        assert by[id(names["backbone.net.blocks.0.norm1.bias"])][1] == 0.0
        assert by[id(names["backbone.net.blocks.0.attn.qkv.bias"])][1] == 1e-4
        assert by[id(names["backbone.net.blocks.1.attn.rel_pos_h"])][1] == 1e-4
        torch.optim.AdamW(groups, lr=1e-4)  # accepted as-is

    def test_flat(self, tiny):
        by = self._by_param(vitdet_param_groups(tiny, lr_schedule="flat", backbone_lr_factor=0.1))
        for n, p in tiny.detr.named_parameters():
            assert by[id(p)][0] == pytest.approx(1e-5 if "backbone" in n else 1e-4)
        with pytest.raises(ValueError):
            vitdet_param_groups(tiny, lr_schedule="nope")


# --------------------------------------------------------------------------- MAE loading (synthetic)
class TestMaeMapping:
    def test_map_and_load(self, tmp_path):
        m = _tiny()
        state = _fake_mae(m.vit)
        mapped, dropped = map_mae_state_dict(state)
        assert dropped == ["cls_token", "norm.bias", "norm.weight"]
        torch.save({"model": state}, tmp_path / "mae.pth")
        missing, unexpected = load_mae_weights(m, tmp_path / "mae.pth", verify=False)
        assert all(k.endswith((".attn.rel_pos_h", ".attn.rel_pos_w")) for k in missing)
        assert len(missing) == 2 * TINY["backbone.net.depth"]
        assert unexpected == dropped
        assert torch.equal(m.vit.blocks[1].mlp.fc1.weight, state["blocks.1.mlp.fc1.weight"])
        assert torch.equal(m.vit.pos_embed, state["pos_embed"])

    def test_bad_checkpoint_raises(self, tmp_path):
        m = _tiny()
        state = _fake_mae(m.vit)
        del state["blocks.0.attn.qkv.weight"]
        state["something.else"] = torch.zeros(1)
        torch.save(state, tmp_path / "bad.pth")
        with pytest.raises(RuntimeError, match="does not fit"):
            load_mae_weights(m, tmp_path / "bad.pth", verify=False)

    def test_verify_rejects_wrong_file(self, tmp_path):
        p = tmp_path / "x.pth"
        p.write_bytes(b"0" * 10)
        with pytest.raises(ValueError):
            mae_assets.verify_mae_weights(p)
        with pytest.raises(FileNotFoundError):
            mae_assets.verify_mae_weights(tmp_path / "missing.pth")

    def test_asset_location(self, irwm_env):
        assert mae_assets.mae_weights_path() == irwm_env["ckpt"] / "pretrained" / "mae" / "mae_pretrain_vit_base.pth"


# --------------------------------------------------------------------------- Hydra config + entry point
class TestHydraConfig:
    def test_yaml_and_from_config(self):
        cfg = OmegaConf.load(CONF / "detector" / "vitdet_mae.yaml")
        assert cfg.arch == VITDET_META_ARCH == "vitdet_mae"
        assert cfg.init == "mae" and cfg.num_classes == 2 and list(cfg.image_hw) == [512, 640]
        assert tuple(cfg.image_hw) == POLICY_SIZE[cfg.preprocess_policy]
        assert cfg.optim.lr_schedule == "layer_decay" and cfg.optim.layer_decay == 0.7
        small = OmegaConf.merge(cfg, {"init": "none", "image_hw": list(HW), "use_act_checkpoint": False,
                                      "overrides": TINY})
        m = vitdet_from_config(small)
        assert isinstance(m, ViTDetDINO) and m.depth == 2 and m.image_hw == HW
        vitdet_param_groups(m, **small.optim)
        with pytest.raises(ValueError):
            vitdet_from_config(OmegaConf.merge(small, {"init": "imagenet"}))

    def test_from_config_mae_init(self, tmp_path, monkeypatch):
        state = _fake_mae(_tiny().vit)
        torch.save({"model": state}, tmp_path / "mae.pth")
        cfg = {"init": "mae", "init_path": str(tmp_path / "mae.pth"), "image_hw": list(HW),
               "use_act_checkpoint": False, "overrides": TINY}
        with pytest.raises(ValueError, match="SHA-256"):  # a non-pinned file is refused
            vitdet_from_config(cfg)
        monkeypatch.setattr("irwm.detection.vitdet.verify_mae_weights", lambda p: Path(p))
        m = vitdet_from_config(cfg)
        assert torch.equal(m.vit.blocks[0].attn.qkv.weight, state["blocks.0.attn.qkv.weight"])


# --------------------------------------------------------------------------- preprocess policy
class TestNative640:
    def test_identity_on_aligned_flir(self):
        img = np.random.default_rng(0).integers(0, 256, (512, 640), dtype=np.uint8)
        x, geom = Preprocess("native640", "thermal")(img)
        assert tuple(x.shape) == (3, 512, 640)
        assert geom.crop_box == (0, 0, 640, 512) and geom.out_size == (640, 512)
        raw = x * torch.tensor(IMAGENET_STD)[:, None, None] + torch.tensor(IMAGENET_MEAN)[:, None, None]
        assert torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[1], raw[2], atol=1e-5)
        assert torch.allclose(raw[0], torch.from_numpy(img).float() / 255, atol=1e-5)  # identity resize
        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        assert np.allclose(geom.apply_boxes(boxes)[0], boxes)

    def test_other_aspect_and_existing_policies(self):
        img = np.zeros((1024, 1224, 3), dtype=np.uint8)
        x, geom = Preprocess("native640", "rgb")(img)
        assert tuple(x.shape) == (3, 512, 640)
        assert geom.crop_box[2:] == (1224, 979)  # centre 5:4 band of a 1224x1024 frame
        assert POLICY_SIZE["stretch"] == (448, 896) and POLICY_SIZE["crop"] == (448, 896)
        assert POLICY_SIZE["native"] == (448, 560)


# --------------------------------------------------------------------------- real weights
@pytest.mark.data
def test_real_mae_weights_load():
    path = mae_assets.verify_mae_weights()
    m = ViTDetDINO(use_act_checkpoint=False)
    missing, unexpected = load_mae_weights(m)
    assert len(missing) == 24 and all("rel_pos" in k for k in missing)
    assert unexpected == ["cls_token", "norm.bias", "norm.weight"]
    ref = torch.load(path, map_location="cpu", weights_only=True)["model"]
    assert torch.equal(m.vit.blocks[11].attn.qkv.weight, ref["blocks.11.attn.qkv.weight"])
    assert sum(p.numel() for p in m.vit.parameters()) == pytest.approx(85.8e6, rel=2e-3)


# --------------------------------------------------------------------------- GPU train step
@pytest.mark.gpu
def test_gpu_train_step_real_size():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    msda = select_msda("auto")
    hw = tuple(int(v) for v in os.environ.get("IRWM_MAE_GPU_HW", "512,640").split(","))
    torch.manual_seed(0)
    m = ViTDetDINO(image_hw=hw, use_act_checkpoint=True)
    if mae_assets.mae_weights_path().is_file():
        load_mae_weights(m)
    m = m.cuda().train()
    opt = torch.optim.AdamW(vitdet_param_groups(m), lr=1e-4)
    images = torch.randn(1, 3, *hw, device="cuda")
    targets = [{"boxes": torch.tensor([[50.0, 60.0, 150.0, 260.0], [300.0, 200.0, 420.0, 280.0]], device="cuda"),
                "labels": torch.tensor([0, 1], device="cuda")}]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    losses = m(images, targets)
    loss = sum(losses.values())
    loss.backward()
    grad = m.vit.blocks[0].attn.qkv.weight.grad
    torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1)
    opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nmae gpu step: {hw} msda={msda} {time.perf_counter() - t0:.2f}s peak {peak:.2f} GiB "
          f"loss {float(loss):.3f} on {torch.cuda.get_device_name()}")
    assert torch.isfinite(loss) and grad is not None and float(grad.abs().sum()) > 0
