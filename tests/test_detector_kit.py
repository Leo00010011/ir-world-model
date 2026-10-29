"""Detector kit (projects/detector_kit): MSDA fallback without compiled detrex ops, configs, shell flags.

CPU only, no data or weights. The end-to-end runs (train.sh --quick, inference.sh) need FLIR and
the encoder weights and are recorded in projects/detector_kit/README.md instead.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from irwm.detection.detrex_adapter import MSDA_MODES, cuda_ops_available, select_msda

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "projects" / "detector_kit"


# --------------------------------------------------------------------------- MSDA selection
@pytest.fixture
def msda_module(monkeypatch: pytest.MonkeyPatch):
    """detrex's MSDA module; whatever a test patches into it is restored afterwards."""
    mod = importlib.import_module("detrex.layers.multi_scale_deform_attn")
    monkeypatch.setattr(mod, "MultiScaleDeformableAttnFunction", mod.MultiScaleDeformableAttnFunction)
    return mod


class TestSelectMsda:
    def test_modes(self):
        assert MSDA_MODES == ("auto", "cuda", "pytorch")

    def test_unknown_mode(self, msda_module):
        with pytest.raises(ValueError, match="msda must be one of"):
            select_msda("triton")

    def test_pytorch_swaps_the_function(self, msda_module):
        original = msda_module.MultiScaleDeformableAttnFunction
        assert select_msda("pytorch") == "pytorch"
        assert msda_module.MultiScaleDeformableAttnFunction is not original

    def test_auto_follows_cuda_ops(self, msda_module):
        assert select_msda("auto") == ("cuda" if cuda_ops_available() else "pytorch")

    @pytest.mark.skipif(cuda_ops_available(), reason="checks the no-CUDA-ops error")
    def test_cuda_without_ops_raises(self, msda_module):
        with pytest.raises(RuntimeError, match="compiled CUDA ops are not available"):
            select_msda("cuda")

    def test_swapped_layer_matches_cpu_path(self, msda_module):
        """After the swap the layer gives detrex's CPU output (on CUDA when there is a GPU)."""
        from detrex.layers import MultiScaleDeformableAttention

        torch.manual_seed(0)
        attn = MultiScaleDeformableAttention(embed_dim=32, num_heads=4, num_levels=1, num_points=2,
                                             batch_first=True).eval()
        args = dict(query=torch.randn(1, 4, 32), value=torch.randn(1, 16, 32),
                    reference_points=torch.rand(1, 4, 1, 2), spatial_shapes=torch.tensor([[4, 4]]),
                    level_start_index=torch.tensor([0]))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            ref = attn(**args)
            select_msda("pytorch")
            out = attn.to(device)(**{k: v.to(device) for k, v in args.items()}).cpu()
        torch.testing.assert_close(out, ref)


def test_model_builds_without_detrex_ops():
    """No loadable detrex._C: the real MSDA class survives, the head builds and runs (subprocess)."""
    code = (
        "import importlib.util, sys\n"
        "_spec = importlib.util.spec_from_file_location\n"
        "def _no_ops(name, *a, **k):  # detrex/_C*.so fails to load, as when detrex was never compiled\n"
        "    if name == 'detrex._C':\n"
        "        raise ImportError('simulated: detrex built without ops')\n"
        "    return _spec(name, *a, **k)\n"
        "importlib.util.spec_from_file_location = _no_ops\n"
        "import torch\n"
        "from irwm.detection.detrex_adapter import cuda_ops_available, select_msda\n"
        "assert not hasattr(sys.modules['detrex._C'], 'ms_deform_attn_forward')\n"
        "assert cuda_ops_available() is False\n"
        "assert select_msda('auto') == 'pytorch'\n"
        "from irwm.detection import DINODetector\n"
        "det = DINODetector(overrides={'transformer.encoder.num_layers': 1, 'transformer.decoder.num_layers': 1,"
        " 'num_queries': 100, 'dn_number': 10}).eval()\n"
        "with torch.no_grad():\n"
        "    out = det(torch.randn(1, 1152, 8, 10))\n"
        "assert out[0]['boxes'].shape == (300, 4), out[0]['boxes'].shape\n"
        "print('ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip().endswith("ok")


# --------------------------------------------------------------------------- configs
def _compose(name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(KIT / "conf"), version_base="1.3"):
        return compose(name, overrides=overrides)


class TestConfigs:
    def test_train_matches_f8_recipe(self, irwm_env):
        kit = _compose("train", ["modality=rgb"])
        with initialize_config_dir(config_dir=str(REPO / "projects" / "detection" / "conf"), version_base="1.3"):
            f8 = compose("train", ["experiment=det_rgb"])
        assert kit.data == f8.data
        assert {k: v for k, v in kit.optim.items() if k != "lr_drop_fraction"} == \
            {k: v for k, v in f8.optim.items() if k != "lr_drop_epoch"}
        assert round(kit.optim.lr_drop_fraction * kit.trainer.max_epochs) == f8.optim.lr_drop_epoch
        assert {k: v for k, v in kit.trainer.items() if not k.startswith("limit_")} == dict(f8.trainer)
        assert kit.encoder == f8.encoder
        assert kit.project == "detector_kit"

    def test_train_needs_modality(self, irwm_env):
        from omegaconf.errors import MissingMandatoryValue

        with pytest.raises(MissingMandatoryValue):
            _ = _compose("train", []).modality

    def test_wandb_off_by_default(self, irwm_env, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        assert _compose("train", ["modality=thermal"]).wandb.mode == "disabled"

    def test_inference_defaults(self, irwm_env):
        cfg = _compose("inference", [])
        assert cfg.head is None and cfg.ckpt is None and cfg.images is None and cfg.split is None
        assert cfg.msda == "auto" and cfg.device == "auto"


# --------------------------------------------------------------------------- shell scripts
def _sh(script: str, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRWM_")}
    return subprocess.run(["bash", str(KIT / script), *args], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=60)


class TestShellFlags:
    """Argument errors are reported before any env or download step."""

    @pytest.mark.parametrize("script", ["train.sh", "inference.sh"])
    def test_help(self, script):
        res = _sh(script, "--help")
        assert res.returncode == 0 and "Anything after `--`" in res.stdout

    def test_train_needs_modality(self):
        res = _sh("train.sh", "--epochs", "3")
        assert res.returncode != 0 and "--modality rgb|thermal is required" in res.stderr

    def test_train_unknown_flag(self):
        res = _sh("train.sh", "--modality", "rgb", "--epoch", "3")
        assert res.returncode != 0 and "unknown option '--epoch'" in res.stderr

    def test_inference_needs_model_and_input(self, tmp_path):
        assert "choose a model" in _sh("inference.sh", "--images", str(tmp_path)).stderr
        assert "choose an input" in _sh("inference.sh", "--head", "det_ir").stderr
        assert "not both" in _sh("inference.sh", "--head", "det_ir", "--split", "val",
                                 "--images", str(tmp_path)).stderr

    def test_inference_missing_folder(self, tmp_path):
        res = _sh("inference.sh", "--head", "det_ir", "--images", str(tmp_path / "nope"))
        assert res.returncode != 0 and "is not a folder" in res.stderr
