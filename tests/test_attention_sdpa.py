"""``irwm.features.encoder.set_attention``: SDPA swap of DINOv2 attention layers (CPU, no assets)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from irwm.features.encoder import set_attention


class _Attention(nn.Module):
    """Copy of ``dinov2/layers/attention.py`` ``Attention`` at ``dinof_release`` (85a24602)."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


def _backbone() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(_Attention(96, 12), nn.LayerNorm(96), _Attention(96, 12)).eval()


def test_sdpa_matches_plain_and_switches_back():
    model, x = _backbone(), torch.randn(2, 37, 96)
    with torch.no_grad():
        ref = model(x)
        assert set_attention(model, "sdpa") == 2
        out = model(x)
        assert set_attention(model, "plain") == 2
        back = model(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert torch.equal(back, ref)  # "plain" restores the class forward exactly


def test_rejects_unknown_impl_and_non_vit():
    with pytest.raises(ValueError, match="Unknown attention"):
        set_attention(_backbone(), "flash")
    with pytest.raises(ValueError, match="no DINOv2 attention"):
        set_attention(nn.Linear(4, 4), "sdpa")


def test_rejects_custom_scale():
    model = _backbone()
    model[0].scale = 0.5
    with pytest.raises(ValueError, match="scale"):
        set_attention(model, "sdpa")
