"""Frozen DINOv2-Reg ViT-B/14 (``dinov2_vitb14_reg``) + DINO-Foresight PCA feature encoder.

Reproduces the reference extraction exactly (F1 ``audit.md`` (b)):

    Dino_f.extract_features: get_intermediate_layers(x, n=[2, 5, 8, 11], reshape=False)
        → final norm on every block, CLS + 4 registers dropped by the hub code,
          concatenated → [B, h·w, 3072]                    (src/dino_f.py:285-297)
    Dino_f.pca_transform: standardise, subtract pca_mean, project → [B, h·w, 1152]
    rearrange 'b (h w) c -> b h w c'                       (src/dino_f.py:407)

Inputs are normalised ``[B, 3, H, W]`` batches from ``irwm.features.preprocess.Preprocess``.
Default precision is fp16 autocast over backbone *and* PCA, as Lightning ``16-mixed`` does in
every reference command. The module is frozen: always in ``eval()``, no parameter needs
grad, and encoding runs under ``torch.no_grad()`` (not ``inference_mode``, so a trainable
head can consume the outputs). Spec: spec/2026-10-16-dinov2-reg-encoder-wrapper (FR4-FR7, FR9).
"""

from __future__ import annotations

import contextlib
import warnings
from typing import ContextManager, Literal

import torch
from torch import Tensor, nn

from irwm.features.pca import PCABasis
from irwm.features.preprocess import Policy
from irwm.worldmodel.assets import DINOV2_HUB_PINNED, PCA_EXPORT, dinof_dir, dinov2_code_dir, dinov2_weights_path

Precision = Literal["fp16", "fp32"]
PATCH = 14
_TAG_RES: dict[str, str] = {"stretch": "448x896", "crop": "448x896crop", "native": "448x560"}


def encoder_tag(policy: Policy) -> str:
    """Cache/run tag of the encoder under ``policy`` (TechStack §4.2 for ``stretch``)."""
    if policy not in _TAG_RES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {list(_TAG_RES)}")
    return f"dinov2_vitb14reg_{_TAG_RES[policy]}_l3-6-9-12_pca1152"


class DinoV2RegEncoder(nn.Module):
    """``[B, 3, H, W]`` normalised images → channels-last DINOv2-Reg features (3072-d or PCA 1152-d)."""

    LAYERS: tuple[int, ...] = (2, 5, 8, 11)  # 0-indexed: blocks 3/6/9/12

    def __init__(self, backbone: nn.Module, pca: PCABasis, *, precision: Precision = "fp16") -> None:
        super().__init__()
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"Unknown precision {precision!r}; expected 'fp16' or 'fp32'")
        self.backbone = backbone
        self.pca = pca
        self.precision = precision
        self._warned_cpu = False
        self.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_assets(
        cls,
        *,
        precision: str = "fp16",
        hub_version: str = DINOV2_HUB_PINNED,
        device: str | torch.device = "cpu",
    ) -> "DinoV2RegEncoder":
        """Build from the pinned offline artefacts in ``$IRWM_CKPT_ROOT`` (no network)."""
        from irwm.worldmodel.reference import load_dinov2_backbone

        pca_path = dinof_dir() / PCA_EXPORT
        for path in (dinov2_code_dir(hub_version) / "hubconf.py", dinov2_weights_path(), pca_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found; run projects/dinof_audit/download.py "
                    f"(and pca_export.py for {PCA_EXPORT})"
                )
        enc = cls(load_dinov2_backbone(hub_version), PCABasis.load(pca_path), precision=precision)  # type: ignore[arg-type]
        return enc.to(device)

    def train(self, mode: bool = True) -> "DinoV2RegEncoder":
        """Frozen: stays in eval mode whatever ``mode`` is (Lightning calls ``.train()`` every epoch)."""
        return super().train(False)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _check(x: Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], got {list(x.shape)}")
        if x.shape[2] % PATCH or x.shape[3] % PATCH:
            raise ValueError(f"H and W must be multiples of {PATCH}, got {list(x.shape[2:])}")

    def _precision_ctx(self, x: Tensor) -> ContextManager:
        if self.precision == "fp32":
            return torch.autocast(x.device.type, enabled=False)
        if x.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        if not self._warned_cpu:
            warnings.warn("DinoV2RegEncoder: precision='fp16' on CPU runs in fp32 "
                          "(the reference uses CUDA fp16 autocast only)", UserWarning, stacklevel=3)
            self._warned_cpu = True
        return contextlib.nullcontext()

    def _features(self, x: Tensor, pca: bool) -> Tensor:
        self._check(x)
        h, w = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        with torch.no_grad(), self._precision_ctx(x):
            f = torch.cat(self.backbone.get_intermediate_layers(x, n=list(self.LAYERS), reshape=False, norm=True), dim=-1)
            z = self.pca.transform(f) if pca else f
        return z.float().reshape(x.shape[0], h, w, -1)  # row-major tokens, as einops 'b (h w) c -> b h w c'

    # ------------------------------------------------------------------ public
    def encode_full(self, x: Tensor) -> Tensor:
        """``[B, 3, H, W]`` → ``float32 [B, H/14, W/14, 3072]`` (blocks 3/6/9/12 concatenated)."""
        return self._features(x, pca=False)

    def encode_pca(self, x: Tensor) -> Tensor:
        """``[B, 3, H, W]`` → ``float32 [B, H/14, W/14, 1152]`` PCA coefficients."""
        return self._features(x, pca=True)

    def pca_inverse(self, z: Tensor) -> Tensor:
        """``[..., 1152]`` → ``[..., 3072]``, in fp32 with autocast disabled."""
        with torch.no_grad(), torch.autocast(z.device.type, enabled=False):
            return self.pca.inverse(z.float())
