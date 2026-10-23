"""DINO-DETR head on frozen DINOv2-Reg features: detrex's ``dino_vitdet`` with the ViT swapped out.

Spec: spec/2026-10-23-dino-detr-head-port (FR5-FR7). Every hyperparameter comes from detrex's own
``dino_vitdet.py`` (+ ``dino_r50.py``), loaded with ``LazyConfig``; this module only replaces the
backbone's ViT by ``FeatureStem`` and lists the deviations D1-D8 of plan.md where they are made.
Training needs CUDA: detrex's denoising code hard-codes ``.cuda()`` (``dino.py:370-458``).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from detectron2.config import LazyCall, instantiate
from detectron2.structures import Boxes, ImageList, Instances
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from irwm.data.classes import CLASS_NAMES
from irwm.detection.detrex_adapter import ensure_detrex_projects, load_dino_vitdet_config
from irwm.detection.stem import FeatureStem

ensure_detrex_projects()
DINO = importlib.import_module("projects.dino.modeling").DINO

PATCH = 14  # DINOv2 token pitch in encoder-input pixels; default image_size = 14 * grid
_BASE_LOSSES = {"loss_class", "loss_bbox", "loss_giou", "loss_class_dn", "loss_bbox_dn", "loss_giou_dn"}
_SET_BY_ARGS = ("num_classes", "device")


class FeatureDINO(DINO):
    """detrex's DINO fed with feature maps: no pixel normalisation, no padding (D3)."""

    def preprocess_image(self, batched_inputs: list[dict]) -> ImageList:
        return ImageList.from_tensors([x["image"].to(self.device) for x in batched_inputs])


def build_dino_config(*, in_channels: int = 1152, stem_dim: int = 768, num_classes: int = 6,
                      overrides: Mapping[str, Any] | None = None) -> DictConfig:
    """detrex's ``dino_vitdet`` model config with ``FeatureStem`` as ``backbone.net``.

    ``overrides`` are dotted keys relative to ``model`` (``{"transformer.encoder.num_layers": 2}``).
    Unknown keys raise ``KeyError``; keys set by the arguments raise ``ValueError``.
    """
    model = load_dino_vitdet_config().model
    model._target_ = FeatureDINO
    # D1. out_feature must be passed explicitly: LazyCall records only given arguments, and the
    # pyramid's in_feature="${.net.out_feature}" (dino_vitdet.py:42) interpolates it.
    model.backbone.net = LazyCall(FeatureStem)(in_channels=in_channels, embed_dim=stem_dim,
                                               out_feature="last_feat")
    model.num_classes = num_classes  # D2
    model.device = "cpu"  # D4: synced to the input on every DINODetector call

    missing = object()
    for key, value in (overrides or {}).items():
        if key in _SET_BY_ARGS or key == "backbone.net" or key.startswith("backbone.net."):
            raise ValueError(f"override {key!r} is set by build_dino_config's arguments")
        if OmegaConf.select(model, key, default=missing, throw_on_missing=False) is missing:
            raise KeyError(f"override {key!r} is not a key of detrex's dino_vitdet model config")
        OmegaConf.update(model, key, value, merge=False)

    # D8: aux weights recomputed with dino_r50.py:113-121's rule, for the (possibly overridden) depth.
    suffixes = ("_enc",) + tuple(f"_{i}" for i in range(64))
    weights = OmegaConf.to_container(model.criterion.weight_dict)
    base = {k: v for k, v in weights.items() if not k.endswith(suffixes)}
    assert set(base) == _BASE_LOSSES, sorted(base)
    weight_dict = dict(base)
    if model.aux_loss:
        weight_dict.update({k + "_enc": v for k, v in base.items()})
        for i in range(model.transformer.decoder.num_layers - 1):
            weight_dict.update({k + f"_{i}": v for k, v in base.items()})
    model.criterion.weight_dict = weight_dict
    return model


def _pyramid_tokens(h: int, w: int) -> int:
    """Tokens over the 4 levels built from an ``h × w`` grid: scales 2, 1, ½ and the max-pool p6."""
    h5, w5 = h // 2, w // 2
    levels = [(2 * h, 2 * w), (h, w), (h5, w5), (-(-h5 // 2), -(-w5 // 2))]
    return sum(a * b for a, b in levels)


class DINODetector(nn.Module):
    """Tensor-in, tensors-out wrapper (D7) around detrex's list-of-dicts ``FeatureDINO``.

    ``forward`` switches on ``self.training`` as detrex does: train mode returns the weighted loss
    dict, eval mode returns per-image ``{"boxes" [K, 4] xyxy px, "scores" [K], "labels" [K]}``.
    """

    def __init__(self, *, in_channels: int = 1152, stem_dim: int = 768, num_classes: int = len(CLASS_NAMES),
                 overrides: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.detr = instantiate(build_dino_config(in_channels=in_channels, stem_dim=stem_dim,
                                                  num_classes=num_classes, overrides=overrides))

    def forward(self, feats: Tensor, targets: list[dict[str, Tensor]] | None = None,
                image_size: tuple[int, int] | None = None) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """``feats [B, C, H', W']``; ``image_size = (H, W)`` encoder-input pixels, default ``14·(H', W')``.

        Train-mode ``targets``: one dict per image with ``"boxes"`` float32 ``[N, 4]`` xyxy in
        encoder-input pixels and ``"labels"`` int64 ``[N]``; ``N = 0`` is allowed.
        """
        if feats.ndim != 4 or feats.shape[1] != self.in_channels:
            raise ValueError(f"DINODetector expects a 4-D [B, {self.in_channels}, H', W'] tensor, "
                             f"got {list(feats.shape)}")
        b, _, h, w = feats.shape
        height, width = image_size if image_size is not None else (PATCH * h, PATCH * w)
        tokens = _pyramid_tokens(h, w)
        if tokens < self.detr.num_queries:
            raise ValueError(f"a {h}x{w} grid gives {tokens} pyramid tokens, fewer than the "
                             f"{self.detr.num_queries} queries (two-stage top-k would fail)")
        self.detr.device = feats.device  # D4

        if not self.training:
            outputs = self.detr([{"image": feats[i], "height": height, "width": width} for i in range(b)])
            return [{"boxes": o["instances"].pred_boxes.tensor, "scores": o["instances"].scores,
                     "labels": o["instances"].pred_classes} for o in outputs]

        if targets is None or len(targets) != b:
            raise ValueError(f"train mode needs one target dict per image: batch {b}, "
                             f"got {None if targets is None else len(targets)}")
        if feats.device.type != "cuda":
            raise RuntimeError("DINO training needs a CUDA device: detrex's denoising code hard-codes "
                               ".cuda() (projects/dino/modeling/dino.py:370-458)")
        inputs = []
        for i, t in enumerate(targets):
            inst = Instances((height, width))
            inst.gt_boxes = Boxes(t["boxes"])
            inst.gt_classes = t["labels"]
            inputs.append({"image": feats[i], "instances": inst, "height": height, "width": width})
        return self.detr(inputs)
