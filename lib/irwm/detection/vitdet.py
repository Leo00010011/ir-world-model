"""End-to-end DINO-ViTDet detector for the ``mae`` backbone variant (TechStack §3.1b, F17/F18).

This is detrex's stock ``dino_vitdet`` (its own ViT-B/16 with window + global attention and
relative positions, then SimpleFeaturePyramid and the DINO head), initialised from the official
MAE pre-trained ViT-B (``irwm.detection.mae_assets``) and fine-tuned end to end. It keeps the
tensor contract of ``DINODetector`` but takes ImageNet-normalised images instead of features.

Deviations from detrex's ``dino_vitdet.py`` / ``dino_vitdet_base_4scale_12ep.py`` (V1-V8):

- V1 ``backbone.square_pad`` 1024 → 0. detrex's ``DINO.preprocess_image`` calls
  ``ImageList.from_tensors`` without ``padding_constraints``, so it never took effect; 0 says so.
- V2 ``backbone.net.img_size`` 1024 → ``max(image_hw)`` (640), and the global-attention blocks'
  ``rel_pos_h`` is re-created as ``zeros(2·H/16 − 1, head_dim)``, so both rel-pos tables match the
  32×40 grid exactly (no per-step interpolation). MAE has no rel-pos: these are zero-init either way.
  The absolute ``pos_embed`` stays at its 14×14 (+cls) pre-training size and is bicubic-interpolated
  to 32×40 on every forward by detectron2's ``get_abs_pos``, as stock ViTDet does at 1024.
- V3 inputs arrive already normalised (``Preprocess``), so ``FeatureDINO`` (no detrex normaliser,
  no padding) is the meta-architecture.
- V4 ``num_classes`` 80 → 2 (person, car). V5 ``use_act_checkpoint=True``.
- V6 layer-wise lr decay 0.7 (ViTDet's ``mask_rcnn_vitdet_b_100ep.py``) instead of detrex's flat
  backbone ×0.1, which stays selectable. V7 ``pos_embed`` weight decay 0 (ViTDet's override).
- V8 no multi-scale / LSJ augmentation here: augmentation belongs to the training pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
from detectron2.config import instantiate
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.solver.build import get_default_optimizer_params
from detectron2.structures import Boxes, Instances
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from irwm.detection.detector import FeatureDINO
from irwm.detection.detrex_adapter import load_dino_vitdet_config
from irwm.detection.mae_assets import verify_mae_weights

VITDET_META_ARCH = "vitdet_mae"
VIT_PATCH = 16
_STRIDE = 64  # p6 stride: image sides must be multiples of it
_VIT = "backbone.net."
_SET_BY_ARGS = ("num_classes", "device", "backbone.square_pad", "backbone.net.img_size",
                "backbone.net.use_act_checkpoint")
_BASE_LOSSES = {"loss_class", "loss_bbox", "loss_giou", "loss_class_dn", "loss_bbox_dn", "loss_giou_dn"}
# MAE keys with no counterpart in detectron2's ViT, and ViT params MAE does not provide.
_MAE_UNEXPECTED_PREFIXES = ("cls_token", "mask_token", "norm.", "fc_norm.", "decoder", "head.")
_MAE_MISSING_SUFFIXES = (".attn.rel_pos_h", ".attn.rel_pos_w")


def build_vitdet_config(*, num_classes: int = 2, image_hw: tuple[int, int] = (512, 640),
                        use_act_checkpoint: bool = True,
                        overrides: Mapping[str, Any] | None = None) -> DictConfig:
    """detrex's ``dino_vitdet`` model config with V1-V5 applied.

    ``overrides`` are dotted keys relative to ``model`` (``{"backbone.net.depth": 2}``).
    Unknown keys raise ``KeyError``; keys set by the arguments raise ``ValueError``.
    """
    model = load_dino_vitdet_config().model
    model._target_ = FeatureDINO  # V3
    model.num_classes = num_classes  # V4
    model.device = "cpu"  # synced to the input on every call, as DINODetector's D4
    model.backbone.square_pad = 0  # V1
    model.backbone.net.img_size = max(image_hw)  # V2 (global rel-pos h is resized in ViTDetDINO)
    model.backbone.net.use_act_checkpoint = use_act_checkpoint  # V5

    # Duplicated from irwm.detection.detector.build_dino_config (override validation and its D8
    # aux-weight rule), so this module does not depend on that file's evolving internals.
    missing = object()
    for key, value in (overrides or {}).items():
        if key in _SET_BY_ARGS:
            raise ValueError(f"override {key!r} is set by build_vitdet_config's arguments")
        if OmegaConf.select(model, key, default=missing, throw_on_missing=False) is missing:
            raise KeyError(f"override {key!r} is not a key of detrex's dino_vitdet model config")
        OmegaConf.update(model, key, value, merge=False)

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
    """Tokens over the 4 levels (strides 8, 16, 32, 64) of an ``h × w`` image (multiples of 64)."""
    return sum((h // s) * (w // s) for s in (8, 16, 32, 64))


class ViTDetDINO(nn.Module):
    """Tensor-in, tensors-out DINO-ViTDet, with ``DINODetector``'s contract but image input.

    Train mode returns the weighted loss dict; eval mode returns per-image
    ``{"boxes" [K, 4] xyxy px, "scores" [K], "labels" [K]}``. ``self.vit`` is the ViT backbone.
    """

    def __init__(self, *, num_classes: int = 2, image_hw: tuple[int, int] = (512, 640),
                 use_act_checkpoint: bool = True, overrides: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        h, w = image_hw
        if h % _STRIDE or w % _STRIDE:
            raise ValueError(f"image_hw sides must be multiples of {_STRIDE}, got {tuple(image_hw)}")
        self.image_hw = (h, w)
        cfg = build_vitdet_config(num_classes=num_classes, image_hw=image_hw,
                                  use_act_checkpoint=use_act_checkpoint, overrides=overrides)
        self.depth = int(cfg.backbone.net.depth)
        self.detr = instantiate(cfg)
        # V2: exact rel-pos tables for the (possibly non-square) global-attention grid.
        gh, gw = h // VIT_PATCH, w // VIT_PATCH
        for block in self.vit.blocks:
            attn = block.attn
            if block.window_size == 0 and getattr(attn, "use_rel_pos", False):
                head_dim = attn.rel_pos_h.shape[1]
                attn.rel_pos_h = nn.Parameter(torch.zeros(2 * gh - 1, head_dim))
                attn.rel_pos_w = nn.Parameter(torch.zeros(2 * gw - 1, head_dim))

    @property
    def vit(self) -> nn.Module:
        """detectron2's ``ViT`` inside the SimpleFeaturePyramid."""
        return self.detr.backbone.net

    def forward(self, images: Tensor, targets: list[dict[str, Tensor]] | None = None,
                image_size: tuple[int, int] | None = None) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """``images [B, 3, H, W]`` ImageNet-normalised; ``image_size = (H, W)`` px, default the tensor's.

        Train-mode ``targets``: one dict per image with ``"boxes"`` float32 ``[N, 4]`` xyxy in input
        pixels and ``"labels"`` int64 ``[N]``; ``N = 0`` is allowed.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"ViTDetDINO expects a 4-D [B, 3, H, W] tensor, got {list(images.shape)}")
        b, _, h, w = images.shape
        if h % _STRIDE or w % _STRIDE:
            raise ValueError(f"image sides must be multiples of {_STRIDE}, got {h}x{w}")
        if _pyramid_tokens(h, w) < self.detr.num_queries:
            raise ValueError(f"a {h}x{w} image gives {_pyramid_tokens(h, w)} pyramid tokens, fewer than "
                             f"the {self.detr.num_queries} queries (two-stage top-k would fail)")
        height, width = image_size if image_size is not None else (h, w)
        self.detr.device = images.device

        if not self.training:
            outputs = self.detr([{"image": images[i], "height": height, "width": width} for i in range(b)])
            return [{"boxes": o["instances"].pred_boxes.tensor, "scores": o["instances"].scores,
                     "labels": o["instances"].pred_classes} for o in outputs]

        if targets is None or len(targets) != b:
            raise ValueError(f"train mode needs one target dict per image: batch {b}, "
                             f"got {None if targets is None else len(targets)}")
        if images.device.type != "cuda":
            raise RuntimeError("DINO training needs a CUDA device: detrex's denoising code hard-codes "
                               ".cuda() (projects/dino/modeling/dino.py:370-458)")
        inputs = []
        for i, t in enumerate(targets):
            inst = Instances((h, w))  # the image itself: no padding, so the DINO mask is all valid
            inst.gt_boxes = Boxes(t["boxes"])
            inst.gt_classes = t["labels"]
            inputs.append({"image": images[i], "instances": inst, "height": height, "width": width})
        return self.detr(inputs)


def map_mae_state_dict(state: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], list[str]]:
    """MAE checkpoint keys → keys of detectron2's ``ViT``; returns (mapped, dropped keys).

    MAE (timm ``VisionTransformer``) and detectron2's ``ViT`` share ``pos_embed``,
    ``patch_embed.proj.*`` and ``blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.fc1,mlp.fc2}.*``
    one to one; ``cls_token`` and the final ``norm`` have no counterpart and are dropped.
    """
    mapped, dropped = {}, []
    for k, v in state.items():
        if k.startswith(_MAE_UNEXPECTED_PREFIXES):
            dropped.append(k)
        else:
            mapped[k] = v
    return mapped, sorted(dropped)


def load_mae_weights(model: ViTDetDINO, path: str | Path | None = None, *,
                     verify: bool = True) -> tuple[list[str], list[str]]:
    """Load the MAE ViT-B checkpoint into ``model.vit``; returns ``(missing, unexpected)`` ViT keys.

    ``path`` defaults to the pinned asset, whose SHA-256 is checked (``verify``). Raises if anything
    other than the rel-pos tables is missing, or anything other than the cls token / final norm /
    decoder is unexpected, or a shape differs.
    """
    path = verify_mae_weights(path) if verify else Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    mapped, dropped = map_mae_state_dict(state)
    result = model.vit.load_state_dict(mapped, strict=False)
    missing, unexpected = sorted(result.missing_keys), sorted(set(result.unexpected_keys) | set(dropped))
    bad_missing = [k for k in missing if not k.endswith(_MAE_MISSING_SUFFIXES)]
    bad_unexpected = [k for k in unexpected if not k.startswith(_MAE_UNEXPECTED_PREFIXES)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"MAE checkpoint {path} does not fit the ViT: missing {bad_missing}, "
                           f"unexpected {bad_unexpected}")
    return missing, unexpected


def vitdet_param_groups(model: ViTDetDINO, *, base_lr: float = 1e-4, weight_decay: float = 1e-4,
                        lr_schedule: Literal["layer_decay", "flat"] = "layer_decay", layer_decay: float = 0.7,
                        backbone_lr_factor: float = 0.1) -> list[dict[str, Any]]:
    """AdamW param groups ``[{"params", "lr", "weight_decay"}]`` with detectron2's rules.

    ``layer_decay``: ``get_vit_lr_decay_rate`` (patch/pos embed 0.7^(depth+1), block i 0.7^(depth−i),
    pyramid and head 1). ``flat``: detrex's ×``backbone_lr_factor`` on names containing ``backbone``.
    Weight decay is 0 on ``torch.nn`` norm layers and ``pos_embed``; biases and rel-pos keep it.
    """
    if lr_schedule == "layer_decay":
        factor = partial(get_vit_lr_decay_rate, lr_decay_rate=layer_decay, num_layers=model.depth)
    elif lr_schedule == "flat":
        factor = lambda name: backbone_lr_factor if "backbone" in name else 1.0  # noqa: E731
    else:
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}; expected 'layer_decay' or 'flat'")
    return get_default_optimizer_params(model.detr, base_lr=base_lr, weight_decay=weight_decay,
                                        weight_decay_norm=0.0, lr_factor_func=factor,
                                        overrides={"pos_embed": {"weight_decay": 0.0}})


def vitdet_from_config(cfg: Mapping[str, Any]) -> ViTDetDINO:
    """Build a ``ViTDetDINO`` from the ``detector=vitdet_mae`` config and apply its ``init`` (mae/none)."""
    cfg = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    init = cfg.get("init", "mae")
    if init not in ("mae", "none"):
        raise ValueError(f"unknown init {init!r}; expected 'mae' or 'none'")
    model = ViTDetDINO(num_classes=cfg.get("num_classes", 2), image_hw=tuple(cfg.get("image_hw", (512, 640))),
                       use_act_checkpoint=cfg.get("use_act_checkpoint", True),
                       overrides=cfg.get("overrides") or None)
    if init == "mae":
        load_mae_weights(model, cfg.get("init_path"))
    return model
