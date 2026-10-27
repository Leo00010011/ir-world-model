"""Image preprocessing for the DINOv2-Reg encoder: resolution policy, thermal 1→3, ImageNet norm.

Mirrors the reference loader (third_party/DINO-Foresight/src/data.py:158-165): open as RGB
(``convert("RGB")`` replicates a thermal ``L`` frame to 3 channels), ``T.Resize((H, W))`` on
the PIL image, ``ToTensor``, ``Normalize``. Resolution policies (Roadmap R5, still open):

- ``stretch``: the full frame resized to 448×896. This *is* the reference transform.
- ``crop``: centre 2:1 band, then 448×896. A deviation from the reference.
- ``native``: centre 5:4 crop, then 448×560 (32×40 tokens). A deviation; never crops thermal.

Only deterministic centre crops: training augmentation belongs to F8. Runs per image on the
CPU (DataLoader workers); the encoder (``irwm.features.encoder``) takes the batched output.
Spec: spec/2026-10-16-dinov2-reg-encoder-wrapper (FR1-FR3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch import Tensor

from irwm.worldmodel.reference import IMAGENET_MEAN, IMAGENET_STD, center_crop_box

Policy = Literal["stretch", "crop", "native"]
_Modality = Literal["rgb", "thermal"]
POLICY_SIZE: dict[Policy, tuple[int, int]] = {"stretch": (448, 896), "crop": (448, 896), "native": (448, 560)}
_CROP_ASPECT: dict[Policy, tuple[int, int] | None] = {"stretch": None, "crop": (2, 1), "native": (5, 4)}


@dataclass(frozen=True)
class ResizeGeometry:
    """How one source frame was mapped to the encoder input, for boxes in both directions."""

    src_size: tuple[int, int]  # (W, H) of the source frame
    crop_box: tuple[int, int, int, int]  # (left, top, w, h) in source pixels; full frame for stretch
    out_size: tuple[int, int]  # (W, H) of the output

    def _scale(self) -> tuple[float, float]:
        return self.out_size[0] / self.crop_box[2], self.out_size[1] / self.crop_box[3]

    def apply_boxes(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """COCO ``xywh`` in source px ``[N, 4]`` → (``xywh`` in output px ``float32 [N, 4]``, ``keep bool [N]``).

        Boxes are clipped to the crop box, shifted by its offset, then scaled per axis.
        Rows with no visible area get ``keep=False`` and hold zeros.
        """
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        left, top, cw, ch = self.crop_box
        x0 = np.clip(b[:, 0], left, left + cw)
        y0 = np.clip(b[:, 1], top, top + ch)
        x1 = np.clip(b[:, 0] + b[:, 2], left, left + cw)
        y1 = np.clip(b[:, 1] + b[:, 3], top, top + ch)
        w, h = x1 - x0, y1 - y0
        keep = (w > 0) & (h > 0)
        sx, sy = self._scale()
        out = np.stack([(x0 - left) * sx, (y0 - top) * sy, w * sx, h * sy], axis=1)
        out[~keep] = 0.0
        return out.astype(np.float32), keep

    def invert_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """``xywh`` in output px ``[N, 4]`` → ``xywh`` in source px ``float32 [N, 4]`` (no clipping)."""
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        left, top, _, _ = self.crop_box
        sx, sy = self._scale()
        out = np.stack([b[:, 0] / sx + left, b[:, 1] / sy + top, b[:, 2] / sx, b[:, 3] / sy], axis=1)
        return out.astype(np.float32)


def _crop_box(policy: Policy, width: int, height: int) -> tuple[int, int, int, int]:
    """Centre crop ``(left, top, w, h)`` of a ``width×height`` frame under ``policy``."""
    aspect = _CROP_ASPECT[policy]
    return (0, 0, width, height) if aspect is None else center_crop_box(width, height, aspect)


class Preprocess:
    """``uint8 [H, W]`` (thermal) or ``[H, W, 3]`` (rgb) → (``float32 [3, H_out, W_out]``, geometry).

    Picklable (no lambdas), so it can be the ``transform`` of ``FlirImageDataset``.
    """

    def __init__(self, policy: Policy, modality: _Modality) -> None:
        if policy not in POLICY_SIZE:
            raise ValueError(f"Unknown policy {policy!r}; expected one of {list(POLICY_SIZE)}")
        if modality not in get_args(_Modality):
            raise ValueError(f"Unknown modality {modality!r}; expected one of {list(get_args(_Modality))}")
        self.policy = policy
        self.modality = modality
        self.size = POLICY_SIZE[policy]  # (H, W)
        # A (H, W) tuple resizes to exactly that size, as the reference does; an int would keep the aspect.
        self.resize = T.Resize(self.size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def _check(self, img: np.ndarray) -> None:
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
            raise ValueError(f"Expected a uint8 numpy array, got {getattr(img, 'dtype', type(img))}")
        if self.modality == "thermal" and img.ndim != 2:
            raise ValueError(f"thermal expects [H, W], got shape {list(img.shape)}")
        if self.modality == "rgb" and (img.ndim != 3 or img.shape[2] != 3):
            raise ValueError(f"rgb expects [H, W, 3], got shape {list(img.shape)}")

    def geometry(self, width: int, height: int) -> ResizeGeometry:
        """The geometry ``__call__`` returns for a ``width×height`` frame, without any pixels (F8 FR1)."""
        return ResizeGeometry(src_size=(width, height), crop_box=_crop_box(self.policy, width, height),
                              out_size=(self.size[1], self.size[0]))

    def __call__(self, img: np.ndarray) -> tuple[Tensor, ResizeGeometry]:
        self._check(img)
        pil = Image.fromarray(img).convert("RGB")  # thermal L → 3 identical channels
        geom = self.geometry(*pil.size)
        left, top, cw, ch = geom.crop_box
        if (cw, ch) != geom.src_size:
            pil = pil.crop((left, top, left + cw, top + ch))
        x = self.normalize(self.to_tensor(self.resize(pil)))
        return x, geom

    def __repr__(self) -> str:
        return f"Preprocess(policy={self.policy!r}, modality={self.modality!r})"
