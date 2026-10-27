"""FLIR detection data for training the DINO-DETR head on features encoded on the fly.

Spec: spec/2026-10-26-detector-training-pipeline (FR2, FR3). Each item is one FLIR frame,
preprocessed for the frozen encoder (``irwm.features.preprocess.Preprocess``, ``native`` by
default), with its 6-class boxes mapped into the encoder input as xyxy pixels by the same
``ResizeGeometry`` the image went through.

- **Ignore regions are not targets** (user decision 2026-10-23, F7 plan.md "Notes for F8"):
  boxes of other FLIR classes are dropped here and never reach the model. They stay crowd GT
  in val mAP (``FlirSplit.to_coco_gt``, F2).
- **Empty frames** (no kept box after the crop mapping) are dropped from train, as detrex's
  ``filter_empty=True`` does, and kept in val.
- **Augmentation:** horizontal flip only, on the preprocessed image before the encoder.

Frame selection uses metadata only (``Preprocess.geometry``), so no image is decoded in ``__init__``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from irwm.data.flir import FlirImageDataset, FlirRecord, FlirSplit, Modality
from irwm.features.preprocess import Policy, Preprocess


def _kept_boxes(pre: Preprocess, r: FlirRecord) -> tuple[np.ndarray, np.ndarray]:
    """``r``'s kept boxes as (``float32 [N, 4]`` xyxy encoder-input px, ``int64 [N]`` labels)."""
    xywh, keep = pre.geometry(r.width, r.height).apply_boxes(r.boxes)
    xywh = xywh[keep]
    xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1).astype(np.float32)
    return xyxy, r.labels[keep].astype(np.int64)


class FlirDetectionDataset(Dataset):
    """Frames of one ``FlirSplit`` → ``{"image" [3, H, W], "boxes" [N, 4] xyxy px, "labels" [N], ...}``."""

    def __init__(self, split: FlirSplit, *, policy: Policy = "native", flip_p: float = 0.0,
                 filter_empty: bool = False, image_ids: Sequence[int] | None = None) -> None:
        if not 0.0 <= flip_p <= 1.0:
            raise ValueError(f"flip_p must be in [0, 1], got {flip_p}")
        self.split = split
        self.policy = policy
        self.flip_p = flip_p
        self.preprocess = Preprocess(policy, split.modality)
        self._images = FlirImageDataset(split)  # decoding + mode/size checks (F2), no transform
        position = {r.image_id: i for i, r in enumerate(split.records)}

        if image_ids is None:
            records = split.records
        else:
            wanted = {split.by_image_id(int(i)).image_id for i in image_ids}  # KeyError on unknown ids
            records = [r for r in split.records if r.image_id in wanted]

        self._positions: list[int] = []
        self._targets: list[tuple[np.ndarray, np.ndarray]] = []
        self.n_dropped_empty = 0
        for r in records:
            boxes, labels = _kept_boxes(self.preprocess, r)
            if filter_empty and len(boxes) == 0:
                self.n_dropped_empty += 1
                continue
            self._positions.append(position[r.image_id])
            self._targets.append((boxes, labels))

    def _selected_records(self) -> list[FlirRecord]:
        """The selected records, ascending ``image_id``."""
        return [self.split.records[p] for p in self._positions]

    def _image_ids(self) -> list[int]:
        return [self.split.records[p].image_id for p in self._positions]

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, i: int) -> dict:
        item = self._images[self._positions[i]]
        image, geometry = self.preprocess(item["image"])
        boxes_np, labels_np = self._targets[i]
        boxes, labels = torch.from_numpy(boxes_np.copy()), torch.from_numpy(labels_np.copy())
        flipped = bool(self.flip_p > 0 and torch.rand(()) < self.flip_p)
        if flipped:
            width = image.shape[-1]
            image = image.flip(-1)
            boxes = torch.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1)
        return {"image": image, "boxes": boxes, "labels": labels, "image_id": item["image_id"],
                "geometry": geometry, "flipped": flipped}


def collate_detection(items: list[dict]) -> dict:
    """Items → ``{"images" [B, 3, H, W], "targets": [{"boxes", "labels"}], "image_ids", "geometries"}``."""
    return {
        "images": torch.stack([it["image"] for it in items]),
        "targets": [{"boxes": it["boxes"], "labels": it["labels"]} for it in items],
        "image_ids": [int(it["image_id"]) for it in items],
        "geometries": [it["geometry"] for it in items],
    }


def first_nonempty_ids(split: FlirSplit, n: int, policy: Policy = "native") -> list[int]:
    """The first ``n`` ``image_id``s (ascending) with ≥ 1 kept box after the ``policy`` mapping."""
    ids = FlirDetectionDataset(split, policy=policy, filter_empty=True)._image_ids()
    if n > len(ids):
        raise ValueError(f"asked for {n} non-empty frames, FLIR {split.modality}/{split.split} has {len(ids)}")
    return ids[:n]


class FlirDetectionDataModule(L.LightningDataModule):
    """Train (flip, empty frames dropped) and val (all frames) of one modality, plus the val COCO GT."""

    def __init__(self, *, modality: Modality, policy: Policy = "native", batch_size: int = 16,
                 num_workers: int = 0, flip_p: float = 0.5, train_image_ids: Sequence[int] | None = None,
                 val_split: Literal["val", "train"] = "val", val_image_ids: Sequence[int] | None = None) -> None:
        super().__init__()
        if val_split not in ("val", "train"):
            raise ValueError(f"val_split must be 'val' or 'train', got {val_split!r}")
        self.modality = modality
        self.policy = policy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flip_p = flip_p
        self.train_image_ids = None if train_image_ids is None else list(train_image_ids)
        self.val_split = val_split
        self.val_image_ids = None if val_image_ids is None else list(val_image_ids)
        self.train_set: FlirDetectionDataset | None = None
        self.val_set: FlirDetectionDataset | None = None
        self.val_gt: dict = {}

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is not None:  # the entry point calls setup before Lightning does
            return
        train_split = FlirSplit.load(self.modality, "train")
        val_split = train_split if self.val_split == "train" else FlirSplit.load(self.modality, "val")
        self.train_set = FlirDetectionDataset(train_split, policy=self.policy, flip_p=self.flip_p,
                                              filter_empty=True, image_ids=self.train_image_ids)
        self.val_set = FlirDetectionDataset(val_split, policy=self.policy, flip_p=0.0,
                                            filter_empty=False, image_ids=self.val_image_ids)
        self.val_gt = FlirSplit(self.modality, val_split.split, val_split.root,
                                self.val_set._selected_records()).to_coco_gt("normalised")

    def _loader(self, dataset: FlirDetectionDataset, train: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, drop_last=train,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0, collate_fn=collate_detection)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, train=False)
