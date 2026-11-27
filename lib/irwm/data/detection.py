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

Aligned FLIR v1 (F18): ``FlirDetectionDataset`` also takes an ``irwm.data.aligned.AlignedSplit``
(duck-typed; its pixels come from ``split.image_dataset()``), and ``AlignedDetectionDataModule``
loads aligned train / aligned val (person + car, no ignore regions). FLIR v2 behaviour is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from irwm.data.flir import FlirImageDataset, FlirRecord, FlirSplit, Modality
from irwm.features.preprocess import Policy, Preprocess


def _kept_boxes(pre: Preprocess, r: FlirRecord) -> tuple[np.ndarray, np.ndarray]:
    """``r``'s kept boxes as (``float32 [N, 4]`` xyxy encoder-input px, ``int64 [N]`` labels)."""
    xywh, keep = pre.geometry(r.width, r.height).apply_boxes(r.boxes)
    xywh = xywh[keep]
    xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1).astype(np.float32)
    return xyxy, r.labels[keep].astype(np.int64)


class FlirDetectionDataset(Dataset):
    """Frames of one ``FlirSplit`` → ``{"image" [3, H, W], "boxes" [N, 4] xyxy px, "labels" [N], ...}``.

    Any split object with ``modality``, ``split``, ``records`` and ``by_image_id`` works; one with an
    ``image_dataset()`` method (``AlignedSplit``) supplies its own pixel reader.
    """

    def __init__(self, split: FlirSplit, *, policy: Policy = "native", flip_p: float = 0.0,
                 filter_empty: bool = False, image_ids: Sequence[int] | None = None) -> None:
        if not 0.0 <= flip_p <= 1.0:
            raise ValueError(f"flip_p must be in [0, 1], got {flip_p}")
        self.split = split
        self.policy = policy
        self.flip_p = flip_p
        self.preprocess = Preprocess(policy, split.modality)
        image_dataset = getattr(split, "image_dataset", None)
        # decoding + mode/size checks (F2 / F16), no transform
        self._images = image_dataset() if callable(image_dataset) else FlirImageDataset(split)
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
        raise ValueError(f"asked for {n} non-empty frames, {split.modality}/{split.split} has {len(ids)}")
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


class AlignedDetectionDataModule(FlirDetectionDataModule):
    """Aligned FLIR v1 (F16/F18): aligned train (flip, empty frames dropped) and aligned val, plus its COCO GT.

    Same loaders and arguments as ``FlirDetectionDataModule``; ``val_split="train"`` with
    ``val_image_ids`` validates on train frames (smoke / overfit mode). ``root`` defaults to
    ``irwm.data.aligned.aligned_root()``; ``val_file`` defaults to the committed val-ids file. The GT
    has categories ``{1: person, 2: car}`` and no crowd.
    """

    def __init__(self, *, root=None, val_file=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.root = root
        self.val_file = val_file

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is not None:
            return
        from irwm.data.aligned import AlignedSplit

        load = dict(root=self.root, **({} if self.val_file is None else {"val_file": self.val_file}))
        train_split = AlignedSplit.load(self.modality, "train", **load)
        val_split = train_split if self.val_split == "train" else AlignedSplit.load(self.modality, "val", **load)
        self.train_set = FlirDetectionDataset(train_split, policy=self.policy, flip_p=self.flip_p,
                                              filter_empty=True, image_ids=self.train_image_ids)
        self.val_set = FlirDetectionDataset(val_split, policy=self.policy, flip_p=0.0,
                                            filter_empty=False, image_ids=self.val_image_ids)
        self.val_gt = val_split.subset(self.val_set._image_ids()).to_coco_gt("normalised")


EmptyRatio = float | Literal["all"]


class EmptyMixSampler(Sampler[int]):
    """Every positive index once per epoch, plus a fresh random draw of ``n_empty`` empty indices, shuffled.

    The draw depends only on ``(seed, epoch)``: Lightning calls ``set_epoch`` before each epoch (also after
    a resume), so a requeued run sees the same frames. Across epochs different empty frames are drawn,
    so the whole empty pool is covered over a run. ``len`` is constant: ``len(positive) + n_empty``.
    """

    def __init__(self, positive: Sequence[int], empty: Sequence[int], n_empty: int, seed: int = 0) -> None:
        if not 0 <= n_empty <= len(empty):
            raise ValueError(f"n_empty must be in [0, {len(empty)}], got {n_empty}")
        self.positive = [int(i) for i in positive]
        self.empty = [int(i) for i in empty]
        self.n_empty = int(n_empty)
        self.seed = int(seed)
        self.epoch = 0
        self._calls = 0  # iterations within this epoch (only > 0 if iterated without set_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._calls = 0

    def __len__(self) -> int:
        return len(self.positive) + self.n_empty

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch * 1_009 + self._calls)
        self._calls += 1
        drawn = [self.empty[i] for i in torch.randperm(len(self.empty), generator=g)[: self.n_empty].tolist()]
        idx = self.positive + drawn
        return iter([idx[i] for i in torch.randperm(len(idx), generator=g).tolist()])


def n_empty_per_epoch(empty_ratio: EmptyRatio, n_positive: int, n_pool: int) -> int:
    """Empty frames drawn per epoch: ``round(empty_ratio · n_positive)`` capped at the pool, or all of it."""
    if empty_ratio == "all":
        return n_pool
    ratio = float(empty_ratio)
    if ratio < 0:
        raise ValueError(f"empty_ratio must be >= 0 or 'all', got {empty_ratio!r}")
    return min(round(ratio * n_positive), n_pool)


class KaistDetectionDataModule(FlirDetectionDataModule):
    """KAIST-Paired: train (flip, empty frames dropped), val, and a fixed train subset for train-set metrics.

    - ``train`` / ``val`` are the splits of ``irwm.data.kaist`` (whole sets, ``train-all-04`` frames),
      read from ``modality``'s own camera annotations.
    - ``val_stride`` keeps 1 frame in ``val_stride`` per val video (1 = every ``train-all-04`` frame).
    - ``train_eval_stride`` (``None`` disables it) adds a second validation dataloader: 1 in
      ``train_eval_stride`` frames per train video, never flipped, empty frames kept, the same frames
      at every validation. Its GT is ``train_eval_gt``.
    - ``occlusion_variants`` (``{name: levels}``, default ``OCCLUSION_LEVELS``) builds the extra GTs
      ``val_variants`` / ``train_eval_variants``: only person boxes at those occlusion levels are GT.
    - ``val_split="train"`` with ``val_image_ids`` validates on train frames (smoke / overfit mode).
    - **Frame subsets (user, 2026-11-28)** for val and the train subset: ``nonempty`` = frames with
      ≥ 1 person/cyclist box (``val_subsets``: its own GT, scored with AP) and ``empty`` = frames with
      no box at all (``val_negatives``: scored with FPPI), the same definition as the training negatives.
      Frames whose only boxes are ignore regions are in neither (they stay in the all-frames metrics).
      ``extra_eval()`` carries the train subset's.
    - ``empty_ratio`` (user, 2026-11-28; default 0.25): train frames with no box at all are negatives.
      Each epoch keeps every positive frame and draws ``round(empty_ratio · n_positive)`` empty frames
      afresh (``EmptyMixSampler``, seeded by ``seed`` and the epoch); ``"all"`` keeps every empty frame,
      ``0`` drops them all (the F8/F18 convention). Frames whose only boxes are ignore regions are never
      negatives: they are dropped, since training on them as background would teach the head to suppress
      the people / person? groups that scoring ignores.

    GT categories are ``{1: person, 2: cyclist}``; ``people``, ``person?`` and ``unpaired`` are crowd.
    """

    def __init__(self, *, root=None, paired_root=None, val_stride: int = 1, train_eval_stride: int | None = None,
                 occlusion_variants: dict[str, Sequence[int]] | None = None, empty_ratio: EmptyRatio = 0.25,
                 seed: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        from irwm.data.kaist import OCCLUSION_LEVELS

        n_empty_per_epoch(empty_ratio, 0, 0)  # validates the value
        self.empty_ratio = empty_ratio
        self.seed = int(seed)
        self.train_sampler: EmptyMixSampler | None = None
        self.n_positive = self.n_empty_pool = self.n_ignore_only = self.n_empty_per_epoch = 0

        self.root = root
        self.paired_root = paired_root
        self.val_stride = int(val_stride)
        self.train_eval_stride = None if train_eval_stride is None else int(train_eval_stride)
        self.occlusion_variants = {k: tuple(v) for k, v in
                                   (OCCLUSION_LEVELS if occlusion_variants is None else occlusion_variants).items()}
        self.train_eval_set: FlirDetectionDataset | None = None
        self.train_eval_gt: dict = {}
        self.val_variants: dict[str, dict] = {}
        self.train_eval_variants: dict[str, dict] = {}
        self.val_subsets: dict[str, dict] = {}
        self.val_negatives: dict[str, list[int]] = {}
        self.train_eval_subsets: dict[str, dict] = {}
        self.train_eval_negatives: dict[str, list[int]] = {}

    def _gts(self, split) -> tuple[dict, dict[str, dict]]:
        return (split.to_coco_gt("normalised"),
                {name: split.to_coco_gt("normalised", occlusion=lv) for name, lv in self.occlusion_variants.items()})

    @staticmethod
    def _frame_subsets(dataset: FlirDetectionDataset, split) -> tuple[dict[str, dict], dict[str, list[int]]]:
        """``({"nonempty": {"gt", "image_ids"}}, {"empty": image_ids})`` of an evaluation dataset."""
        nonempty, empty = [], []
        for r, (boxes, _) in zip(dataset._selected_records(), dataset._targets):
            if len(boxes):
                nonempty.append(r.image_id)
            elif not len(r.ignore_boxes):
                empty.append(r.image_id)
        return ({"nonempty": {"gt": split.subset(nonempty).to_coco_gt("normalised"), "image_ids": nonempty}},
                {"empty": empty})

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is not None:
            return
        from irwm.data.kaist import KaistSplit, train_eval_ids

        load = dict(root=self.root, paired_root=self.paired_root)
        train_split = KaistSplit.load(self.modality, "train", **load)
        if self.val_split == "train":
            val_split, val_ids = train_split, self.val_image_ids
        else:
            val_split = KaistSplit.load(self.modality, "val", **load)
            val_ids = self.val_image_ids if self.val_image_ids is not None else train_eval_ids(val_split, self.val_stride)
        self._setup_train(train_split)
        self.val_set = FlirDetectionDataset(val_split, policy=self.policy, flip_p=0.0,
                                            filter_empty=False, image_ids=val_ids)
        self.val_gt, self.val_variants = self._gts(val_split.subset(self.val_set._image_ids()))
        self.val_subsets, self.val_negatives = self._frame_subsets(self.val_set, val_split)
        if self.train_eval_stride is not None:
            ids = train_eval_ids(train_split, self.train_eval_stride)
            if self.train_image_ids is not None:  # smoke mode: score only frames that are trained on
                wanted = set(self.train_image_ids)
                ids = [i for i in ids if i in wanted] or sorted(wanted)
            self.train_eval_set = FlirDetectionDataset(train_split, policy=self.policy, flip_p=0.0,
                                                       filter_empty=False, image_ids=ids)
            self.train_eval_gt, self.train_eval_variants = self._gts(
                train_split.subset(self.train_eval_set._image_ids()))
            self.train_eval_subsets, self.train_eval_negatives = self._frame_subsets(self.train_eval_set, train_split)

    def _setup_train(self, train_split) -> None:
        """Train set over every frame, then positives / empty pool / ignore-only frames and the sampler."""
        full = FlirDetectionDataset(train_split, policy=self.policy, flip_p=self.flip_p,
                                    filter_empty=False, image_ids=self.train_image_ids)
        records = full._selected_records()
        positive, empty, ignore_only = [], [], []
        for i, (r, (boxes, _)) in enumerate(zip(records, full._targets)):
            if len(boxes):
                positive.append(i)
            elif len(r.ignore_boxes):
                ignore_only.append(i)
            else:
                empty.append(i)
        n_empty = n_empty_per_epoch(self.empty_ratio, len(positive), len(empty))
        if n_empty == 0:  # no negatives: exactly the F8/F18 dataset (empty frames filtered out)
            self.train_set = FlirDetectionDataset(train_split, policy=self.policy, flip_p=self.flip_p,
                                                  filter_empty=True, image_ids=self.train_image_ids)
            self.train_sampler = None
        else:
            keep = sorted(positive + empty)  # drop the ignore-only frames
            self.train_set = FlirDetectionDataset(train_split, policy=self.policy, flip_p=self.flip_p,
                                                  filter_empty=False,
                                                  image_ids=[records[i].image_id for i in keep])
            pos = set(positive)
            self.train_sampler = EmptyMixSampler([k for k, i in enumerate(keep) if i in pos],
                                                 [k for k, i in enumerate(keep) if i not in pos],
                                                 n_empty, seed=self.seed)
        self.n_positive, self.n_empty_pool, self.n_ignore_only = len(positive), len(empty), len(ignore_only)
        self.n_empty_per_epoch = n_empty

    @property
    def train_epoch_size(self) -> int:
        """Frames seen per training epoch (positives + the empty draw)."""
        return self.n_positive + self.n_empty_per_epoch

    def train_dataloader(self) -> DataLoader:
        if self.train_sampler is None:
            return self._loader(self.train_set, train=True)
        return DataLoader(self.train_set, batch_size=self.batch_size, sampler=self.train_sampler, drop_last=True,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0, collate_fn=collate_detection)

    def extra_eval(self, prefix: str = "train_eval") -> list[dict]:
        """``DetectorModule(extra_eval=...)`` for the train subset (empty if it is disabled)."""
        if self.train_eval_set is None:
            return []
        return [{"prefix": prefix, "gt": self.train_eval_gt, "variants": self.train_eval_variants,
                 "subsets": self.train_eval_subsets, "negatives": self.train_eval_negatives}]

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        val = self._loader(self.val_set, train=False)
        if self.train_eval_set is None:
            return val
        return [val, self._loader(self.train_eval_set, train=False)]
