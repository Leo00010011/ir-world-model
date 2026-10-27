# F8 · Detector training pipeline: Requirements

## Goal

Train the F7 DINO-DETR head on FLIR ADAS v2 features encoded on the fly by the frozen F4 encoder, once per modality, and freeze the result as `det_rgb` and `det_ir`. These two heads are the instrument of the whole thesis: F9 measures H1 with them on real features, and F12 measures H2 by applying them, unchanged, to DINO-Foresight's predicted features. F8 delivers the data pipeline (FLIR frames → `native` preprocessing → flip → frozen DINOv2-Reg + PCA), a `LightningModule` with detrex's DINO-ViTDet 50ep recipe, an off-the-shelf val mAP (pycocotools on `FlirSplit.to_coco_gt`), an end-to-end overfit check on real frames, the two full training runs on ctcomp3, a frozen export with a loader for later features, and a results note.

## Scope

In scope:
- `irwm.data.detection`: a FLIR detection dataset (boxes mapped to the encoder input, ignore classes dropped, empty frames filtered, horizontal flip), its collate function and a `LightningDataModule`.
- `irwm.detection.coco_val`: pycocotools val mAP on FLIR-pixel boxes, with ignore regions as crowd GT (F2).
- `irwm.detection.lightning`: the `LightningModule` (frozen encoder with SDPA + trainable `DINODetector`, detrex optimiser and 50ep schedule, val mAP, timing of the encoder share).
- `irwm.detection.frozen`: export format and loader of the frozen heads.
- `Preprocess.geometry` (a small addition to F4's `irwm.features.preprocess`).
- `projects/detection/train.py` (training and overfit mode) and `projects/detection/export.py`, with Hydra configs `experiment=det_rgb` and `experiment=det_ir`.
- CPU tests and GPU tests in `tests/test_detector_training.py`.
- The overfit runs and the two 50-epoch runs on ctcomp3 (user steps), all with FLIR staged to node-local scratch (`IRWM_STAGE_DATA=1`, existing `slurm/stage_data.sh`), the exports, and `spec/2026-10-26-detector-training-pipeline/results.md`.
- fp16 everywhere on the GPU: Lightning `16-mixed` for training, validation and the overfit, the encoder's fp16 autocast, and fp16 autocast recorded as the required precision of the frozen heads.

Out of scope:
- The rigorous evaluator (bootstrap CIs, per-horizon breakdown, F6) and any test-split or video evaluation (F9).
- The H1 comparison itself (F9). Val numbers in the results note are selection numbers, not H1 evidence.
- Feature caching (F5). F8 only reports whether encoding is the bottleneck.
- An Optuna sweep config (user decision, 2026-10-26: detrex's hyperparameters are the fixed reference, and each run costs ≈ 13 GPU-h).
- Multi-scale or crop augmentation (user decision: flip only).
- Any change to detrex, detectron2 or the F7 detector module.

## Functional Requirements

**FR1 — Preprocess geometry without pixels (`lib/irwm/features/preprocess.py`).**
- `Preprocess.geometry(self, width: int, height: int) -> ResizeGeometry` returns the geometry that `__call__` would return for a `width×height` frame, without decoding an image.
- `Preprocess.__call__` builds its geometry through `geometry`, so both always agree. The F4 tests stay green unchanged.

**FR2 — FLIR detection dataset (`lib/irwm/data/detection.py`).**
- `class FlirDetectionDataset(torch.utils.data.Dataset)` with `__init__(self, split: FlirSplit, *, policy: Policy = "native", flip_p: float = 0.0, filter_empty: bool = False, image_ids: Sequence[int] | None = None)`.
- Frame selection happens in `__init__`, from metadata only:
  1. start from `split.records` (ascending `image_id`), restricted to `image_ids` when it is given (an unknown id raises `KeyError`);
  2. map each record's kept boxes (`FlirRecord.boxes`, the 6 classes) with `Preprocess(policy, split.modality).geometry(r.width, r.height).apply_boxes`, and keep the rows with `keep=True`;
  3. if `filter_empty`, drop records left with 0 boxes.
- **Ignore regions are not targets and are never passed to the model** (`FlirRecord.ignore_boxes` is unused here).
- `__len__` is the number of selected frames. `n_dropped_empty: int` holds how many frames step 3 removed.
- `__getitem__(i) -> dict` with:
  - `"image"`: `float32 [3, H, W]` from `Preprocess(policy, modality)` (`[3, 448, 560]` under `native`);
  - `"boxes"`: `float32 [N, 4]` **xyxy in encoder-input pixels**, converted from the xywh output of `apply_boxes`;
  - `"labels"`: `int64 [N]` in `0..5`;
  - `"image_id"`: `int`; `"geometry"`: the `ResizeGeometry` of that frame; `"flipped"`: `bool`.
- Flip: with probability `flip_p`, drawn with `torch.rand(())` (global torch RNG, seeded per worker by Lightning), the image becomes `image.flip(-1)` and boxes become `x0' = W − x1`, `x1' = W − x0` (y unchanged). `geometry` is the unflipped one.
- `flip_p` outside `[0, 1]` → `ValueError`.

**FR3 — Collate and DataModule (`lib/irwm/data/detection.py`).**
- `collate_detection(items: list[dict]) -> dict` → `{"images": float32 [B, 3, H, W] (stacked), "targets": list[{"boxes", "labels"}], "image_ids": list[int], "geometries": list[ResizeGeometry]}`.
- `class FlirDetectionDataModule(lightning.LightningDataModule)` with `__init__(self, *, modality: Modality, policy: Policy = "native", batch_size: int = 16, num_workers: int = 0, flip_p: float = 0.5, train_image_ids: Sequence[int] | None = None, val_split: Literal["val", "train"] = "val", val_image_ids: Sequence[int] | None = None)`.
- `setup(stage)` loads `FlirSplit.load(modality, "train")` and, when `val_split == "val"`, `FlirSplit.load(modality, "val")`. It builds:
  - train: `FlirDetectionDataset(train_split, policy, flip_p=flip_p, filter_empty=True, image_ids=train_image_ids)`;
  - val: `FlirDetectionDataset(<val_split>, policy, flip_p=0.0, filter_empty=False, image_ids=val_image_ids)`;
  - `val_gt: dict`, the COCO GT of exactly the val frames: `FlirSplit(modality, split, root, selected_records).to_coco_gt("normalised")`.
- `train_dataloader()`: `shuffle=True`, `drop_last=True`, `pin_memory=True`, `persistent_workers = num_workers > 0`, `collate_fn=collate_detection`. `val_dataloader()`: `shuffle=False`, `drop_last=False`, same collate.
- `first_nonempty_ids(split: FlirSplit, n: int, policy: Policy = "native") -> list[int]` returns the first `n` `image_id`s (ascending) with ≥ 1 kept box after mapping. `n` larger than available → `ValueError`.

**FR4 — Val mAP (`lib/irwm/detection/coco_val.py`).**
- `to_flir_detections(preds: list[dict[str, Tensor]], image_ids: list[int], geometries: list[ResizeGeometry]) -> list[dict]` turns `DINODetector` eval outputs (xyxy encoder-input pixels) into COCO result dicts: xyxy → xywh → `geometry.invert_boxes` (FLIR pixels), `category_id = LABEL_TO_CATEGORY_ID[label]`, `score` as float. All K ≤ 300 boxes are kept; `COCOeval`'s `maxDets` keeps the top 100 per image.
- `coco_map(gt: dict, detections: list[dict]) -> dict[str, float]` runs `pycocotools` `COCOeval(iouType="bbox")` with default parameters, with its stdout suppressed. It returns:
  - `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl` (COCO `stats[0:6]`; size buckets on the 640×512-normalised areas that `to_coco_gt` writes);
  - `AP_<class>` and `AP50_<class>` for each of `CLASS_NAMES`, from `eval["precision"][:, :, k, 0, -1]` (area `all`, `maxDets` 100), averaging entries `> -1`. A class with no GT in `gt` gets `nan`.
- Empty `detections` → every value `0.0` except `nan` classes, without raising.

**FR5 — Training module (`lib/irwm/detection/lightning.py`).**
- `class DetectorModule(lightning.LightningModule)` with `__init__(self, *, detector: Mapping[str, Any], encoder: Mapping[str, Any], optim: Mapping[str, Any], val_gt: dict | None = None, encoder_module: nn.Module | None = None, perf_every: int = 50)`; `save_hyperparameters(ignore=["val_gt", "encoder_module"])`.
  - `self.detector = DINODetector(**detector)` (F7, trainable).
  - `self.encoder`: `encoder_module` if given (tests), else `DinoV2RegEncoder.from_assets(precision=encoder["precision"], hub_version=encoder["hub_version"])` followed by `set_attention(self.encoder.backbone, encoder["attention"])`. `encoder["attention"]` defaults to `"sdpa"`. The encoder stays frozen (F4 already forces `eval()` and `requires_grad_(False)`).
  - `self.image_size = POLICY_SIZE[encoder["policy"]]`.
- `encode(images: Tensor[B, 3, H, W]) -> Tensor[B, 1152, H', W']`: `self.encoder.encode_pca(images).permute(0, 3, 1, 2).contiguous()`.
- `training_step(batch)`: encode → `self.detector(feats, batch["targets"], image_size=self.image_size)` → `loss = sum(losses.values())`. It logs `train/loss` (per step and as an epoch mean, `on_step=True, on_epoch=True`) and the six base losses (`loss_class`, `loss_bbox`, `loss_giou` and their `_dn`) as `train/<name>`, the learning rate as `train/lr`, and, once per epoch at `on_train_epoch_end`, the GradScaler scale as `train/grad_scale` (when a scaler exists) *(amendment 2026-10-27)*. A non-finite `loss` increments `train/nonfinite_steps` (the step is still returned; the 16-mixed GradScaler skips the update). 100 consecutive non-finite steps → `RuntimeError`.
- `validation_step(batch)`: encode → eval-mode detector → `to_flir_detections`, appended to a per-epoch list. `on_validation_epoch_end`:
  1. gather the lists from all ranks (`all_gather_object`), dedupe by `image_id` (DDP sampler padding);
  2. on rank 0, `coco_map(self.val_gt, detections)`, and write `<metrics_dir>/epoch_<NNN>.json` (`NNN` = `current_epoch`, zero-padded) with metrics, epoch, global step and `git_info()`. `metrics_dir` is the attribute `self.metrics_dir: Path | None` (default `None`, which skips the JSON); the entry point sets it to `<hydra run dir>/val_metrics`;
  3. broadcast the metrics dict from rank 0, then log every key as `val/<key>` with `sync_dist=False` on every rank.
  - During Lightning's sanity check (`trainer.sanity_checking`), the validation steps run as usual but `on_validation_epoch_end` only discards the collected detections: no mAP, no JSON, no `val/*` logging *(amendment 2026-10-26)*.
- `configure_optimizers()`:
  - `torch.optim.AdamW(get_default_optimizer_params(self.detector, base_lr=optim["lr"], weight_decay_norm=0.0, lr_factor_func=lambda name: optim["backbone_lr_factor"] if "backbone" in name else 1.0), lr=optim["lr"], betas=tuple(optim["betas"]), weight_decay=optim["weight_decay"])`. Only `self.detector` parameters enter the optimiser.
  - `torch.optim.lr_scheduler.MultiStepLR(milestones=[optim["lr_drop_epoch"]], gamma=0.1)` with `interval="epoch"`. `optim["lr_drop_epoch"] = None` means no scheduler (constant lr), used by overfit mode *(amendment 2026-10-26)*.
- Timing (FR10 needs it): on every `perf_every`-th train batch (constructor kwarg `perf_every: int = 50`), with `torch.cuda.synchronize()` at the batch start, after `encode` and at the batch end, it measures `encode_ms` and `step_ms` (whole batch: encode + forward + backward + optimiser). At `on_train_epoch_end` it logs the epoch's medians over the timed batches as `perf/encode_ms`, `perf/step_ms` and `perf/encode_share` (median of `encode_ms / step_ms`) *(amendment 2026-10-27: Lightning writes `on_step` values only every `log_every_n_steps` steps, so per-step logs on other steps were dropped)*. Per epoch it also logs `perf/epoch_s` and `perf/train_images_per_s` (wall clock between `on_train_epoch_start` and `on_train_epoch_end`, images = batches × batch size).
- Checkpoints carry no encoder weights: `on_save_checkpoint` removes every `state_dict` key starting with `encoder.`; `on_load_checkpoint` re-inserts the current `self.encoder.state_dict()` under the `encoder.` prefix, so Lightning's strict load passes.
- `on_save_checkpoint` also writes `checkpoint["irwm"] = {"metrics_dir": str(self.metrics_dir) if self.metrics_dir else None}`, so the export (FR6) finds the val metrics of the checkpoint's epoch.

**FR6 — Frozen export and loader (`lib/irwm/detection/frozen.py`).**
- `FROZEN_NAMES = ("det_rgb", "det_ir")`, `FROZEN_MODALITY = {"det_rgb": "rgb", "det_ir": "thermal"}`; `frozen_dir(name) -> Path` = `ckpt_root() / "detection" / "frozen" / name`. Another name → `ValueError`.
- `export_frozen(name: str, ckpt_path: Path, *, overwrite: bool = False) -> Path` loads the Lightning checkpoint on CPU, takes the `detector.` keys (prefix stripped, so keys start with `detr.`), and writes into `frozen_dir(name)`:
  - `head.pt`: `torch.save` of that `state_dict`, unchanged. These are the fp32 master weights that `16-mixed` trains; they are not cast to fp16 (plan.md "Precision");
  - `meta.json`: `name`, `modality` (`FROZEN_MODALITY[name]`), `precision: "16-mixed"` (every consumer runs the head under CUDA fp16 autocast), `source_ckpt` (relative to `ckpt_root()`), `epoch`, `global_step`, `val_metrics` (the `val_metrics/epoch_<NNN>.json` content of that epoch when found, else `null`), `detector` and `encoder` hyperparameters from the checkpoint (`encoder` includes `policy`, `precision`, `hub_version`, `attention`, plus `encoder_tag(policy)`), `image_size`, `head_sha256`, `git` (export time), `created_at` (ISO 8601).
  - An existing `head.pt` with a different SHA-256 and `overwrite=False` → `FileExistsError`. Same SHA → no-op, returns the dir.
- `load_frozen_detector(name: str, *, device: str | torch.device = "cpu") -> DINODetector`: checks `head.pt`'s SHA-256 against `meta.json` (`ValueError` on mismatch), builds `DINODetector(**meta["detector"])`, loads with `strict=True`, then `eval()` and `requires_grad_(False)`.
- `frozen_meta(name: str) -> dict` returns `meta.json`.

**FR7 — Hydra configs (`projects/detection/conf/`).**
- `train.yaml` (`defaults: [irwm_base, {experiment: ???}, _self_]`, `project: detection`) with the keys below. `experiment` is mandatory: the run must state which detector it trains.
  - `data: {batch_size: 16, num_workers: ${cluster.num_workers}, flip_p: 0.5}`;
  - `optim: {lr: 1.0e-4, betas: [0.9, 0.999], weight_decay: 1.0e-4, backbone_lr_factor: 1.0, lr_drop_epoch: 40}`;
  - `trainer: {max_epochs: 50, precision: 16-mixed, accumulate_grad_batches: 1, gradient_clip_val: 0.1, check_val_every_n_epoch: 5, log_every_n_steps: 50}`;
  - `encoder: {attention: sdpa}` (merged into the shared `encoder` group). The shared `lib/irwm/conf/encoder/default.yaml` gains `attention: plain`, so the key exists in every config and F4's tools keep plain attention;
  - `overfit: {enabled: false, n_frames: 256, max_steps: 3000, eval_every: 500, ap50_bar: 0.8, classes: [person, car]}`.
- `experiment/det_rgb.yaml` → `modality: rgb`, `name: det_rgb`; `experiment/det_ir.yaml` → `modality: thermal`, `name: det_ir`. Both are `# @package _global_` and add `wandb.tags: [<name>]`.
- `trainer.precision` is read from `train.yaml`, never from `cluster.precision`.

**FR8 — Training entry point (`projects/detection/train.py`).**
- `@hydra.main(config_name="train")`, `load_env()` at import, `seed_everything(cfg.seed, workers=True)`, W&B through `make_wandb_logger`, `plugins=cluster_plugins()`, checkpoint dir `${paths.ckpt_root}/detection/${run_name}/` as `default_root_dir`, resume with `resume_ckpt_path` (TechStack §2, §5). It exits non-zero with a clear message when `cuda_ops_available()` is false.
- Normal mode: `ModelCheckpoint(dirpath=ckpt_dir, monitor="val/AP", mode="max", save_top_k=1, save_last=True, filename="best-e{epoch:03d}", auto_insert_metric_name=False)`. Lightning `Trainer` gets every `cfg.trainer` key. At the end, rank 0 writes `<run_dir>/train_summary.json`: best checkpoint path, best `val/AP`, its epoch, `n_train_frames`, `n_dropped_empty`, `n_val_frames`, wall seconds of the fit, `data_root` (the resolved `irwm.data.flir.flir_root()`, which shows whether the data was staged), and `git_info()`.
- Overfit mode (`overfit.enabled=true`): train and val on the same `first_nonempty_ids(train_split, overfit.n_frames)` frames, both from the `train` split (`val_split="train"`), `flip_p=0`, `max_steps=overfit.max_steps`, `max_epochs=-1` (so `max_steps` governs: 256 frames are only 16 steps per epoch), constant lr (`optim.lr_drop_epoch=None`: the 50ep milestone at epoch 40 would fall at step 640) *(amendment 2026-10-26)*, `val_check_interval=overfit.eval_every`, `check_val_every_n_epoch=None`, and a `ModelCheckpoint(dirpath=ckpt_dir, save_top_k=0, save_last=True)` only. A callback stops training at the first (non-sanity-check) validation where `AP50_<c> ≥ overfit.ap50_bar` for every `c` in `overfit.classes`. Rank 0 writes `<run_dir>/overfit.json`: `passed`, the step of the pass (or `null`), the metrics of every validation, median `perf/step_ms`, peak `torch.cuda.max_memory_allocated()` in GiB, `data_root`, and `git_info()`.

**FR9 — Export entry point (`projects/detection/export.py`).**
- `@hydra.main(config_name="export")` with `export.yaml`: `defaults: [irwm_base, {experiment: ???}, _self_]`, `project: detection`, `source_run: ???` (a run name under `${paths.ckpt_root}/detection/`), `overwrite: false`.
- It picks the single `best-e*.ckpt` in `${paths.ckpt_root}/detection/${source_run}/` (zero or several → `FileNotFoundError`/`ValueError` listing them), calls `export_frozen(cfg.name, ...)`, and prints the export dir and `head_sha256`.

**FR10 — Training runs and results note (`spec/2026-10-26-detector-training-pipeline/results.md`).**
- Overfit run per modality on ctcomp3 (L4): `passed: true` with the bar of FR8, and peak memory at batch 16.
- One 50-epoch run per modality on ctcomp3, exported as `det_rgb` and `det_ir`.
- The note records, per detector: best epoch, val AP / AP50 / AP75 / APs / APm / APl, per-class AP and AP50, val AP at each of the 10 evaluations, total training wall time and GPU-hours (including requeues), number of SLURM jobs, staging seconds per job (from the `[stage]` log line), median `perf/step_ms`, median `perf/encode_share`, train images/s, peak memory, `n_train_frames` / `n_dropped_empty`, W&B run ids, SLURM job ids, export SHA-256. It states whether encoding is the bottleneck (median `perf/encode_share > 0.5`) and, if so, that F5 must be revisited.
- **Done bar (user decision):** val AP50 ≥ 0.30 for `person` and for `car`, in each of `det_rgb` and `det_ir`.

**FR11 — Tests (`tests/test_detector_training.py`).**
- CPU tests (default suite, no data, no weights, < 60 s on ctdesk236) cover FR1–FR7 and FR9 (validation.md Groups 1–7). They build a tiny FLIR tree in `tmp_path` and use a stub encoder. They run in fp32 only because CUDA fp16 autocast does not exist on CPU; this is the only fp32 path in F8.
- GPU tests (`@pytest.mark.gpu`, skipped with a reason if `cuda_ops_available()` is false) cover a real `Trainer.fit` step and checkpoint round trip with a stub encoder (Group 8).
- One `@pytest.mark.data` + `gpu` test runs the real encoder on 2 real thermal val frames (Group 9).

## Public API Summary

```python
# lib/irwm/features/preprocess.py  (addition)
class Preprocess:
    def geometry(self, width: int, height: int) -> ResizeGeometry: ...

# lib/irwm/data/detection.py
class FlirDetectionDataset(Dataset):
    n_dropped_empty: int
    def __init__(self, split: FlirSplit, *, policy: Policy = "native", flip_p: float = 0.0,
                 filter_empty: bool = False, image_ids: Sequence[int] | None = None): ...
    def __getitem__(self, i: int) -> dict: ...  # image [3,H,W], boxes [N,4] xyxy px, labels [N], image_id, geometry, flipped
def collate_detection(items: list[dict]) -> dict: ...  # images [B,3,H,W], targets, image_ids, geometries
def first_nonempty_ids(split: FlirSplit, n: int, policy: Policy = "native") -> list[int]: ...
class FlirDetectionDataModule(L.LightningDataModule):
    val_gt: dict
    def __init__(self, *, modality: Modality, policy: Policy = "native", batch_size: int = 16,
                 num_workers: int = 0, flip_p: float = 0.5, train_image_ids: Sequence[int] | None = None,
                 val_split: Literal["val", "train"] = "val", val_image_ids: Sequence[int] | None = None): ...

# lib/irwm/detection/coco_val.py
def to_flir_detections(preds: list[dict[str, Tensor]], image_ids: list[int],
                       geometries: list[ResizeGeometry]) -> list[dict]: ...
def coco_map(gt: dict, detections: list[dict]) -> dict[str, float]: ...

# lib/irwm/detection/lightning.py
class DetectorModule(L.LightningModule):
    def __init__(self, *, detector: Mapping[str, Any], encoder: Mapping[str, Any], optim: Mapping[str, Any],
                 val_gt: dict | None = None, encoder_module: nn.Module | None = None, perf_every: int = 50): ...
    def encode(self, images: Tensor) -> Tensor: ...  # [B,3,H,W] -> [B,1152,H',W']

# lib/irwm/detection/frozen.py
FROZEN_NAMES: tuple[str, ...]  # ("det_rgb", "det_ir")
FROZEN_MODALITY: dict[str, str]  # {"det_rgb": "rgb", "det_ir": "thermal"}
def frozen_dir(name: str) -> Path: ...
def export_frozen(name: str, ckpt_path: Path, *, overwrite: bool = False) -> Path: ...
def load_frozen_detector(name: str, *, device: str | torch.device = "cpu") -> DINODetector: ...
def frozen_meta(name: str) -> dict: ...

# lib/irwm/detection/__init__.py additionally re-exports: load_frozen_detector, frozen_meta
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/` `images_{rgb,thermal}_{train,val}` | read | Through `FlirSplit` / JPEGs (F2); never written |
| `$IRWM_CKPT_ROOT/{dino_foresight,torch_hub}/` | read | DINOv2-Reg weights, pinned hub code, `pca_1152.pt` (F1, F4) |
| `irwm.detection.DINODetector`, `cuda_ops_available` | read (import) | F7, unchanged |
| `irwm.features.encoder` (`DinoV2RegEncoder`, `set_attention`, `encoder_tag`), `irwm.features.preprocess` | read (import), one addition (FR1) | F4, F7 |
| `detectron2.solver.build.get_default_optimizer_params` | read (import) | detectron2 @ `70f4543`, `solver/build.py:142-235` |
| `$LOCAL_SCRATCH/irwm_data/teledyne_flir_adas_v2/` | write (copy), then read | Staged by `slurm/stage_data.sh` at every job start and every requeue (12.2 GB, 46,450 files, ≈ 283 s, R11); wiped at job end |
| `$IRWM_CKPT_ROOT/detection/<run_name>/` | write | Lightning `last.ckpt`, `best-e<NNN>.ckpt`, `hpc_ckpt_*.ckpt` (no encoder weights, ≈ 280 MB each with AdamW state) |
| `$IRWM_CKPT_ROOT/detection/frozen/det_{rgb,ir}/` | write | `head.pt` (≈ 92 MB), `meta.json` |
| `$IRWM_OUTPUT_ROOT/detection/...` | write | Hydra run dirs: `val_metrics/epoch_<NNN>.json`, `train_summary.json`, `overfit.json`, W&B files |
| GPU time on ctcomp3 (L4) | — | ≈ 2 × 1 h overfit + 2 × ≈ 13 h training ≈ 28 GPU-h |
| New deps | — | `pycocotools>=2.0.7` declared explicitly in `lib/pyproject.toml` (already installed through detectron2, now imported directly by `irwm`) |
