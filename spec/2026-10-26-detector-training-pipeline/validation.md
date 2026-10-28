# F8 · Detector training pipeline: Validation

## Code Correctness (`tests/test_detector_training.py`, CPU, no data, no weights, < 60 s on ctdesk236 unless marked)

CPU groups run in fp32 only because CUDA fp16 autocast does not exist on CPU (the encoder warns and runs fp32 there, F4). This is a test-only exception: every GPU group runs `16-mixed`.

### Group 1 — `Preprocess.geometry` (FR1)
- [x] For `native` and `rgb`/`thermal`, `Preprocess(...).geometry(w, h)` equals the geometry returned by `Preprocess(...)(img)` for a `uint8` image of that size, for `(w, h)` in `(80, 64)`, `(100, 64)`, `(640, 512)`, `(1224, 1024)`.
- [x] `geometry(1224, 1024)` under `native` has `crop_box == center_crop_box(1224, 1024, (5, 4))` and `out_size == (560, 448)`; under `stretch` its `crop_box == (0, 0, 1224, 1024)`.
- [x] `uv run pytest tests/test_encoder.py` (F4 default tests) still passes unchanged.

### Group 2 — Dataset (FR2), on the tiny FLIR tree
- [x] `FlirDetectionDataset(thermal train, filter_empty=False)` has `len == number of records`; item `"image"` is `float32 [3, 448, 560]`, `"boxes"` `float32 [N, 4]`, `"labels"` `int64 [N]`.
- [x] **Box mapping:** a thermal 80×64 box xywh `[8, 16, 16, 8]` → xyxy `[56, 112, 168, 168]` (scale 7 on both axes) within `atol=1e-4`.
- [x] **RGB crop:** on a 100×64 RGB frame (crop x = 10..90, scale 7), box xywh `[20, 8, 10, 10]` → xyxy `[70, 56, 140, 126]`; a box entirely in x < 10 is dropped.
- [x] **Ignore classes:** a frame whose only annotation is FLIR category 8 has `N = 0` boxes; no item ever contains a label outside `0..5`.
- [x] **Empty filter:** with `filter_empty=True`, the ignore-only frame and the RGB frame whose only box lies in the cropped edge are both absent, and `n_dropped_empty` equals 2 for RGB train (1 for thermal train, which has no crop loss).
- [x] **Flip:** with `flip_p=1.0`, `item["image"] == unflipped["image"].flip(-1)` exactly, boxes satisfy `x0' = 560 − x1`, `x1' = 560 − x0`, y unchanged, and `item["flipped"] is True`; with `flip_p=0.0`, `flipped is False` for every item.
- [x] **Flip rate:** with `flip_p=0.5` and `torch.manual_seed(0)`, 400 draws of one item give a flipped fraction in `[0.42, 0.58]`.
- [x] `flip_p=1.5` → `ValueError`; `image_ids=[999999]` → `KeyError`.
- [x] `first_nonempty_ids(split, 2)` returns the 2 smallest `image_id`s with ≥ 1 kept box after mapping; `n` above the available count → `ValueError`.

### Group 3 — Collate and DataModule (FR3)
- [x] `collate_detection` of 3 items → `images` `[3, 3, 448, 560]`, 3 targets with the items' boxes/labels, `image_ids` and `geometries` in item order.
- [x] `FlirDetectionDataModule(modality="thermal", batch_size=2).setup("fit")` twice → loads each split once (`FlirSplit.load` called 2 times in total, checked with a spy).
- [x] `val_gt["images"]` ids == the val dataset's ids; its annotations include `iscrowd: 1` entries for the ignore box (one per kept category, F2).
- [x] `train_dataloader()` has `drop_last=True` and shuffles; `val_dataloader()` has `drop_last=False` and keeps ascending `image_id` order.
- [x] `val_split="train", val_image_ids=ids` → val dataset ids == `ids` and `val_gt` contains exactly those images.

### Group 4 — Val mAP (FR4)
- [x] **Perfect predictions:** detections equal to every kept GT box (in FLIR pixels, score 1.0, correct category) → `AP == AP50 == 1.0` within `1e-6`, and `AP_<c> == 1.0` for every class present in the GT.
- [x] **Round trip through the encoder frame:** GT boxes mapped with `apply_boxes` → xyxy → `to_flir_detections` (with the frames' geometries) → `coco_map` gives `AP50 == 1.0` (RGB frames included, so the crop offset is exercised).
- [x] **Ignore regions:** a detection placed exactly on an ignore box (any category) leaves `AP` unchanged versus the same detections without it.
- [x] **Wrong class:** perfect boxes with every label replaced by `(label + 1) % 6` → `AP == 0.0`.
- [x] `to_flir_detections` maps label `4` → `category_id 10` and label `5` → `12`, and converts xyxy `[10, 20, 30, 60]` to an xywh with `w = 20/sx`, `h = 40/sy`.
- [x] Empty detections → all metrics `0.0` (classes without GT `nan`), no exception; a class absent from the GT → `AP_<c>` is `nan`.
- [x] `coco_map` prints nothing to stdout (`capsys`).

### Group 5 — `DetectorModule` on CPU (FR5), stub encoder, reduced detector
- [x] `encode(images [2, 3, 448, 560])` → `float32 [2, 1152, 32, 40]`, contiguous.
- [x] **Optimiser groups:** every `detector` parameter appears in exactly one group and no encoder parameter does; `torch.nn.LayerNorm`/`GroupNorm` params have `weight_decay == 0.0`, a `Linear` weight has `1e-4`; with `backbone_lr_factor=1.0` every lr is `1e-4`, with `0.1` exactly the `detr.backbone.*` params have `1e-5`.
- [x] **Schedule:** stepping the scheduler 39 times keeps lr `1e-4`; after 40 steps lr is `1e-5`; after 50, still `1e-5`.
- [x] **Checkpoint stripping:** a checkpoint dict produced by `on_save_checkpoint` has no key starting with `encoder.` and has `checkpoint["irwm"]["metrics_dir"]`; after `on_load_checkpoint`, `load_state_dict(strict=True)` on a fresh module succeeds.
- [x] **Non-finite guard:** feeding 100 consecutive `nan` losses through the counter raises `RuntimeError`; 99 followed by a finite loss resets it and does not raise.
- [x] With `encoder_module=None` and no assets in the tmp `$IRWM_CKPT_ROOT`, construction raises `FileNotFoundError` naming `download.py` (F4 behaviour surfaces unchanged).

### Group 6 — Frozen export and loader (FR6), CPU, reduced detector
- [x] A fake Lightning checkpoint (`state_dict` with `detector.*` keys, `hyper_parameters`, `epoch: 9`, `irwm.metrics_dir` pointing to a dir with `epoch_009.json`) → `export_frozen("det_ir", ...)` writes `head.pt` with keys starting `detr.` only, and `meta.json` with `modality: "thermal"`, `precision: "16-mixed"`, `epoch: 9`, `val_metrics` = that JSON, `encoder.attention`, `encoder_tag`, `image_size: [448, 560]`, a 64-hex `head_sha256`.
- [x] `load_frozen_detector("det_ir")` returns a `DINODetector` with `training is False`, every `requires_grad is False`, and parameters equal to the source `detector.*` tensors.
- [x] Re-export of the same checkpoint → no-op; export of a different checkpoint → `FileExistsError`; with `overwrite=True` → replaced.
- [x] Flipping one byte of `head.pt` → `load_frozen_detector` raises `ValueError` (SHA mismatch).
- [x] `frozen_dir("det_x")` → `ValueError`; every path written is under the tmp `$IRWM_CKPT_ROOT`.

### Group 7 — Configs and entry points (FR7, FR9)
- [x] Composing `train` with `experiment=det_rgb` → `modality == "rgb"`, `name == "det_rgb"`, `trainer.precision == "16-mixed"`, `data.batch_size == 16`, `trainer.max_epochs == 50`, `optim.lr_drop_epoch == 40`, `optim.backbone_lr_factor == 1.0`, `trainer.check_val_every_n_epoch == 5`, `encoder.attention == "sdpa"`, `encoder.policy == "native"`; with `experiment=det_ir` → `modality == "thermal"`.
- [x] Composing `train` without `experiment` raises a Hydra `MissingMandatoryValue`/`ConfigCompositionException`.
- [x] Composing `irwm_base` alone gives `encoder.attention == "plain"`.
- [x] `export.py`'s checkpoint pick: a run dir with one `best-e004.ckpt` → that file; none → `FileNotFoundError`; two → `ValueError` listing both.
- [x] Running `projects/detection/train.py experiment=det_ir` on a machine without CUDA ops exits with a non-zero status and a message containing `cuda_ops_available`.

### Group 8 — GPU training path (`@pytest.mark.gpu`, ctcomp3 L4)
- [x] `Trainer(max_steps=4, precision="16-mixed", limit_val_batches=1, check_val_every_n_epoch=1)` fit on the tiny FLIR tree (thermal, batch 2, stub encoder, reduced detector) completes; `train/loss` is finite at every logged step; `val/AP` is present in `callback_metrics`.
- [x] The saved `last.ckpt` has no `encoder.*` key, and resuming from it (`ckpt_path=last`) for 2 more steps works and continues `global_step` from 4.
- [x] With a `ModelCheckpoint(monitor="val/AP")` saving at the end of epoch 0's validation, the checkpoint's `epoch` equals the `NNN` of the `val_metrics/epoch_<NNN>.json` written in that validation (FR6 lookup).
- [x] *(accepted from the 50-epoch runs, user decision 2026-10-28: jobs 5526115/5526116 at `b81ef6c` logged all three `perf/*` medians and `train/grad_scale` in each of the 50 epochs, `encode_share` 0.118–0.123, `grad_scale` 512–4096; the test itself was not re-run)* `perf/encode_ms`, `perf/step_ms`, `perf/encode_share` are logged at the end of epoch 0 (medians of the timed batches; batch 0 is timed) with `0 < encode_share < 1`, and `train/grad_scale > 0` is logged per epoch *(amended 2026-10-27; the step-0 version passed on hpc-gpu5 at `6425d4c`, re-run pending)*.

### Group 9 — Real encoder (`@pytest.mark.data` and `@pytest.mark.gpu`, ctcomp3 L4)
- [x] `DetectorModule` with the real `DinoV2RegEncoder` (SDPA, fp16) on 2 real thermal val frames, under `torch.autocast("cuda", dtype=torch.float16)` as `16-mixed` does: `encode` → `[2, 1152, 32, 40]` finite, and one training step with the default (non-reduced) detector gives a finite loss.

## Data Validity (evidence: `overfit.json`, `train_summary.json`, `val_metrics/`, export `meta.json`, W&B)
- [x] **Overfit, RGB and thermal (user decision bar):** `overfit.json` has `passed: true`, with `AP50_person ≥ 0.8` and `AP50_car ≥ 0.8` at some validation within ≤ 3000 steps. If either fails, Step 10 does not start.
- [x] **fp16 health (deferred fallback, plan.md):** in every overfit and full run, `train/nonfinite_steps` stays < 1 % of steps and the 100-consecutive abort never fires. If one of these fails, the implementer stops and reports (plan.md "Precision"); the item stays open until you decide.
- [x] **Staging happened:** every job log of Steps 9–10 (each requeue included) shows `[stage] copied <n> files` with `n == 46450` and `IRWM_DATA_ROOT=<$LOCAL_SCRATCH>/irwm_data`, and every `train_summary.json` / `overfit.json` has `data_root` under `$LOCAL_SCRATCH`.
- [x] *(det_ir on V100S accepted, user decision 2026-10-27)* **Memory at batch 16:** overfit peak `torch.cuda.max_memory_allocated` ≤ 22 GiB on the L4 for both modalities, or the OOM plan was applied (batch 8 × accumulate 2), recorded as an amendment, and the overfit re-run passes at batch 8.
- [x] **Training completed:** each 50-epoch run reaches `trainer.current_epoch == 50` (10 validations logged at epochs 5, 10, …, 50), with `train/nonfinite_steps` total < 1 % of all steps.
- [x] **Learning:** `train/loss` averaged over epoch 50 is ≤ 50 % of its epoch-1 average, for each detector.
- [x] **Done bar (user decision):** best-checkpoint val `AP50_person ≥ 0.30` and `AP50_car ≥ 0.30` for `det_rgb` and for `det_ir`.
- [x] **lr drop visible:** `train/lr` is `1e-4` before epoch 40 and `1e-5` from epoch 40 on.
- [x] **Frame counts:** `n_train_frames + n_dropped_empty` equals the train split size (RGB 10,318, thermal 10,742, F2), and `n_val_frames` equals the val split size (RGB 1,085, thermal 1,144).
- [x] **Encoder share reported:** median `perf/encode_share` is recorded per detector; the results note states "encoding is the bottleneck, revisit F5" if it is > 0.5, otherwise that it is not.
- [x] **Results note complete:** `results.md` contains every item listed in FR10.

## Data Architecture Integrity
- [x] **Frozen encoder:** no encoder parameter is in any optimiser group (Group 5), and no checkpoint of the real runs contains an `encoder.*` key (checked by the implementer with `torch.load` on both `best-e*.ckpt` files during Step 11).
- [x] **Exports are self-describing:** each real `meta.json` records `source_ckpt`, `epoch`, non-null `val_metrics` whose `AP` equals `train_summary.json`'s best `val/AP` within `1e-6`, `encoder.attention == "sdpa"`, `precision == "16-mixed"`, `encoder_tag == "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152"`, `git.commit` and `head_sha256`; `load_frozen_detector(name)` succeeds on ctdesk236 or the ctcomp3 login node (CPU).
- [x] **Provenance:** every `val_metrics/epoch_<NNN>.json`, `train_summary.json` and `overfit.json` contains `git_info()`; every W&B run has the resolved config and git info (`make_wandb_logger`).
- [x] **Paths:** nothing is written outside `$IRWM_CKPT_ROOT/detection/`, `$IRWM_OUTPUT_ROOT/detection/` (and `slurm_logs/`) and the read-only staged copy under `$LOCAL_SCRATCH/irwm_data/`; `grep -rnE '"/(home|mnt|scratch|tmp)/' lib/irwm/data/detection.py lib/irwm/detection/ projects/detection/` finds nothing.
- [ ] **Read-only data and third-party code:** `git status --porcelain` is empty in `third_party/detrex` and `third_party/detrex/detectron2` after the runs; no file under the shared FLIR copy (`$IRWM_DATA_ROOT` before staging) changes mtime.
- [x] **fp16 everywhere:** the resolved config of every Step 9–10 W&B run has `trainer.precision == "16-mixed"` and `encoder.precision == "fp16"`; no run used `32-true`.
- [~] *(deferred, user decision 2026-10-28: the forced requeue was not done, so both runs are `restart 0` with no `hpc_ckpt_*`; only Group 8's `last.ckpt` resume is proven. Revisit at the first run that needs a requeue, or before a run longer than the partition's practical uptime)* **Resumability:** the `det_rgb` 50-epoch run was requeued by the forced signal of plan.md Step 10 (the partition has no wall limit) and resumed from `hpc_ckpt_*.ckpt` with `global_step` continuing from the pre-requeue value, shown in the job log and in W&B.
- [x] **Default suite unaffected:** `uv run pytest` (default markers) passes on ctdesk236 with the new tests included.
