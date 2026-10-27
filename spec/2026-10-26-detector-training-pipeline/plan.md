# F8 · Detector training pipeline: Plan

## Context and Design Decisions

### Recipe (detrex's DINO-ViTDet, adapted to FLIR)
- **Schedule = detrex's 50ep (user decision, 2026-10-26).** 50 epochs, lr ×0.1 from epoch 40, no warmup (`detrex/config/configs/common/coco_schedule.py`: `lr_multiplier_50ep = default_coco_scheduler(50, 40, 0)`; `dino_vitdet_base_4scale_50ep.py:61`). detrex's scheduler is iteration-based but scale-invariant, so an epoch-interval `MultiStepLR(milestones=[40], gamma=0.1)` is the same schedule. Rejected: 12ep (≈ 8k iterations at batch 16 on 10.5k frames against detrex's 90k on COCO, so likely under-trained) and 24ep. Cost ≈ 13 h per detector on an L4 (budget below).
- **Optimiser = detrex's 12ep/50ep AdamW, lr 1e-4, betas (0.9, 0.999), weight decay 1e-4, grad clip L2 0.1** (`dino_vitdet_base_4scale_12ep.py:28-30,37-39`; the 50ep file imports it). Param groups come from detectron2's own `get_default_optimizer_params(..., weight_decay_norm=0.0)` (`common/optim.py` AdamW; `detectron2/solver/build.py:142-235`), so `torch.nn.LayerNorm`/`GroupNorm` parameters get no weight decay, exactly as in detrex. Note: the pyramid's `"LN"` is detectron2's own `LayerNorm` class (not in `norm_module_types`, `build.py:203-215`), so it keeps weight decay, also exactly as in detrex.
- **Backbone lr factor = 1.0, not detrex's 0.1 (user decision).** detrex multiplies the lr of every parameter whose name contains `backbone` by 0.1 (`dino_vitdet_base_4scale_12ep.py:40`), meant for the pretrained MAE ViT. Our `detr.backbone.*` is the new `FeatureStem` plus the simple feature pyramid, all randomly initialised, so the reason does not apply. **Deviation E1.** Kept as `optim.backbone_lr_factor` so 0.1 can be tried later.
- **Batch 16 per step (user decision), effective batch = detrex's 16** (`dino_vitdet_base_4scale_12ep.py:48`). F7 measured 9.07 GiB for the head at B = 8 fp16 with 5 synthetic boxes; B = 16 is ≈ 18 GiB plus the encoder (SDPA, B = 16: ≈ 1 GiB) and real box counts, tight on the L4's 24 GB. **OOM plan (user decision):** the overfit step (Step 9) records peak memory at B = 16. If it OOMs or peaks above 22 GiB, set `data.batch_size: 8` and `trainer.accumulate_grad_batches: 2` in `train.yaml` (same effective batch and lr; `MultiStepLR` is per epoch, so unaffected), record an amendment, and re-run the overfit at B = 8. The training code has no automatic fallback.
- **Precision = fp16 everywhere on the GPU (user decision, 2026-10-26).** Training, validation, the overfit and every later use of the frozen heads run in Lightning `16-mixed` (CUDA fp16 autocast + GradScaler). The encoder keeps its fp16 autocast (TechStack §3.1). F7: 13.1 img/s fp16 vs 8.2 fp32 (1.6×). detrex itself trains in fp32 (`common/train.py`: `amp.enabled=False`), so this is **deviation E2**. bf16 is not possible: detrex's CUDA MSDA casts only fp16 `value` to fp32 (`detrex/layers/multi_scale_deform_attn.py:341-356`), and V100S nodes have no bf16.
  - **What `16-mixed` keeps in fp32, and why `head.pt` stays fp32.** Autocast runs matmuls, convolutions and attention in fp16. The parameters remain fp32 master weights, updated in fp32 by AdamW: an update of ≈ lr 1e-4 × grad is below fp16's ≈ 1e-3 relative resolution for weights of order 1, so it would round to zero in fp16. Normalisation layers, softmax and the losses also read the fp32 weights under autocast. The trained weights therefore use all fp32 bits, and casting `head.pt` to fp16 would export a head that differs from the one validated. `meta.json` records `precision: "16-mixed"` so F9/F12 also run the head under fp16 autocast.
  - **The CPU unit tests are the only place F8 itself runs in fp32**, because CUDA fp16 autocast does not exist on CPU. The ops autocast keeps in fp32 by design (normalisation, softmax, losses) are part of `16-mixed` and are not a deviation.
  - **`trainer.precision` comes from `train.yaml`, never from `cluster.precision`.** `cluster/slurm.yaml` was switched to `precision: 16-mixed` on 2026-10-26 (user decision, made while writing this spec), but `cluster/local.yaml` keeps `32-true` for CPU runs. F8's GPU precision must not depend on which cluster group is selected.
  - **Deferred: what to do if fp16 misbehaves** (user decision, 2026-10-26: "test the training, then decide"). Trouble means any of these: the overfit fails its bar, `train/nonfinite_steps` reaches ≥ 1 % of steps, or the 100-consecutive abort fires (FR5). The GradScaler skips steps with inf/NaN gradients by itself. If trouble happens, the implementer **stops and reports**: which losses went non-finite, at which steps, and the GradScaler scale (`trainer.precision_plugin.scaler.get_scale()`, logged as `train/grad_scale` every 50 steps by FR5's logging). It does not switch to fp32 or to any other technique without a new user decision.
- **Augmentation = horizontal flip, p = 0.5, train only (user decision).** detrex uses `RandomFlip` plus multi-scale resize and crop (`common/data/coco_detr.py`). Our token grid is fixed at 32×40 (R5 `native`), and F9/F12 apply the frozen heads to 448×560 features, so multi-scale would train on scales never seen at test. The flip is applied to the preprocessed image **before** the encoder (DINOv2 features are not flip-equivariant, so flipping features would be a different augmentation). It is identical for both modalities.
- **Ignore regions are dropped from targets (user decision, 2026-10-23, F7 plan.md "Notes for F8"):** "ignored classes are like they don't exist in the dataset; the model shouldn't know about them". In val mAP they stay crowd GT as F2 fixed it (`FlirSplit.to_coco_gt`, TechStack §3.4).
- **Empty train frames are dropped, val frames are all kept.** detrex trains through `get_detection_dataset_dicts`, whose default `filter_empty=True` (`detectron2/data/build.py:217-219`) drops images without annotations; detrex's test loader sets `filter_empty=False`. "Empty" is decided **after** the ignore classes are dropped and after the `native` crop mapping, so an RGB frame whose only boxes lie in the cropped edges is also dropped. The counts are logged (`n_dropped_empty`).
- **Box mapping uses F4's `ResizeGeometry.apply_boxes` keep rule (`w > 0 and h > 0` after clipping to the crop box, `preprocess.py:46-63`).** It is the same rule DINO later inverts with `invert_boxes`. The decision to need geometry without decoding an image (to filter empty frames in `__init__`) is why FR1 adds `Preprocess.geometry`.
- **Encoder attention = SDPA, in training and in every later use of the frozen heads (user decision).** F7: 2.44× faster encoder, drift (mean token cosine 0.9999982) below the plain encoder's own fp16-vs-fp32 drift (TechStack §3.1). The export manifest records `attention: sdpa` so F9/F12 encode with the same attention. Encoder precision stays fp16 autocast (TechStack §3.1, F4).
- **Encoder inside the module, not in the checkpoint.** `DetectorModule.encoder` is a registered submodule, so Lightning moves it to the GPU, but `on_save_checkpoint` strips its 86 M parameters (≈ 350 MB per checkpoint) and `on_load_checkpoint` puts the live encoder's tensors back so the strict load passes. The encoder is rebuilt from the pinned assets at construction (F4 `from_assets`). Rejected: `strict_loading = False`, which would also hide real missing detector keys.
- **FLIR is staged to node-local scratch for every job (user decision, 2026-10-26).** Every Step 9–10 command exports `IRWM_STAGE_DATA=1`, so `slurm/job.sbatch` sources `slurm/stage_data.sh` before the entry point. The script copies FLIR to `$LOCAL_SCRATCH/irwm_data/`, checks the file count and repoints `IRWM_DATA_ROOT` (R11). Reasons:
  - it is cheap: ≈ 283 s per job start (and again per requeue), < 1 % of a 13 h run;
  - it cuts the two `coco.json` loads at job start from ≈ 176 s to ≈ 6 s (R11), which nearly pays for the copy;
  - it takes BeeGFS out of the per-epoch read path. R11's "no end-to-end gain" was measured for the encoder alone (F7 jobs 5525665/5525666), not under the random-access training loader, so the user does not treat it as settled.
  
  The training code does not depend on staging: it only reads `IRWM_DATA_ROOT`. `train_summary.json` and `overfit.json` record `data_root`, so the evidence shows the staged path. **Reminder (R11 / `stage_data.sh`):** the scratch copy is read-only input, so nothing may be written under it.
- **Seeds.** `seed: 0` (irwm_base), `seed_everything(..., workers=True)`. Runs are not bit-reproducible: detrex's CUDA MSDA backward uses atomic adds, and the 16-mixed GradScaler adds more non-determinism. The validation items therefore never ask for bit-equal training.

### Validation metric and checkpoint selection
- **Val mAP = pycocotools on `FlirSplit.to_coco_gt("normalised")`, in FLIR pixels.** TechStack §1 names `pycocotools` as the detection metric, and `to_coco_gt` already encodes ignore regions as per-category crowd GT and 640×512-normalised areas (F2). torchmetrics was rejected: it would need its own crowd encoding of the same thing. Predictions go xyxy (encoder px) → xywh → `ResizeGeometry.invert_boxes` (FLIR px, `preprocess.py:65-71`). `COCOeval` defaults (`maxDets` 1/10/100) are used, as detrex's `COCOEvaluator` does.
- **GT is the full FLIR frame.** Under `native`, RGB frames lose 4–17 % at the left/right edges (R5), so RGB objects there cannot be detected and count as misses. This is the honest protocol (Mission: each detector is scored on its own modality's annotations) and the one F9 will use. It lowers RGB val AP a little; the results note says so.
- **Selection = best `val/AP`, validated every 5 epochs (user decision).** 10 validations over 50 epochs (epochs 5, 10, …, 50, Lightning's `check_val_every_n_epoch=5` counts from 1). The best of them is exported. Rejected: every epoch (≈ 1–2 min each, 50–100 min per run) and last-epoch-only.
- **Done bar (user decision): val AP50 ≥ 0.30 for `person` and for `car`, in each detector.** Low enough not to presume H1, high enough to rule out a broken pipeline (≈ 86–89 % of FLIR boxes are small, R3). Person and car are the classes not distorted by the annotation asymmetry (R10).
- **Overfit bar (user decision): AP50 ≥ 0.8 for person and car** on the first 256 non-empty train frames (ascending `image_id`), trained and evaluated on those frames with the same COCO path, no flip, ≤ 3000 iterations at batch 16, evaluated every 500. 3000 × ≈ 1.2 s ≈ 1 h per modality at most. It tests every link F7's synthetic overfit could not: JPEG → preprocess → encoder → box mapping → loss → inversion to FLIR pixels → COCOeval. Failing it is a stop condition: the implementer reports and does not start Step 10.

### Budget (L4, from F7 results.md §4, §6)
- Per epoch: RGB ≈ 10.3k frames × (1/13.1 + 1/89.3) s ≈ 15 min, thermal ≈ 15.7 min (fewer after empty filtering). 50 epochs + 10 validations ≈ **13 h per detector, ≈ 26 GPU-h for both**, plus ≈ 2 h of overfit runs. Real FLIR frames carry more boxes than F7's 5, so matching may be slower; the results note records the measured figures.
- **SLURM time (user, 2026-10-26): the ctcomp3 partition has no wall limit, but a fair-use policy lowers job priority as usage grows.** Each full run is therefore one job with `cluster.timeout_min=1200` (20 h ≈ 1.5× the ≈ 13 h estimate plus staging). An explicit limit stops a hung job from burning fair-share usage for days. If a run overruns, Lightning's `SLURMEnvironment(auto_requeue=True)` (TechStack §5, `irwm.utils.slurm.cluster_plugins`) checkpoints on `SIGUSR1` 90 s before the limit (`slurm/job.sbatch`), and the job resumes under the same `run_name` (`slurm-<job id>`). Both runs may be submitted together: their fair-share usage is the same either way, and together they finish sooner.
- **Resume must still be proven.** With no wall limit, no requeue happens on its own, so Step 10 forces one with `scancel --signal=USR1 --batch` on the `det_rgb` run. The overfit jobs use `cluster.timeout_min=120`.
- Resources per job: `cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32` (F7 measured the loader with 8 workers; F4 found 2 workers already enough at 36 fps, so 8 is margin).

### Code facts the implementer relies on
- **`DINODetector` contract (F7, `lib/irwm/detection/detector.py:98-133`).** Train mode needs CUDA features and one target dict per image with `boxes` xyxy encoder px and `labels` int64; `N = 0` is allowed. Eval mode returns ≤ 300 xyxy boxes clipped to `image_size`. `image_size` must be passed as `(448, 560)` (`POLICY_SIZE["native"]`, `preprocess.py:30`); the default `14·grid` gives the same value but is passed explicitly for clarity.
- **The encoder returns channels-last `[B, H', W', 1152]`, computed in fp16 and cast to an fp32 tensor** (`encoder.py:154-155`, F4 behaviour, not changed here). Under `16-mixed` the head's first matmul casts it back to fp16, so this is a dtype of the tensor at the interface, not an fp32 computation; the detector needs `[B, 1152, H', W']` (TechStack §7). `encode_pca` runs under `no_grad` (not `inference_mode`), so its output can feed a trainable head (`encoder.py:12-15`).
- **`DinoV2RegEncoder.train()` always stays in eval** (`encoder.py:125-127`), so Lightning's `.train()` calls are harmless; `requires_grad_(False)` is set in `__init__` (`encoder.py:101`), so `get_default_optimizer_params(self.detector, ...)` would skip encoder parameters anyway (it ignores `requires_grad=False`), but only `self.detector` is passed.
- **`FlirSplit(modality, split, root, records)` is a public constructor** (`flir.py:121-131`), so a subset GT is `FlirSplit(..., selected_records).to_coco_gt()`. `to_coco_gt` assigns crowd ids from `max(ann id)+1` of the records it sees, which is fine for a subset.
- **`FlirImageDataset` decodes and checks mode and size** (`flir.py:298-310`). `FlirDetectionDataset` reuses its decoding by holding a `FlirImageDataset(split)` without transform and indexing it by the record position.
- **Hydra mandatory group.** `defaults: [irwm_base, {experiment: ???}, _self_]` makes Hydra fail with "You must specify 'experiment'" when it is missing. `slurm/submit.sh` forwards `experiment=det_rgb` unchanged (it only reads `cluster.*` for the sbatch flags, `irwm/utils/slurm.py:243-245`).
- **Detection tests need a fake FLIR tree.** The F2 fixture `fake_flir` lives in `tests/test_flir_data.py`, not in `conftest.py`. F8's test file builds its own tiny tree with a local helper instead of moving F2's fixture.

### Deviations from detrex's DINO-ViTDet recipe (complete list)
- **E1 — backbone lr factor 1.0 instead of 0.1** (user decision, above).
- **E2 — 16-mixed precision instead of fp32** (user decision, above).
- **E3 — augmentation is horizontal flip only, at a fixed 448×560 input**, instead of flip + multi-scale resize + crop (user decision, above).
- **E4 — no ImageNet-pretrained init of the backbone** (`train.init_checkpoint` MAE ViT, `dino_vitdet_base_4scale_50ep.py:66`): there is no ViT; the frozen DINOv2 encoder is outside the head (F7 D1).
- **E5 — epoch-interval `MultiStepLR`** instead of detectron2's per-iteration `WarmupParamScheduler`; equivalent with zero warmup (above).
- **E6 — validation every 5 epochs with best-AP selection** instead of every 5000 iterations with no selection (`dino_vitdet_base_4scale_12ep.py:19`).

## Step 1 — `Preprocess.geometry`
In `lib/irwm/features/preprocess.py`, add `geometry(width, height)` that returns `ResizeGeometry(src_size=(width, height), crop_box=_crop_box(self.policy, width, height), out_size=(self.size[1], self.size[0]))`, and make `__call__` use it. Covers: FR1.

## Step 2 — Detection dataset, collate, DataModule
Create `lib/irwm/data/detection.py` per FR2, FR3.
- `__init__` precomputes, per selected record, `boxes_xyxy float32 [N, 4]` and `labels int64 [N]` (after `apply_boxes` keep), so `__getitem__` only decodes, preprocesses and flips.
- xywh → xyxy: `x1 = x0 + w`, `y1 = y0 + h`.
- Flip uses `W = image.shape[-1]` (560 under `native`).
- `FlirDetectionDataModule.setup` is idempotent (guard on an attribute), because the entry point calls it once before building the module (to get `val_gt`) and Lightning calls it again.
- `first_nonempty_ids` builds a `FlirDetectionDataset(split, policy=policy, filter_empty=True)` and returns the first `n` ids.
- Module docstring cites this spec and the ignore-region decision.

Covers: FR2, FR3.

## Step 3 — Val mAP
Create `lib/irwm/detection/coco_val.py` per FR4.
- Build the GT with `pycocotools.coco.COCO()`, `coco.dataset = copy.deepcopy(gt)`, `coco.createIndex()`, inside `contextlib.redirect_stdout(io.StringIO())`. Same for `loadRes`, `evaluate`, `accumulate`, `summarize`.
- `loadRes([])` raises in pycocotools, so empty detections return the zero/`nan` dict without calling it.
- Per-class AP: `p = E.eval["precision"]` `[T=10, R=101, K=6, A=4, M=3]`; category index `k` follows `E.params.catIds` (sorted FLIR ids = label order, `classes.py:26`). `AP_<c> = mean(p[:, :, k, 0, 2][p[:, :, k, 0, 2] > -1])`; `AP50_<c>` uses `p[0, :, k, 0, 2]`. No valid entry → `nan`.
- Add `"pycocotools>=2.0.7"` to `lib/pyproject.toml` dependencies with a comment, then `uv lock`.

Covers: FR4.

## Step 4 — `DetectorModule`
Create `lib/irwm/detection/lightning.py` per FR5.
- `encoder` hyperparameters arrive as the merged `cfg.encoder` group (`policy`, `precision`, `hub_version`, `attention`).
- Timing: keep `torch.cuda.synchronize()` calls and `time.perf_counter()` only on `batch_idx % perf_every == 0` steps, in `on_train_batch_start`, after `encode` in `training_step`, and in `on_train_batch_end`; the samples are kept per epoch and their medians logged at `on_train_epoch_end` (amendment 2026-10-27). Skip them (and log nothing) on CPU.
- Non-finite counter: `self._nonfinite_run` reset on a finite loss.
- Validation gather: `torch.distributed.all_gather_object` only when `torch.distributed.is_initialized()`; broadcast with `self.trainer.strategy.broadcast(metrics, src=0)`. `nan` values are logged as they are (W&B accepts them); `val/AP` is never `nan` because person/car exist in every split.
- Checkpoint hooks: `checkpoint["state_dict"] = {k: v for k, v in ... if not k.startswith("encoder.")}`, and the reverse in `on_load_checkpoint`.
- Re-export `load_frozen_detector`, `frozen_meta` from `lib/irwm/detection/__init__.py` (Step 5); do **not** import `lightning.py` from `__init__`, so `import irwm.detection` stays light.

Covers: FR5.

## Step 5 — Frozen export and loader
Create `lib/irwm/detection/frozen.py` per FR6. `torch.load(ckpt_path, map_location="cpu")`; hyperparameters are at `ckpt["hyper_parameters"]` (`save_hyperparameters`). The val metrics file is found through `ckpt["irwm"]["metrics_dir"]` (written by `DetectorModule.on_save_checkpoint`, FR5) as `<metrics_dir>/epoch_<NNN>.json` with `NNN = ckpt["epoch"]`. The checkpoint is saved at the end of the validation of epoch `e`, and Lightning's `ckpt["epoch"]` is then `e`, the same value `current_epoch` had when the metrics JSON was named; a test pins this (Group 8). SHA-256 over the `head.pt` bytes. Covers: FR6.

## Step 6 — Hydra configs
Create `projects/detection/conf/train.yaml`, `export.yaml`, `experiment/det_rgb.yaml`, `experiment/det_ir.yaml` per FR7, FR9. Header comments cite this spec, the detrex source line of every recipe value (as in the decisions above) and the deviations E1–E6. The `encoder.attention` key is new in the merged group: add it to the shared `lib/irwm/conf/encoder/default.yaml` as `attention: plain` with a comment (plain = reference, bit-exact; F8 trains with sdpa), so every config has the key and F4's tools keep their behaviour. This breaks the exact-dict assertion in `tests/test_encoder.py:305-306` (`cfg.encoder == {"policy", "precision", "hub_version"}`). Extend that expected dict with `"attention": "plain"`; that is the only change to F4's tests. Covers: FR7.

## Step 7 — Entry points
Create `projects/detection/train.py` (FR8) and `projects/detection/export.py` (FR9), structured like `projects/smoke/train.py`.
- `train.py` order: `cuda_ops_available()` check → `seed_everything` → `FlirDetectionDataModule(**...)` and `dm.setup("fit")` → `DetectorModule(..., val_gt=dm.val_gt)` → `module.metrics_dir = run_dir / "val_metrics"` → `Trainer(...)` → `fit(model, dm, ckpt_path=resume_ckpt_path(ckpt_dir))`.
- Overfit mode: `ids = first_nonempty_ids(FlirSplit.load(modality, "train"), n)` and `FlirDetectionDataModule(train_image_ids=ids, val_split="train", val_image_ids=ids, flip_p=0)`. The pass callback reads `trainer.callback_metrics["val/AP50_<c>"]`.
- Log `data/n_train_frames`, `data/n_dropped_empty`, `data/n_val_frames` once at start.
- **Reminder (TechStack §6):** no hard-coded paths. The checkpoint dir comes from `cfg.paths.ckpt_root`, the run dir from Hydra, the export dir from `ckpt_root()`. Entry points are the usual place this slips.

Covers: FR8, FR9.

## Step 8 — Tests
Create `tests/test_detector_training.py` covering validation.md Groups 1–9.
- **Tiny FLIR tree helper** in the test file: for `rgb` and `thermal`, `train` and `val` splits, write `coco.json` (FLIR file-name pattern `data/video-<vid>-frame-<6 digits>-<uid>.jpg`, `extra_info.video_id`, categories with FLIR ids 1, 2, 3, 4, 10, 12 and one ignore class, id 8) and JPEGs (thermal `L` 80×64, RGB `RGB` 100×64, so `native` crops RGB to 80×64 from x = 10). Include: one frame with only an ignore box, one RGB frame whose only kept box lies in x < 10, and at least 4 frames with 1–3 kept boxes in every train and val split (Group 8 needs ≥ 2 batches of 2 after empty filtering). Kept boxes cover every class at least once in val, so per-class AP is defined.
- **Stub encoder**: an `nn.Module` whose `encode_pca(x: [B, 3, H, W]) -> [B, H/14, W/14, 1152]` is `F.avg_pool2d(x, 14)` → `permute(0, 2, 3, 1)` → a fixed `Linear(3, 1152)` (seeded with `torch.manual_seed(0)` at construction), under `no_grad`, with `requires_grad_(False)`. It needs no `backbone` attribute, because `encoder_module` bypasses `from_assets` and `set_attention` (FR5).
- **Reduced detector** for CPU/GPU tests: F7's `overrides={"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1, "transformer.decoder.num_layers": 2}` on the 32×40 grid (F7 plan.md Step 7).
- GPU tests skip with `pytest.mark.skipif(not cuda_ops_available(), ...)` in addition to the `gpu` marker.

Covers: FR11.

## Step 9 — Overfit runs on ctcomp3 (user)
1. `git pull && bash slurm/setup_env.sh` on the ctcomp3 login node (picks up `pycocotools` in the lock; no rebuild of detrex needed).
2. `srun --gres=gpu:1 --pty bash -c 'source slurm/env.sh && uv run pytest tests/test_detector_training.py -m "gpu or data" -v'` on an L4 node.
3. For each `E` in `det_rgb det_ir`: `IRWM_STAGE_DATA=1 bash slurm/submit.sh ctcomp3 projects/detection/train.py experiment=$E overfit.enabled=true cluster.timeout_min=120 cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32`.

The user returns: the pytest output, both `overfit.json` files, the SLURM job ids, the W&B run ids, the GPU name (`nvidia-smi -L`) and the `[stage]` lines from each job log. The implementer then checks the overfit bar and the OOM plan (decisions above) before Step 10.

## Step 10 — Full training runs on ctcomp3 (user)
1. For each `E` in `det_rgb det_ir` (both may be submitted at once): `IRWM_STAGE_DATA=1 bash slurm/submit.sh ctcomp3 projects/detection/train.py experiment=$E cluster.timeout_min=1200 cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32`.
   Once the `det_rgb` run is at least one epoch in, the user forces one requeue: `scancel --signal=USR1 --batch <job id>`. They note the `global_step` before and after from W&B, and check that the requeued job staged the data again.
2. When `train_summary.json` exists: `uv run python projects/detection/export.py experiment=$E source_run=slurm-<job id>` (login node is fine; CPU only).

The user returns: both `train_summary.json` files, the `val_metrics/` folders, both export `meta.json` files, the SLURM job ids (including requeues and their wall times), and the W&B run ids.

## Step 11 — Results note and constitution updates
- Write `spec/2026-10-26-detector-training-pipeline/results.md` per FR10 from the returned files.
- Then follow the Roadmap done convention: mark F8 ✓ DONE with a result note; update the Component Registry (rows for `irwm.data.detection`, `irwm.detection.{coco_val,lightning,frozen}`, `projects/detection/{train,export}.py`, the frozen exports `det_rgb`/`det_ir`, the new `encoder.attention` key).
- Amend TechStack §3.2 with the training recipe pointer (this spec, deviations E1–E6) and the frozen-head location (`$IRWM_CKPT_ROOT/detection/frozen/<name>/`, loaded with `load_frozen_detector`, encoded with SDPA); amend §4.4 with the frozen layout; in Roadmap F5 record whether encoding was the bottleneck; and in R11 record that F8 stages by default (user decision, 2026-10-26), with the measured staging seconds.

Covers: FR10.

## Implementation Order
1. Step 1 — `Preprocess.geometry`
2. Step 2 — Detection dataset, collate, DataModule
3. Step 3 — Val mAP
4. Step 4 — `DetectorModule`
5. Step 5 — Frozen export and loader
6. Step 6 — Hydra configs
7. Step 7 — Entry points (check locally that `train.py experiment=det_ir` exits non-zero with the CUDA-ops message, and that `train.py` without `experiment` fails at composition)
8. Step 8 — Tests → `uv run pytest tests/test_detector_training.py -v`, then the full suite `uv run pytest`
9. Step 9 — Overfit runs on ctcomp3 (user)
10. Step 10 — Full training runs on ctcomp3 (user)
11. Step 11 — Results note and constitution updates

## Amendments
- **2026-10-26 — Overfit mode: unbounded epochs, constant lr** (user decision). At 256 frames and batch 16 an epoch is 16 steps, so `trainer.max_epochs: 50` would end the overfit at step 800 and the per-epoch milestone 40 would cut lr ×0.1 at step 640. Overfit mode now sets `max_epochs=-1` (so `overfit.max_steps` governs) and `optim.lr_drop_epoch=None`, which `DetectorModule.configure_optimizers` reads as "no scheduler". The 50-epoch runs are unchanged. FR5 (scheduler), FR8 (overfit mode).
- **2026-10-26 — No metrics from the sanity-check validation** (user decision). Lightning's 2-batch sanity check would otherwise write `val_metrics/epoch_000.json` and log `val/*` from 2 batches, and the overfit pass callback would see it. The sanity batches still run (an error in the val path shows within seconds), but `on_validation_epoch_end` discards them, and the overfit callback ignores sanity checks. FR5 (validation), FR8 (overfit callback).
- **2026-10-26 — Citation line numbers** (no behaviour change). The 50ep schedule is `detrex/config/configs/common/coco_schedule.py:80` (`lr_multiplier_50ep = default_coco_scheduler(50, 40, 0)`), imported at `dino_vitdet_base_4scale_50ep.py:10`; the MAE `init_checkpoint` is `dino_vitdet_base_4scale_50ep.py:15`.
- **2026-10-27 — Timing and GradScaler scale logged once per epoch** (user decision). Step 9 found `train/grad_scale` absent from W&B and `perf/*` almost absent: Lightning writes `on_step` values only on steps where `(step + 1) % log_every_n_steps == 0` (`logger_connector.py:51-63`), and both were logged on other steps (`global_step % 50 == 0`, `batch_idx % 50 == 0`). Every 50th batch is still timed, but the samples are kept per epoch and their medians logged at `on_train_epoch_end` (`perf/encode_ms`, `perf/step_ms`, `perf/encode_share`), together with `train/grad_scale`. FR10's medians are taken over these per-epoch medians. `overfit.json`'s `median_step_ms` was never affected (it reads the module's own list). FR5, plan Step 4, validation Group 8 item 4 (re-run pending).
- **2026-10-27 — Overfit memory on V100S accepted; no GPU pinning** (user decision). The `det_ir` overfit (job 5526085) ran on hpc-gpu1, a V100S 32 GB, not an L4: peak 18.23 GiB, against 18.13 GiB for `det_rgb` on an L4 (job 5526084), same tensor shapes. Accepted as the memory evidence without a re-run, and the overfits at `6425d4c` stand as evidence for the logging-only change above. The L4 is the planning lower bound for speed and memory, not a target: Step 10 runs on whatever GPU the scheduler gives, and the results note reports the GPU per run.
