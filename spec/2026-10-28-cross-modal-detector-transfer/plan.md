# F15 · Cross-modal detector transfer: Plan

## Context and Design Decisions

### Protocol
- **Test 2×2 only, no `val` cells (user decision, 2026-10-28).** The roadmap fixes `video_*_test` as the split. A `val` 2×2, whose diagonal would have reproduced F8's val AP, was offered and declined. The inference path is instead proven equal to F8's validation path by a CPU test (Group 3).
- **Every frame of each test split is scored (3,749 per modality), including frames with no kept box.** Mission "Comparison": each detector is scored on its modality's full test split with that modality's own annotations. Frames are not restricted to F3 clip frames. Saved detections let F9/F12 take subsets later. `filter_empty=False` matches F8's val loader (F8 plan.md "Empty train frames are dropped, val frames are all kept").
- **Inputs follow the test modality, the head is the only thing that changes (Roadmap F15).** `Preprocess(policy, split.modality)` replicates thermal 1→3 and centre-crops RGB to 5:4 (`preprocess.py:79-119`). So `det_rgb` sees thermal frames exactly as `det_ir` saw them in training, and the reverse.
- **One encoder pass per test split, shared by both heads (Roadmap F15).** Both `meta.json` files record the same `encoder` block (`policy: native`, `precision: fp16`, `hub_version: dinof_release`, `attention: sdpa`, `encoder_tag: dinov2_vitb14reg_448x560_l3-6-9-12_pca1152`) and `image_size: [448, 560]` (read from `checkpoints/detection/frozen/det_rgb/meta.json` on 2026-10-28). FR1 turns that observation into a checked precondition, so a future re-export with other settings fails loudly instead of feeding a head the wrong features.
- **Precision = F8's validation precision.** F8 validated under Lightning `16-mixed`, i.e. the detector forward inside CUDA fp16 autocast, with the encoder applying its own fp16 autocast (`encoder.py:140-142`). `predict_split` wraps only the detector forward in `torch.autocast("cuda", dtype=torch.float16)`. The encoder already autocasts itself, so wrapping it again changes nothing. `head.pt` holds fp32 master weights, and `frozen.py:10-11` requires fp16 autocast at use.
- **Batch 16, ascending `image_id`, no shuffle.** This is F8's val batch size (F8 FR7 `data.batch_size: 16`) and order (`FlirDetectionDataModule.val_dataloader`). The eval forward of each image does not depend on the rest of the batch (all frames are 448×560, so there is no padding; `detector.py:115-118` loops per image), so batch size changes no prediction.
- **Compiled MSDA required.** `transfer.py` exits if `cuda_ops_available()` is false, as `train.py` does. The heads were trained and validated with detrex's CUDA op (F7, 1.9–2.8× faster inference). The library function itself runs on CPU through detrex's PyTorch fallback (`detrex/layers/multi_scale_deform_attn.py:341`), which is what the CPU tests use.

### Metrics
- **Scoring = F8's `coco_map` on `FlirSplit.to_coco_gt("normalised")` of the full test split (Roadmap F15).** Ignore regions are crowd GT, areas are normalised to 640×512 (TechStack §3.4), and predictions are inverted to FLIR px with `to_flir_detections` (`coco_val.py:28-45`).
- **AP_person_car = mean(AP_person, AP_car), same for AP50.** COCO AP is the mean of per-category AP over `catIds`, so this equals `COCOeval` restricted to `catIds = [1, 3]` (area `all`, `maxDets` 100), which is what `coco_map`'s per-class values already read (`coco_val.py:83-87`). Computing it from the existing per-class values needs no second `COCOeval` run. Roadmap R10 is why this subset exists: on the test split, thermal labels far fewer sign/light/bike than RGB.
- **Transfer ratio = cross-modal ÷ in-domain on the same test modality (my choice, a reporting convenience).** For example, `transfer_ratio["det_rgb"]["thermal"]["AP"] = AP(det_rgb on thermal) / AP(det_ir on thermal)`. The denominator shares the GT and the frames, so the ratio isolates the head. It is not a ratio across GT sets.
- **Diagonal vs F8 val: reported, not gated (user decision, 2026-10-28).** `diag_vs_val` records test AP / val AP per head. The roadmap's "a big gap gets investigated before the off-diagonal cells are read" is replaced by: results.md states the ratio, and adds one sentence if it lies outside [0.5, 1.5]. Val (sparse frames) and test (8 video sequences) differ in content, so a moderate gap is expected.

### Detections file
- **All ≤ 300 boxes per frame are kept (user decision, 2026-10-28).** pycocotools applies `maxDets` per (image, category), not per image (`pycocotools/cocoeval.py:243,260`). A per-image top-100 would therefore change AP on re-scoring. Keeping everything makes re-scoring identical by construction, and keeps the low-score tail for F13's calibration analysis.
- **Gzipped JSON at a stable, write-once path (user decision).** The path is `$IRWM_OUTPUT_ROOT/detection/detections/<encoder_tag>/<head>/<modality>_<split>.json.gz`. The `detections` field is the COCO results list `coco_map` consumes. An existing file stops the run unless `overwrite=true` (user decision; rejected: silent reuse, silent overwrite). The check runs **before** encoding (FR6 step 2), so a mistaken rerun fails within seconds, not after inference.
- **Per-frame metadata in the same file (user decision: "enough metadata to recover the tags per prediction").** `images[]` carries `video_id` and `frame_index` (`FlirRecord`, `flir.py:95-101`), and `video_tags` is `split.video_tags()` (`flir.py:231-237`, test split only). A prediction joins to its tags by `image_id → video_id → video_tags`. A single file avoids a sidecar drifting from its detections.
- **Deterministic bytes.** `gzip` writes the mtime into its header, so `mtime=0` plus `sort_keys=True` makes the file's SHA-256 depend only on its content (except `created_at`, `git` and `device`, which differ between runs by design). Group 4 checks identical bytes for identical input objects.

### Run
- **No data staging.** The job reads 2 `coco.json` files and 7,498 JPEGs once. R11 measured that staging pays only for repeated image reads or the 6-file metadata load (copy 57–317 s vs ≈ 170 s saved). The user may still set `IRWM_STAGE_DATA=1`; nothing depends on it.
- **Any GPU; the GPU is recorded** (user memory: the L4 is a lower bound for planning, never a target). Inference at batch 16 with no grad fits well inside an L4 (F7: head inference at B = 8 is far below training memory; F8 val ran at B = 16 inside the training job's 20.9 GiB).
- **Cost estimate.** Encoder ≈ 89 frames/s → 7,498 / 89 ≈ 1.4 min. Heads: F7 full-detector inference ≈ 31 ms per frame on an L4, minus the encoder part, × 4 × 3,749 ≈ 6 min at most. Plus startup (R11: `import torch` 40 s, `coco.json` parse up to 3 min on BeeGFS). `cluster.timeout_min=60` covers this with margin.

### Code facts the implementer relies on
- **`FlirDetectionDataset` works for the `test` split unchanged.** It only needs a `FlirSplit` (`detection.py:43-66`). `FlirSplit.load(modality, "test")` maps to `video_{modality}_test` (`flir.py:34,41-47`). Its targets are computed but unused here.
- **`DetectorModule.validation_step` (`lightning.py:162-165`) is the reference behaviour** that `predict_split` must equal: `detector(encode(images), image_size=self.image_size)`, then `to_flir_detections([pred], [image_id], [geom])` per image. `encode` = `encode_pca(...).permute(0, 3, 1, 2).contiguous()` (`lightning.py:78-80`).
- **`DINODetector` eval returns per-image `{"boxes" xyxy, "scores", "labels"}`, ≤ `select_box_nums_for_evaluation` = 300** (`detrex/projects/dino/modeling/dino.py:72,526-533`; `detector.py:120-123`).
- **`load_frozen_detector` returns the head in `eval()` with no grad, SHA-checked** (`frozen.py:126-136`). It takes `device`.
- **`make_wandb_logger(cfg, save_dir)`** needs `cfg.wandb.*` and `cfg.run_name` (`tracking.py:50-70`), both provided by `irwm_base`. `logger.experiment` is the W&B run for the table. With `WANDB_MODE=disabled` (tests, `conftest.py`) logging is a no-op.
- **Test fixtures are local to the test file.** F8 keeps its tiny tree in `tests/test_detector_training.py` rather than `conftest.py` (F8 plan.md "Detection tests need a fake FLIR tree"). F15 writes its own tiny `video_*_test` tree and stub encoder in `tests/test_detector_transfer.py`, and does not import from another test module.

## Step 1 — Inference library
Create `lib/irwm/detection/inference.py` with a module docstring citing this spec. Implement `SplitPrediction`, `shared_encoder_meta`, `check_encoder_config`, `predict_split` and `with_person_car`.
- `predict_split` checks `detectors` is non-empty and that every detector has `training is False`, before building the loader. It builds `FlirDetectionDataset(split, policy=policy, flip_p=0.0, filter_empty=False)` and a `DataLoader` per FR2.
- Per batch: move `images` to `device`, time `encode` (sync on CUDA), then for each head (in `detectors` insertion order) time `det(feats, image_size=image_size)` under the autocast context of FR2, and extend that head's list with `to_flir_detections(preds, batch["image_ids"], batch["geometries"])`.
- Wrap the whole loop in `torch.no_grad()`.
- Return `image_ids` in loader order (ascending).

Covers: FR1, FR2, FR3.

## Step 2 — Detections file helpers
In the same module, add `detections_path`, `save_detections`, `load_detections` and `check_outputs_free` per FR4.
- `save_detections` checks every detection's `image_id` against `{r.image_id for r in split.records}` before writing.
- Write path: `tmp = path.with_name(path.name + ".tmp")`, `gzip.GzipFile(tmp, "wb", mtime=0)` wrapping the UTF-8 `json.dumps(obj, sort_keys=True)`, then `tmp.replace(path)`. SHA-256 is computed over the final file bytes.
- **Reminder (TechStack §6):** the root comes from `irwm.utils.paths.output_root()`, never a literal path. Entry points and file helpers are where this usually slips.

Covers: FR4.

## Step 3 — Config and entry point
Create `projects/detection/conf/transfer.yaml` (FR5) with a header comment giving the submit command of Step 5. Create `projects/detection/transfer.py` (FR6), following `train.py`'s structure: `load_env()` at import, the `cuda_ops_available()` exit first, then heavy imports inside `main`.
- The `cfg.encoder` vs shared-meta check is `check_encoder_config` (FR1), called right after `shared_encoder_meta`.
- `run_dir = Path(HydraConfig.get().runtime.output_dir)`. W&B `save_dir=run_dir`.
- Metrics logging: `logger.log_metrics({...}, step=0)` for the scalar keys, then `logger.experiment.log({"transfer/matrix": wandb.Table(...)})`.
- `transfer.json` is written by the process (single GPU, no DDP; TechStack §5's "no rank 0 assumption" concerns Lightning training, and this script is not a Lightning fit). Use `json.dumps(..., indent=2)`. `nan` values are written as JSON `NaN`, as F8's `val_metrics` do.
- Update `projects/detection/pyproject.toml`'s `description` to mention F15.

Covers: FR5, FR6.

## Step 4 — Tests
Create `tests/test_detector_transfer.py`.
- **Tiny test tree helper:** for `rgb` (100×64, RGB) and `thermal` (80×64, L), write `FLIR_ADAS_v2/video_{rgb,thermal}_test/coco.json`, `index.json` (`{"videos": [{"id": "vidA", "tags": ["night", "rain"]}, {"id": "vidB", "tags": ["day"]}]}`) and noise JPEGs under `data/video-<vid>-frame-<6 digits>-u<id>.jpg`. Use 6 frames (3 per video) with 1–3 kept boxes covering all 6 classes, one frame with only an ignore box (category 8), and one frame with no annotation. Mirror F8's helper (`tests/test_detector_training.py:44-110`) for categories and sizes.
- **Stub encoder:** same definition as F8's `StubEncoder` (`test_detector_training.py:116-128`), plus a call counter on `encode_pca`.
- **Reduced detector:** `DINODetector(in_channels=1152, stem_dim=768, num_classes=6, overrides=REDUCED)` with F7's `REDUCED` (`test_detector_training.py:38-40`), seeded with `torch.manual_seed(k)` before construction, then `.eval()`.
- **Frozen-head fixtures:** a local helper `write_fake_frozen(name, det, *, encoder, image_size=[448, 560])` writes `irwm_env["ckpt"]/detection/frozen/<name>/head.pt` (`torch.save(det.state_dict())` of a reduced detector) and `meta.json` directly. `meta.json` holds every key of F8 FR6: `detector` = the reduced kwargs (so `load_frozen_detector` rebuilds it), `encoder` with `encoder_tag`, `head_sha256` of the written `head.pt`, `val_metrics: {"metrics": {"AP": 0.3, "AP50": 0.6}}`, `precision: "16-mixed"`. It does not go through `export_frozen`, which needs a Lightning checkpoint.

Covers: FR8 (all groups in validation.md).

## Step 5 — Run on ctcomp3 (user)
1. On the login node: `git pull` (clean tree at the commit to run), then `uv sync`.
2. Submit:
   ```
   bash slurm/submit.sh ctcomp3 projects/detection/transfer.py \
       cluster.timeout_min=60 cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32
   ```
3. When the job ends, copy back to the laptop or ctdesk236 (git-ignored `outputs/`, as for F8):
   - the Hydra run dir `outputs/detection/<date>/<time>/` (holds `transfer.json` and the offline W&B files);
   - `outputs/detection/detections/` (the 4 `.json.gz` files);
   - the job log `outputs/slurm_logs/irwm-detection-<job>.out`.

The user returns `transfer.json`, the 4 detection files, the job log, and the W&B run id (or the offline run dir if `WANDB_MODE=offline`).

Covers: FR7.

## Step 6 — Re-score check and results note
1. On CPU, for each of the 4 files: `load_detections(path)`, then `with_person_car(coco_map(FlirSplit.load(modality, "test").to_coco_gt("normalised"), obj["detections"]))`. Compare with `transfer.json` `cells` (Data Validity item "Re-score").
2. Write `spec/2026-10-28-cross-modal-detector-transfer/results.md` per FR7.
3. Constitution updates. The spec-time edits were already made on 2026-10-28: the F15 decision note and the reworded "Done when" in the Roadmap, the detections path in TechStack §4.4, and the three ☐ rows in TechStack §8. The implementer now does three things:
   - Roadmap F15 → ✓ DONE, with a result summary (2×2 AP/AP50, GPU, commit);
   - TechStack §8: flip the three F15 rows to ✓;
   - TechStack §4.4: drop "planned" from the detections bullet.

Covers: FR7.

## Implementation Order
1. Inference library (Step 1)
2. Detections file helpers (Step 2)
3. Config and entry point (Step 3)
4. Tests (Step 4) → `uv run pytest tests/test_detector_transfer.py` (CPU), then the full default suite `uv run pytest`; on ctcomp3 inside a GPU allocation: `uv run pytest tests/test_detector_transfer.py -m gpu` and `-m "data and gpu"`
5. Run on ctcomp3 (Step 5, user)
6. Re-score check and results note (Step 6)
