# F15 · Cross-modal detector transfer: Validation

## Code Correctness (`tests/test_detector_transfer.py`, CPU, no data, no weights, < 60 s)

### Group 1 — Shared encoder settings (FR1)
- [x] Two fake frozen heads with identical `meta.json` `encoder`, `image_size` and `precision` → `shared_encoder_meta(["det_rgb", "det_ir"])` returns `{"encoder": <that dict>, "image_size": (448, 560), "precision": "16-mixed"}`.
- [x] Same, but `det_ir`'s `encoder.attention = "plain"` → `ValueError` whose message contains `attention`, `sdpa` and `plain`.
- [x] Differing `image_size` → `ValueError` containing `image_size`.
- [x] `shared_encoder_meta([])` → `ValueError`.

### Group 2 — Multi-head inference (FR2)
- [x] Tiny thermal test split (6 frames), two reduced detectors (seeds 0 and 1), stub encoder, `batch_size=4`, CPU → `SplitPrediction.detections` has keys `["det_a", "det_b"]` in insertion order. `image_ids` equals `[r.image_id for r in split.records]` (all 6, ascending, including the ignore-only and the empty frame).
- [x] Same run → the stub's `encode_pca` was called exactly `ceil(6 / 4) = 2` times (once per batch, not once per head).
- [x] Every detection has keys `{"image_id", "category_id", "bbox", "score"}`, `category_id ∈ {1, 2, 3, 4, 10, 12}`, and each image has ≤ 300 detections.
- [x] The two heads' detection lists differ (different weights) → at least one `score` differs.
- [x] RGB tiny split (100×64 frames, `native` crop x = 10..90) → every `bbox` satisfies `x ≥ 10 − 1e-3` and `x + w ≤ 90 + 1e-3`. DINO clips boxes to `image_size`, so this shows `invert_boxes` added the crop offset.
- [x] A detector put in `.train()` → `ValueError` before the stub encoder is called (call count 0).
- [x] `predict_split({}, ...)` → `ValueError`.
- [x] `encode_s ≥ 0` and `head_s` has one non-negative float per head.

### Group 3 — Equal to F8's validation path (FR2)
- [x] Tiny thermal **and** RGB test split, one reduced detector, stub encoder, CPU fp32. Path A: `predict_split({"h": det}, split, stub, policy="native", image_size=(448, 560), batch_size=4)`. Path B: a `DetectorModule` (F8) built with the same detector weights (`module.detector.load_state_dict(det.state_dict())`) and `encoder_module=stub`, put in eval, with `validation_step` run over a `DataLoader(FlirDetectionDataset(split, flip_p=0.0, filter_empty=False), batch_size=4, collate_fn=collate_detection)` and `module._val_dets` read afterwards. Path A's list equals path B's concatenation in ascending `image_id` **exactly** (same dicts, `|Δ| = 0` on every `bbox` value and `score`).

### Group 4 — Detections file (FR4)
- [x] `detections_path("det_rgb", "thermal")` with `IRWM_OUTPUT_ROOT=<tmp>` → `<tmp>/detection/detections/dinov2_vitb14reg_448x560_l3-6-9-12_pca1152/det_rgb/thermal_test.json.gz`.
- [x] `save_detections` then `load_detections` → `detections` equal to the input list. `images` has 6 entries with `video_id ∈ {"vidA", "vidB"}` and the `frame_index` parsed from the file name. `video_tags == {"vidA": ["night", "rain"], "vidB": ["day"]}`. `n_images == 6`, `n_detections == len(input)`. `head_sha256`, `encoder` and `image_size` equal the fake `meta.json`.
- [x] Every `images[*].video_id` is a key of `video_tags`, so each detection's tags are recoverable by `image_id → video_id → video_tags`.
- [x] Saving the same object twice to two paths with `created_at`, `git` and `device` monkeypatched to constants → the two files' SHA-256 are equal, and equal to the returned value.
- [x] Saving to an existing path with `overwrite=False` → `FileExistsError`, and the file's bytes are unchanged. With `overwrite=True` → replaced, no `*.tmp` left in the directory.
- [x] A detection with an `image_id` not in the split → `ValueError`, and no file is created.
- [x] `check_outputs_free([p1, p2], overwrite=False)` with `p2` existing → `FileExistsError` whose message contains `p2`. With `overwrite=True` → no error. With neither existing → no error.

### Group 5 — Person + car metric (FR3)
- [x] `{"AP_person": 0.4, "AP_car": 0.6, "AP50_person": 0.8, "AP50_car": 0.6, ...}` → `AP_person_car == 0.5`, `AP50_person_car == 0.7`, and all input keys are unchanged.
- [x] `AP_car = nan` → `AP_person_car` is `nan`.
- [x] On the tiny thermal split with perfect detections (GT boxes as detections, score 1.0), `with_person_car(coco_map(gt, dets))["AP_person_car"]` equals `COCOeval` run with `params.catIds = [1, 3]` to `1e-12` (AP_person_car is the restricted COCO AP).

### Group 6 — Config and entry point (FR5, FR6)
- [x] Compose `transfer.yaml` (Hydra compose API, `pkg://irwm.conf` search path) → `heads == ["det_rgb", "det_ir"]`, `test_modalities == ["rgb", "thermal"]`, `split == "test"`, `batch_size == 16`, `encoder.attention == "sdpa"`, `encoder.policy == "native"`, `overwrite is False`, `"transfer" in wandb.tags`.
- [x] `check_encoder_config({"policy": "native", "precision": "fp16", "hub_version": "dinof_release", "attention": "plain"}, shared)` with the fake shared meta (`sdpa`) → `ValueError` naming `attention`. With `attention: "sdpa"` → no error. The composed `transfer.yaml` `encoder` node passes against the fake meta.
- [x] Without CUDA ops (test skipped if `cuda_ops_available()`): `python projects/detection/transfer.py` exits non-zero and stderr contains `cuda_ops_available`.

### Group 7 — GPU (`@pytest.mark.gpu`, run on ctcomp3 inside a GPU allocation)
- [ ] Tiny thermal test split, reduced detector moved to CUDA, stub encoder on CUDA, `device="cuda"` → finite `score`s in `[0, 1]`, the same `image_ids` as the CPU run, and ≤ 300 detections per image. Detector parameters stay fp32 after the call (autocast does not cast weights).

### Group 8 — Real heads (`@pytest.mark.data` + `gpu`, ctcomp3; skipped if `frozen/det_rgb/meta.json` is missing)
- [ ] `shared_encoder_meta(["det_rgb", "det_ir"])` returns `encoder.attention == "sdpa"`, `encoder.policy == "native"`, `image_size == (448, 560)`.
- [ ] The first 4 thermal test frames (ascending `image_id`, a `FlirSplit` restricted to those records), both real heads, the real encoder (`from_assets` + `set_attention(..., "sdpa")`) on CUDA → 2 lists, each with 4 distinct `image_id`s, 1–300 detections per image, all `bbox` inside the 640×512 frame within 1 px.

## Data Validity (`transfer.json`, the 4 detection files, the job log; results.md)
- [ ] **Four cells complete:** `cells` has `det_rgb` and `det_ir`, each with `rgb` and `thermal`. Each cell has the 6 base keys (`AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`), the 12 per-class keys and the 2 person + car keys (20 keys).
- [ ] **All frames scored:** `n_images == {"rgb": 3749, "thermal": 3749}` (F2), and each detection file has `n_images == 3749`.
- [ ] **Detection files complete:** 4 files exist at `detections_path(head, modality)`. Each has `1 ≤ n_detections ≤ 3749 × 300 = 1,124,700`, and every `image_id` belongs to the matching test split. The `sha256` in `transfer.json` equals `sha256sum` of the file.
- [ ] **Re-score:** for each of the 4 files, `with_person_car(coco_map(to_coco_gt, load_detections(path)["detections"]))` on CPU equals `transfer.json` `cells[head][modality]` for every key, with `|Δ| = 0` (NaN == NaN). If this fails, the saved files cannot stand in for the run, and F9 cannot reuse them.
- [ ] **Encoded once per split:** `transfer.json` `timing` has exactly one `encode_s` per modality (2 values) and four `head_s` values. Total encoded frames = 7,498.
- [ ] **Diagonal reported:** `diag_vs_val` has `det_rgb` (val AP 0.3021, AP50 0.5726) and `det_ir` (val AP 0.3297, AP50 0.6264), each with `test_AP`, `ratio`, `test_AP50`, `ratio50`. The ratios are copied into results.md. There is no gate (user decision, 2026-10-28); a ratio outside [0.5, 1.5] gets one sentence in results.md.
- [ ] **Transfer ratios:** `transfer_ratio` has exactly `det_rgb.thermal` and `det_ir.rgb`, each with 4 keys, and `transfer_ratio["det_rgb"]["thermal"]["AP"] == cells["det_rgb"]["thermal"]["AP"] / cells["det_ir"]["thermal"]["AP"]` to `1e-12`.
- [ ] **Cost and GPU recorded:** `device.name` is non-empty, `device.peak_mem_gib > 0`, `device.slurm_job_id` is set, and `timing.wall_s` is in results.md together with the GPU name.
- [ ] **W&B:** the run has `test/det_rgb/thermal/AP` (and the other 79 cell keys), the 8 `transfer/*` keys and the `transfer/matrix` table with 4 rows. The run id is in results.md.
- [ ] **Results note:** results.md has the 2×2 tables for AP, AP50, AP_person_car and AP50_person_car, per-class AP/AP50 and APs/APm/APl per cell, the transfer ratios, `diag_vs_val`, cost, provenance (commit, job, W&B id, file SHA-256s) and the three caveats of FR7 (R10, preliminary until F6, RGB `native` edge crop).

## Data Architecture Integrity
- [ ] **Provenance:** `transfer.json` `git.dirty == false`, and its commit is the one in results.md. Each `heads[*].head_sha256` equals the frozen `meta.json` value (`96db5065…` for `det_rgb`, `2ff38732…` for `det_ir`), and equals `head_sha256` inside the matching detection files.
- [ ] **Frozen heads untouched:** `sha256sum $IRWM_CKPT_ROOT/detection/frozen/*/head.pt` after the run equals the values above, and the `meta.json` mtimes are unchanged.
- [ ] **Writes only under the allowed roots:** the run creates files only under `$IRWM_OUTPUT_ROOT` (run dir and `detection/detections/`). Nothing is written under `$IRWM_DATA_ROOT` or `$IRWM_CKPT_ROOT`, and the git tree is clean after the run.
- [x] **Write-once enforced (reviewed by eye + Group 4):** in `transfer.py`, `check_outputs_free` on the 4 detection paths runs before `load_frozen_detector` and before the encoder is built (FR6 step 2). Group 4's `check_outputs_free` and `save_detections` tests pass. No second cluster job is submitted for this.
- [x] **No change to third-party code or F8 modules:** `git diff --stat` of the feature touches no file under `third_party/`, `lib/irwm/detection/{coco_val,frozen,lightning,detector}.py` or `lib/irwm/data/`.
