# F8 · Detector training pipeline: Results (2026-10-28)

Results note per FR10, written from the user's ctcomp3 runs of 2026-10-27 (plan.md Steps 9–10) and the exports of 2026-10-28. Raw outputs, in the repo's git-ignored `outputs/` and `checkpoints/`:
- Run dirs: `outputs/detection/2026-10-27/{12-27-09,12-28-00}/` hold `train_summary.json`, `val_metrics/` and the offline W&B files.
- Overfits: `outputs/detection/2026-10-27/{11-12-19,11-12-33}/overfit.json`.
- Job logs: `outputs/slurm_logs/irwm-detection-<job>.out`.
- Checkpoints: `checkpoints/detection/slurm-<job>/`.
- Frozen heads: `checkpoints/detection/frozen/{det_rgb,det_ir}/`.

Per-epoch series (`train/loss_epoch`, `train/lr`, `train/nonfinite_steps`, `train/grad_scale`, `perf/*`) and GPU memory were read from the runs' `.wandb` logs. GPU memory is the system metric `gpu.0.memoryAllocatedBytes`.

**Provenance.**
- **Full runs:** commit `b81ef6c`, clean (`git.dirty = false` in every `val_metrics`, `train_summary.json` and export `meta.json`). `b81ef6c` contains the 2026-10-27 per-epoch logging amendment. `HEAD` (`4246f45`) differs from it only in `.gitignore`.
- **Overfits:** commit `6425d4c`, clean. They were accepted as evidence for the logging-only change (plan.md Amendments, 2026-10-27).
- **Exports:** run on 2026-10-28 from `b81ef6c`.
- **Software:** torch `2.2.0+cu121`, detrex and detectron2 as in F7.

**Epoch numbering.** Lightning epochs are 0-based, so `best-e014.ckpt` is the checkpoint after the 15th epoch. The 10 validations run after epochs 5, 10, …, 50, which is `epoch` 4, 9, …, 49 in the JSONs.

## 1. Runs

| | `det_rgb` | `det_ir` |
|---|---|---|
| Modality | RGB | thermal |
| SLURM job (requeues) | 5526115 (none, `restart 0`) | 5526116 (none, `restart 0`) |
| Node, GPU | hpc-gpu5, NVIDIA L4 24 GB | hpc-gpu6, NVIDIA L4 24 GB |
| W&B run | `6xlqcqo3` (`slurm-5526115`) | `cdkr1ggq` (`slurm-5526116`) |
| Staging (`[stage]` line) | 46,450 files, 12,159 MB in **218 s** | 46,450 files, 12,159 MB in **57 s** |
| `fit_wall_s` | 48,362 s (13.43 h) | 50,167 s (13.94 h) |
| Job wall, approx.¹ | 48,648 s (13.51 h) | 50,288 s (13.97 h) |
| GPU-hours (1 GPU) | **13.5** | **14.0** |
| Epochs completed | 50 (`max_epochs=50` reached) | 50 (`max_epochs=50` reached) |
| Global steps | 31,350 (627 / epoch) | 32,650 (653 / epoch) |
| `n_train_frames` + `n_dropped_empty` | 10,038 + 280 = 10,318 | 10,462 + 280 = 10,742 |
| `n_val_frames` | 1,085 | 1,144 |
| Median `perf/step_ms` | 1,527 ms | 1,520 ms |
| Median `perf/encode_ms` | 183 ms | 185 ms |
| Median `perf/encode_share` | **0.120** | **0.122** |
| Median `perf/train_images_per_s` | 10.48 | 10.52 |
| Median `perf/epoch_s` | 957 s | 994 s |
| Peak GPU memory (W&B `gpu.0.memoryAllocatedBytes`)² | 22,403,874,816 B (20.87 GiB, 92.8 %) | 22,378,708,992 B (20.84 GiB, 92.7 %) |
| `train/nonfinite_steps` (max over run) | 0 | 0 |
| `train/grad_scale` (min–max of per-epoch values) | 512–2048 | 512–4096 |

¹ This is the time from the job-log start line to the W&B run start, plus W&B `_runtime`. `sacct` was not queried.

² This is device memory in use as NVML reports it, so it includes the PyTorch caching allocator's reserve. It is not `torch.cuda.max_memory_allocated`, which `train.py` records only for overfit runs (plan.md Amendments, 2026-10-28). The allocator peak at batch 16 from the overfits is 18.13 GiB for RGB (L4) and 18.23 GiB for thermal (V100S), with the same tensor shapes.

Total for the two 50-epoch detectors: **≈ 27.5 L4 GPU-hours**, both run concurrently, about 14 h of wall time.

## 2. Best checkpoints and exports

Selection is by highest `val/AP`, the COCO AP over IoU 0.50:0.95 on the full val split (E6). The figures below are the export's `meta.json.val_metrics`. `meta.json.val_metrics.AP` equals `train_summary.json.best_val_AP` within 1e-8.

| | `det_rgb` | `det_ir` |
|---|---|---|
| Best epoch (0-based) / global step | 14 / 9,405 | 34 / 22,855 |
| Source checkpoint | `detection/slurm-5526115/best-e014.ckpt` | `detection/slurm-5526116/best-e034.ckpt` |
| **AP** | **0.3021** | **0.3297** |
| AP50 | 0.5726 | 0.6264 |
| AP75 | 0.2790 | 0.3120 |
| APs / APm / APl | 0.2764 / 0.4479 / 0.4679 | 0.2519 / 0.5165 / 0.6478 |
| `head_sha256` | `96db506529ba407402e6a0dfd7fdc7b6c17aae66aef3f577c75288ebf7c7fdfa` | `2ff38732df45bd92acbb33dc4c645416a720211bd7ee76a24f83202d742d0fcb` |

Per class (AP / AP50):

| class | `det_rgb` | `det_ir` |
|---|---|---|
| person | 0.3184 / **0.6408** | 0.3764 / **0.7311** |
| bike | 0.2475 / 0.4611 | 0.2762 / 0.5460 |
| car | 0.4957 / **0.7639** | 0.5265 / **0.8129** |
| motor | 0.3232 / 0.5981 | 0.3240 / 0.6295 |
| light | 0.1674 / 0.4793 | 0.2184 / 0.5565 |
| sign | 0.2606 / 0.4924 | 0.2565 / 0.4823 |

**Done bar (AP50 ≥ 0.30 for person and car, in each detector): met.** The lowest of the four values is 0.641, for RGB person.

Both exports record:
- `precision: 16-mixed`, `encoder.attention: sdpa`, `encoder.precision: fp16`;
- `encoder_tag: dinov2_vitb14reg_448x560_l3-6-9-12_pca1152`;
- `image_size: [448, 560]`, `git.commit` and `head_sha256`.

On ctdesk236 (CPU), `load_frozen_detector` returns a `DINODetector` for both names, with `training=False`, no parameter requiring grad, and 23,001,286 parameters. This used `IRWM_CKPT_ROOT` pointing at the copied `checkpoints/`. Neither `best-e*.ckpt` contains an `encoder.*` key: each has 394 keys, all under `detector.`.

## 3. Validation curves

`val/AP` at each of the 10 validations (`val_metrics/epoch_<NNN>.json`), with AP50 for person and car in brackets:

| epoch (0-based) | `det_rgb` AP (AP50 person, car) | `det_ir` AP (AP50 person, car) |
|---|---|---|
| 4 | 0.2473 (0.556, 0.701) | 0.2757 (0.649, 0.766) |
| 9 | 0.2909 (0.633, 0.755) | 0.3090 (0.725, 0.817) |
| 14 | **0.3021** (0.641, 0.764) | 0.3260 (0.736, 0.824) |
| 19 | 0.2978 (0.639, 0.765) | 0.3263 (0.743, 0.823) |
| 24 | 0.2949 (0.639, 0.756) | 0.3258 (0.742, 0.822) |
| 29 | 0.2877 (0.610, 0.751) | 0.3215 (0.736, 0.819) |
| 34 | 0.2809 (0.611, 0.749) | **0.3297** (0.731, 0.813) |
| 39 | 0.2893 (0.618, 0.749) | 0.3240 (0.729, 0.806) |
| 44 | 0.2826 (0.601, 0.745) | 0.3290 (0.724, 0.806) |
| 49 | 0.2747 (0.585, 0.742) | 0.3162 (0.712, 0.797) |

- **RGB peaks early.** `val/AP` is highest at epoch 14 and drifts down to 0.275 by epoch 49, a loss of 0.027. The lr drop at epoch 40 gives a small rise at epoch 39, then the decline continues. Training loss keeps falling throughout (§4), which points to overfitting the 10k RGB train frames.
- **Thermal plateaus.** It sits between 0.321 and 0.330 from epoch 14 to epoch 44 and is lowest at epoch 49. Its best epoch, 34, is only 0.0034 above epoch 19.
- Selecting by best `val/AP` (E6) protects both exports from the late-epoch decline. A schedule of about 15–20 epochs would likely reach the same RGB result at less than half the cost. This is an observation, not a spec decision.

## 4. Training health

- **Learning:**
  - `det_rgb`: `train/loss_epoch` 19.44 (epoch 0) → 4.56 (epoch 49), a ratio of **0.234**.
  - `det_ir`: 19.04 → 4.13, a ratio of **0.217**.
  - Both are within the ≤ 0.5 bar.
- **lr:** `train/lr` is 1e-4 in every logged step of epochs 0–39 and 1e-5 in every step of epochs 40–49, for both runs.
- **fp16:** `train/nonfinite_steps` is 0 throughout both full runs and both overfit runs, and the 100-consecutive abort never fired. The GradScaler scale stayed between 512 and 4096.
- **Overfit (Step 9, bar AP50 ≥ 0.8 for person and car on 256 train frames):**
  - `det_rgb` (5526084, hpc-gpu5, L4) passed at step **500**: person 0.895, car 0.901.
  - `det_ir` (5526085, hpc-gpu1, V100S) passed at step **1000**: person 0.939, car 0.969.

## 5. Is encoding the bottleneck?

**No.** Median `perf/encode_share` is 0.120 for `det_rgb` and 0.122 for `det_ir`, far below the 0.5 threshold. The frozen SDPA encoder takes about 184 ms of a 1.52 s step at batch 16 (≈ 87 frames/s). The DINO-DETR head takes the other ≈ 88 %. The F5 feature cache does not need to be revisited for F8's training cost; F4/F7 predicted ≈ 13 %. Training throughput is ≈ 10.5 images/s at batch 16 on an L4. F7's synthetic bench at batch 8 measured 13.1 images/s for the head alone. The real step pays for more boxes per image and for the encoder.

## 6. Staging

Every job staged FLIR to node-local scratch, and every `data_root` is under `/mnt/scratch/leonardo.ulloa/<job>/`. Each log shows `copied 46450 files`:

| job | node | seconds | MB/s |
|---|---|---|---|
| 5526084 (overfit RGB) | hpc-gpu5 | 317 | 38 |
| 5526085 (overfit IR) | hpc-gpu1 | 312 | 38 |
| 5526115 (`det_rgb`) | hpc-gpu5 | 218 | 55 |
| 5526116 (`det_ir`) | hpc-gpu6 | 57 | 213 |

Staging costs 1–5 min per job, which is negligible against a ≈ 14 h run.

## 7. Deviations and amendments in effect
- E1–E6 (`projects/detection/conf/train.yaml` header): backbone lr factor 1.0, `16-mixed`, epoch-interval `MultiStepLR`, validation every 5 epochs with best-`val/AP` selection.
- plan.md Amendments, 2026-10-26: constant lr and unbounded epochs in overfit mode; no metrics from the sanity check.
- plan.md Amendments, 2026-10-27: per-epoch timing and GradScaler logging; overfit memory on V100S accepted; no GPU pinning.
- plan.md Amendments, 2026-10-28: Group 8 item 4 accepted from the full runs; resumability deferred; peak memory from W&B system metrics.

## 8. Not run / caveats
- **No requeue happened.** The forced USR1 requeue of `det_rgb` (plan Step 10) was not done. The SLURM signal → `hpc_ckpt_*` → resume path is untested on the cluster; only the `last.ckpt` resume of Group 8 is tested. This is deferred (validation.md, user decision 2026-10-28).
- **Group 8 item 4 was not re-run as a test** after `b81ef6c`. It was accepted on the full runs' logs: `perf/*` and `train/grad_scale` were present in all 50 epochs, with `encode_share` between 0.118 and 0.123.
- **Peak memory for the full runs** comes from NVML device memory, not the allocator peak (§1, footnote 2).
- **One run per detector, seed 0.** There is no estimate of seed variance. The RGB/IR AP gap (0.302 vs 0.330) is not a comparison of modalities: the val sets differ and nothing controls for it. F6/F9 cover that comparison.
- The val split is used both for checkpoint selection and for the numbers above, so they are optimistic as test estimates.
