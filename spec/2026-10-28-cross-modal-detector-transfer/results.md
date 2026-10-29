# F15 · Cross-modal detector transfer: Results

**Provenance.**
- Run: SLURM job **5526475** on ctcomp3 node hpc-gpu1, **Tesla V100S-PCIE-32GB**, at commit **`1179143`** (`git.dirty: false`), 2026-10-28.
- W&B run **`j3apbc75`** (`leo00010011-usc/ir-world-model`, online).
- Hydra run dir `$IRWM_OUTPUT_ROOT/detection/2026-10-28/13-49-32/` (`transfer.json`).
- Config: `projects/detection/conf/transfer.yaml` defaults, `cluster=ctcomp3`, 8 workers, batch 16, no data staging.
- Heads: `det_rgb` (`96db5065…`, `detection/slurm-5526115/best-e014.ckpt`, epoch 14) and `det_ir` (`2ff38732…`, `detection/slurm-5526116/best-e034.ckpt`, epoch 34). The encoder for both is `native` 448×560, SDPA, fp16, `dinov2_vitb14reg_448x560_l3-6-9-12_pca1152`.
- GPU tests ran before this job, on the same commit: Group 7 + 8 in job 5526472 and Group 8 in job 5526473.

Every number below is from `transfer.json` (`cells`, `transfer_ratio`, `diag_vs_val`, `timing`, `device`) unless stated otherwise. Values are rounded to 3 decimals; the JSON has full precision.

## 2×2 matrix

Each cell is the row's head scored on the column's `video_*_test` split, against that split's own annotations: all 3,749 frames, ignore regions as crowd, 640×512-normalised areas.

**AP**

| | RGB test | IR test |
|---|---|---|
| `det_rgb` | **0.157** | 0.102 |
| `det_ir` | 0.089 | **0.167** |

**AP50**

| | RGB test | IR test |
|---|---|---|
| `det_rgb` | **0.331** | 0.205 |
| `det_ir` | 0.201 | **0.334** |

**AP, person + car** (mean of `AP_person` and `AP_car`, R10)

| | RGB test | IR test |
|---|---|---|
| `det_rgb` | **0.278** | 0.251 |
| `det_ir` | 0.157 | **0.385** |

**AP50, person + car**

| | RGB test | IR test |
|---|---|---|
| `det_rgb` | **0.537** | 0.477 |
| `det_ir` | 0.307 | **0.713** |

## Transfer ratios

Cross-modal value ÷ in-domain head's value on the **same** test split (same GT and frames).

| Cell | AP | AP50 | AP person+car | AP50 person+car |
|---|---|---|---|---|
| `det_rgb` on IR test (÷ `det_ir` on IR) | 0.615 | 0.614 | 0.653 | 0.669 |
| `det_ir` on RGB test (÷ `det_rgb` on RGB) | 0.569 | 0.607 | 0.566 | 0.571 |

## Diagonal vs F8 val

Reported, not gated (user decision 2026-10-28). Val is the head's selection checkpoint value in its `meta.json`.

| Head | test AP | val AP | ratio | test AP50 | val AP50 | ratio50 |
|---|---|---|---|---|---|---|
| `det_rgb` | 0.157 | 0.302 | 0.518 | 0.331 | 0.573 | 0.578 |
| `det_ir` | 0.167 | 0.330 | 0.505 | 0.334 | 0.626 | 0.534 |

All four ratios are inside [0.5, 1.5].

## Per cell: size buckets and per-class AP

| Head | Test | AP75 | APs | APm | APl |
|---|---|---|---|---|---|
| `det_rgb` | rgb | 0.128 | 0.124 | 0.400 | 0.362 |
| `det_rgb` | thermal | 0.089 | 0.068 | 0.250 | 0.595 |
| `det_ir` | rgb | 0.067 | 0.056 | 0.292 | 0.354 |
| `det_ir` | thermal | 0.145 | 0.136 | 0.306 | 0.632 |

Per-class AP / AP50:

| Head | Test | person | bike | car | motor | light | sign |
|---|---|---|---|---|---|---|---|
| `det_rgb` | rgb | 0.215 / 0.472 | 0.012 / 0.029 | 0.340 / 0.603 | 0.191 / 0.461 | 0.044 / 0.139 | 0.136 / 0.284 |
| `det_rgb` | thermal | 0.202 / 0.395 | 0.000 / 0.000 | 0.300 / 0.559 | 0.050 / 0.139 | 0.039 / 0.099 | 0.023 / 0.039 |
| `det_ir` | rgb | 0.077 / 0.169 | 0.003 / 0.009 | 0.237 / 0.444 | 0.151 / 0.400 | 0.024 / 0.092 | 0.042 / 0.093 |
| `det_ir` | thermal | 0.359 / 0.706 | 0.000 / 0.000 | 0.410 / 0.720 | 0.080 / 0.223 | 0.081 / 0.208 | 0.070 / 0.148 |

On thermal, bike AP is 2.1e-6 (`det_rgb`) and 7.4e-5 (`det_ir`). It rounds to 0.000 above. Thermal test has only 113 bike boxes (R2).

## Detection files

Path: `$IRWM_OUTPUT_ROOT/detection/detections/dinov2_vitb14reg_448x560_l3-6-9-12_pca1152/<head>/<modality>_test.json.gz`.

| Head | Test | SHA-256 | Size | n_images | n_detections |
|---|---|---|---|---|---|
| `det_rgb` | rgb | `12f1a74d261db4ef46e2ac744c164d2a68eace62d78407b5569b771a70a3c28a` | 18.2 MB | 3749 | 1,124,700 |
| `det_rgb` | thermal | `fc0036be5bdfb26c7b8efef81a76a38079b69e1d9cbfd835e2b685c4a51dc4b8` | 14.6 MB | 3749 | 1,124,700 |
| `det_ir` | rgb | `e91ac8d1f33fc91bf9cca3a267083f98ca1380eb36bddf14968b8e91eff21622` | 18.6 MB | 3749 | 1,124,700 |
| `det_ir` | thermal | `87f24b0daa5f39e2d8dcf1a2038cc5bb7e1de71522d2596e4204f547c5311d85` | 14.8 MB | 3749 | 1,124,700 |

- **Every frame has exactly 300 detections.** DINO's top-k always returns `select_box_nums_for_evaluation`, and no score threshold is applied. So `n_detections = 3749 × 300` in every file.
- **Hashes.** `sha256sum` of the synced copies equals `transfer.json` `detections[*][*].sha256`.
- **Metadata.** Each file's `head_sha256` equals the frozen `meta.json`. Its `git` is `1179143` (clean), and all 8 test videos are present in `video_tags`.
- **Re-score.** This was Step 6.1, run on CPU on ctdesk236 from the synced files: `with_person_car(coco_map(FlirSplit.load(m, "test").to_coco_gt("normalised"), load_detections(p)["detections"]))`. It reproduces `cells[head][modality]` **exactly**: |Δ| = 0 on all 20 keys of all 4 cells. It took 76–96 s per file. The saved files can therefore stand in for the run (F6, F9, F12).

## Cost

From `timing` and `device`:
- **Encoder:** 26.6 s (RGB) and 25.1 s (thermal) for 3,749 frames each, so ≈ 141 and ≈ 150 frames/s. Each split was encoded once (7,498 frames in total).
- **Heads:** 42.3 / 40.9 s (`det_rgb` on RGB / thermal) and 39.4 / 39.3 s (`det_ir`), so ≈ 11 ms per frame per head. The head forward is timed alone, without `to_flir_detections`.
- **Wall time:** 752 s for the whole job, including startup, `coco.json` parsing, writing four 1.1M-detection files and four `coco_map` runs. Inference itself (encode + heads) was 213.5 s of that.
- **Peak GPU memory:** 2.26 GiB allocated.

## Caveats

- **R10 (annotation asymmetry).** The cross-modal cells compare different labelling densities for sign, light and bike. On the same time-synced frames, thermal GT has far fewer boxes of these classes than RGB GT. So `det_rgb` on thermal is scored against a GT that omits objects it may correctly find, and `det_ir` on RGB is scored against objects that thermal GT never labels. Read the person + car tables for the modality comparison.
- **Preliminary until F6.** There are no bootstrap CIs. Test has 8 sequences per modality (R2), so differences between cells are preliminary. F9 re-scores the diagonal cells with F6.
- **`native` RGB edge crop.** RGB frames are centre-cropped to 5:4 before encoding: 4.4 % of the height for 1224×1024 frames and 6.25 % of the width for 1024×768 frames. The GT is the full FLIR frame, so RGB GT boxes entirely in the cropped edges cannot be detected by either head. This affects both RGB-test cells. Thermal (640×512) is never cropped.
- **GPU.** The job ran on a V100S, not the L4 that the F7/F8 costs were measured on, so the timings are not directly comparable with those.

## Pending

Two Data Architecture Integrity items need a check on ctcomp3 that was not part of the returned outputs:
- **Frozen heads untouched:** `sha256sum $IRWM_CKPT_ROOT/detection/frozen/*/head.pt` and the `meta.json` mtimes after the run.
- **Writes only under the allowed roots:** `git status` on ctcomp3 after the run, and nothing new under `$IRWM_DATA_ROOT` or `$IRWM_CKPT_ROOT`.
