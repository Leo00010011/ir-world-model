# F4 · DINOv2-Reg encoder wrapper: Results

All numbers come from three runs on **hpc-gpu5 (NVIDIA L4, 24 GB; 48 CPUs, 2 allocated by SLURM)**, torch `2.2.0+cu121`, commit **`4a577ae`** (clean tree):

| Run | Source | W&B | Run dir (`$IRWM_OUTPUT_ROOT/feature_extraction/…`) |
|---|---|---|---|
| FR8 equivalence | `uv run pytest -m data tests/test_encoder.py -s` (printed lines `[F4 Test A/B/C]`, `[F4 determinism]`) | — | — (pytest does not record the commit; branch at `4a577ae` per the user) |
| FR12 throughput | `throughput.json` | `ei06fokp` (SLURM 5525011) | `2026-10-16/16-16-26/` |
| FR11 R5 probe | `r5_probe.json` | `xpo3w3xh` (SLURM 5525012) | `2026-10-16/16-16-47/` |

The backbone everywhere is **DINOv2-Reg ViT-B/14 (`dinov2_vitb14_reg`, 4 registers)** from the hub code pinned at `dinof_release` (`85a24602`). The PCA is `pca_1152.pt` (source SHA-256 `16e897a6f092…`).

## 1. Equivalence with the reference (FR8)

Demo clip `frankfurt_01_017101` as a single unsplit batch `[5, 3, 448, 896]`, on the L4:

| Test | Ours | Reference | max \|Δ\| full (3072) | max \|Δ\| PCA (1152) | Required |
|---|---|---|---|---|---|
| A (recipe) | `precision="fp32"` | `extract_features` / `pca_transform`, no autocast | **0** | **0** | 0 |
| B (production path) | `precision="fp16"` | same calls inside `torch.autocast("cuda", float16)` | **0** | **0** | 0 |

Test B is bit-exact, so FR8's 1e-2 fallback is not used and there is no differing op to name. Running `encode_pca` twice in fp16 gives |Δ| = 0 (determinism).

**fp16 drift of our encoder (Test C, information only):** fp16 vs fp32 `encode_pca` gives max |Δ| = 4.05e-1, mean |Δ| = 3.37e-3, and a mean token cosine of **0.999995** (expected ≥ 0.999). The max is a single outlier coefficient; the mean and the cosine show fp16 barely moves the features. fp16 is the reference behaviour (`16-mixed`), so this is not a gate.

The xFormers warnings in the pytest output come from the pinned hub code falling back to plain PyTorch attention. That fallback is also what the reference runs, and Tests A/B match it exactly.

## 2. Input pipeline and positional embeddings

- **`stretch` equals the reference transform:** `T.Resize((448, 896))` on the whole frame (`third_party/DINO-Foresight/src/data.py:165`), checked with |Δ| = 0 on the demo frames. **`crop` and `native` are our deviations.** Both are deterministic centre crops: `crop` keeps the central 2:1 band; `native` crops to 5:4 and resizes to 448×560, so it **never crops thermal** (640×512 is already 5:4).
- **DINOv2-Reg interpolates its own positional embedding on every forward pass:** `DinoVisionTransformer.interpolate_pos_encoding` (`dinov2/models/vision_transformer.py:180-212` in the pinned `dinof_release` hub copy; called from `prepare_tokens_with_masks`, `:221`) resizes the pretrained square patch grid bicubically to the input grid. This happens for **every** policy, 32×64 included, not only for `native`'s 32×40.
- **DINO-Foresight interpolates its factorised height/width tables once, at load time,** when `high_res_adapt=True` (`Dino_f.on_load_checkpoint`, `src/dino_f.py:234-240`, bilinear). `inference_args(img_size=(448, 560))` turns this on; the probe's `reference_load["(448, 560)"].args_diff` records `high_res_adapt: True`. `stretch`/`crop` use the trained 32×64 tables unchanged.

## 3. Throughput (FR12)

### GPU-only (synthetic input, 5 warm-up + 20 timed batches; `throughput.json` → `gpu_only`)

| Input (H×W) | Precision | B = 1 | B = 16 | B = 64 | Peak memory at B = 16 |
|---|---|---|---|---|---|
| 448×896 (`stretch`, `crop`) | fp16 | 16.8 fps | **17.0 fps** | **oom** | 5.7 GB |
| 448×896 | fp32 | 9.3 fps | 9.3 fps | oom | 7.3 GB |
| 448×560 (`native`) | fp16 | 33.4 fps | **36.0 fps** | 36.8 fps (9.5 GB) | 2.6 GB |
| 448×560 | fp32 | 17.8 fps | 18.9 fps | 18.9 fps (12.1 GB) | 3.3 GB |

- fp16 is 1.8–1.9× faster than fp32 at B = 16 for both sizes, as expected.
- **Validation miss (recorded as failed, user decision 2026-10-19):** at 448×896, **fp16 at B = 64 also runs out of memory** on the 24 GB L4. The item allowed `oom` only for fp32 at B = 64. The cause is the plain attention noted in §1: every layer materialises the full 2053×2053 score matrix per head, so memory grows quickly with batch size.
- **The L4 is the worst-case GPU this project must plan for** (user, 2026-10-19). On it, a 448×896 fp16 encoder batch of 32 fits (the end-to-end run below used B = 32) and 64 does not; B = 16 peaks at 5.7 GB. F8's batch, plus the head's own memory, must fit under that.
- Throughput is flat from B = 1 to B = 16: one 448×896 frame (2048 patch tokens) already saturates the L4. The 448×560 grid (1280 tokens) is 2.1× faster, more than its 1.6× token ratio, because the attention cost grows with the square of the token count.

### End-to-end (FLIR train → `Preprocess("stretch")` in 2 DataLoader workers → fp16 `encode_pca`, batch 32, 10 warm-up + 100 timed batches; `end_to_end`)

| Modality | Frames in split | frames/s | Encode share | Projected seconds per epoch |
|---|---|---|---|---|
| RGB | 10,318 | 17.1 | 1.00 (186.97 s of 186.99 s) | **603 s** (≈ 10 min) |
| Thermal | 10,742 | 17.1 | 1.00 (186.92 s of 186.94 s) | **628 s** (≈ 10.5 min) |

- **Encoding, not data loading, is the bottleneck.** Even with only 2 workers (SLURM `cpus_per_task: 2`), JPEG reading and resizing kept ahead of the GPU, so the end-to-end rate equals the GPU-only rate.
- **For F8:** on-the-fly encoding costs about 10 minutes per epoch per modality on an L4 under `stretch`/`crop`, before any detector compute (about 5 minutes under `native`, from the GPU-only 36 fps). That is the scale of Roadmap F5's revisit trigger (a), "on-the-fly encoding dominates F8 training time": for example, 50 epochs is about 8.5 GPU-hours of encoding alone per detector. F8 decides; F4 only reports it.

## 4. R5 FLIR probe (FR11)

All 330 clips of `flir_test_fs5_c4f6_cs10` (SHA-256 `f9b3735c…c729a6c65`, equal to the pin in `tests/test_clip_index.py`), both modalities, three policies, 1 step. That is 1,980 rows, all finite. Grids are 32×64 for `stretch`/`crop` and 32×40 for `native`. The metric is the **ratio** = model SmoothL1 ÷ copy-last SmoothL1 in PCA space (lower is better; < 1 means the model beats copy-last). Source: `r5_probe.json` → `summary`, `sequence_wins`.

| Modality | Policy | Mean ratio | Median ratio | Summed-SmoothL1 ratio | Worst clip |
|---|---|---|---|---|---|
| RGB | stretch | 0.740 | 0.773 | 0.737 | 0.863 |
| RGB | crop | **0.728** | **0.772** | **0.725** | 0.866 |
| RGB | native | 0.754 | 0.779 | 0.751 | 0.885 |
| Thermal | stretch | 0.808 | 0.806 | 0.800 | 0.957 |
| Thermal | crop | **0.799** | **0.802** | **0.793** | 0.958 |
| Thermal | native | 0.812 | 0.816 | 0.804 | 0.952 |

**Sequence wins** (out of 8 sequences, by per-sequence mean ratio; "A < B" counts the sequences where A is lower):

| Modality | crop < stretch | stretch < crop | stretch < native | native < stretch | crop < native | native < crop |
|---|---|---|---|---|---|---|
| RGB | 6 | 2 | 8 | 0 | 6 | 2 |
| Thermal | 4 | 4 | 6 | 2 | 8 | 0 |

- **Sanity:** the model beats copy-last on every one of the 1,980 clip × modality × policy rows (worst ratio 0.958). No (modality, policy) has a mean ratio ≥ 1.
- **Ordering:** by mean ratio it is crop < stretch < native in both modalities, but the spread is small: 2.6 points (3.5 %) in RGB and 1.3 points (1.6 %) in thermal. `crop` vs `stretch` is clear in RGB (6/8 sequences) and a tie in thermal (4/4). `stretch` beats `native` in 8/8 RGB and 6/8 thermal sequences.
- **Thermal is forecast less well than RGB under every policy:** its ratio is 0.06–0.07 higher. Copy-last is itself a stronger baseline in thermal (mean copy-last cosine 0.736 vs 0.712 for RGB under `stretch`), so part of the gap may be that thermal changes less between frames. This measures forecasting in PCA space only, not detection, so it says nothing about H1/H2 yet.

**Comparison with the F1 Cityscapes probe** (one demo clip centre-cropped to 5:4, V100S; ratios a = stretch 0.662, b = crop 0.551, c = native 0.648; uncropped 2:1 reference 0.607): on FLIR every policy has a much higher ratio (0.73–0.81). The model gains less over copy-last on FLIR than in domain, consistent with the double domain gap (R9). The F1 ordering largely holds: crop lowest and stretch ≈ native. But the gaps shrink a lot over 330 clips and 8 sequences. F1's 17 % advantage for crop over stretch becomes 1.1 % (thermal) to 1.6 % (RGB). That supports F1's reading that crop's single-clip win came mostly from the easier central band. Crop pays for its small edge by dropping 37.5 % of every thermal frame (96 rows at the top and bottom). Native costs 0.5 % (thermal) to 1.9 % (RGB) in mean ratio against stretch, but halves encoding time (§3).

**R5 is still the user's decision.** This note records the evidence and makes no choice. The probe measures forecasting quality, not small-object detection (R3); field of view, px/token and the F8 budget in §3 also bear on it.
