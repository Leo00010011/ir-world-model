# F1 · DINO-Foresight audit: Note

**Status: gate passed (2026-10-07).** The released model loads with the exact weights it was trained with. Its feature recipe is confirmed numerically, and it forecasts real frames better than copy-last on a GPU. Nothing here blocks F4 or F10.

## Sources

| Run | Where | Commit | Sections | Record |
|---|---|---|---|---|
| Main audit | ctcomp3, Tesla V100S-32GB, torch 2.2.0+cu121 | `3783f8f` (clean) | artefacts, weights, recipe, rollout | `audit_results.json` in this folder (kept locally, git-ignored) |
| R5 probe | ctcomp3, job 5523595, W&B `a1oln6go` | `fc991aa` | r5 | `~/irwm/outputs/dinof_audit/2026-10-02/13-56-09/audit_results.json` |

Pinned inputs:
- HF revision `9174a31e86c0ccd4df68c8c51f92ccce3313657a`
- DINOv2 hub code `dinof_release` = `85a24602099d397264d5b30461ad7f3bfd726ca1`
- Submodule `d98f765`

Citations below are `file:line` in `third_party/DINO-Foresight` and `key` paths in the results file. A first run at `2c8d7e6` crashed in `rollout` because of a CPU/GPU device mix in the audit's own dummy-input check (fixed in `3783f8f`). The other sections gave identical numbers in both runs.

**Deferred by user decision (2026-10-02):** the DINOv2 hub-version comparison (`hub`) and the qualitative segmentation check (`segm`). Both are implemented. Run them with `audit.py 'sections=[hub,segm]'` if F4 or F10 shows anything unexpected.

---

## (a) Checkpoints and licences

- All files in `$IRWM_CKPT_ROOT/dino_foresight/9174a31e` and the DINOv2 hub dir match their SHA-256 manifests: `artefacts.verify_manifest` → `[]`. The files are `dinof_highres.ckpt`, `dinof_lowres.ckpt`, the PCA `.pth` and the DPT heads.
- **`dinof_highres.ckpt` loads strictly clean.** The details:
  - `artefacts.load_info.missing_keys` and `unexpected_keys` are both `[]`.
  - It was saved by Lightning 2.4.0 at epoch 19, step 1080. That is the 20-epoch high-res fine-tune, interpolated from a low-res run at epoch 669 (`saved_args.ckpt`).
  - It has 257.8 M world-model parameters (`artefacts.params_world_model`), split into `maskvit` 256.4 M and `embed` 1.3 M. The frozen DINOv2 adds 86.6 M.
- **The checkpoint contains its own encoder and PCA, and both are identical to the official files.**
  - The 176 DINOv2 tensors match the official `dinov2_vitb14_reg4_pretrain.pth`: max |Δ| = 0 (`weights.dinov2_ckpt_vs_official`).
  - The PCA tensors match the released `.pth`: |Δ| = 0 (`weights.pca_ckpt_vs_export`).
  - Our `PCABasis` equals the reference `pca_transform`/`pca_inverse_transform` in both directions: |Δ| = 0 (`weights.pcabasis_vs_reference`).
- **Two saved args are older than the released `train.py`:**
  - `sliding_window_inference` and `high_res_adapt` are missing from the checkpoint. They are filled from the `train.py` defaults (`False`) and recorded in `artefacts.load_info.args_diff`.
  - Other overrides at inference: `eval_mode=False`, `evaluate=False`, and `pca_ckpt` pointed at our local copy.
- `dinof_lowres.ckpt` was downloaded and passes its checksum but was not loaded. We only use the high-res model.
- **Licences:**
  - DINO-Foresight code: MIT (`LICENSE`).
  - Released weights on HF: **MIT** (model-card metadata `license: mit`).
  - DINOv2: Apache-2.0.
  - Research use. We do not redistribute the weights.

## (b) Feature recipe

| Item | Finding | Evidence |
|---|---|---|
| DINOv2 loading | `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)`. It pulls the **unpinned `main`** at runtime. We pin the hub *code* to `dinof_release`. The weights are the official ones either way. | `src/dino_f.py:139`; `recipe.hub_version_used` |
| Blocks | `get_intermediate_layers(x, n=[2,5,8,11])`, 0-indexed, i.e. blocks 3/6/9/12 → 4 × 768 = **3072-d** | `src/dino_f.py:279`; `recipe.feature_dim`, `recipe.embed_dim_per_layer` |
| Final LayerNorm | **Applied to every intermediate block** (DINOv2 default `norm=True`). Hooks + `norm` equal `get_intermediate_layers` with |Δ| = 0, and skipping the norm changes features by up to 92.8. | `recipe.intermediate_equals_hooks_plus_norm`, `recipe.effect_of_final_norm_max_abs` |
| CLS / registers | 2053 tokens out of the backbone. CLS and 4 registers are dropped → **2048 = 32×64** patch tokens. | `recipe.tokens_total_per_image`, `recipe.patch_tokens` |
| Input | 448×896, ImageNet mean/std | `saved_args.img_size` |
| PCA | Standardise by stored `mean`/`std` → subtract `pca_mean` → project on `components` (3072→1152). **No whitening.** fp32 tensors. | `src/dino_f.py:387-398`; `weights.pca_ckpt_vs_export.dtypes_in_ckpt` |
| In-domain sanity | On the Cityscapes demo clip, z-scored dims have mean |μ| 0.11 and mean σ 1.00. PCA explains **90.5 %** of the variance, and the relative reconstruction error is 3.1 %. | `recipe.zscore`, `recipe.pca_explained_variance_demo`, `recipe.pca_relative_recon_error_raw` |

## (c) Temporal protocol

- **Training stride is random:** `num_frames_skip ∈ {1,2,3}` with equal probability, i.e. a step of 2–4 frames at 17 Hz (0.118–0.235 s) (`src/data.py:123`).
  - The high-res fine-tune kept this augmentation (`saved_args.no_timestep_augm = False`, `timestep_augm = None`).
- **Evaluation uses a fixed step of 3 frames** (0.176 s): frames 7, 10, 13, 16 → 19 (`src/data.py:149-157`).
- Context = 4 frames plus 1 masked target slot (`sequence_length = 5`, `train_mask_mode = full_mask`, `masking = simple_replace`).
- **Rollout stays in PCA space.** The prediction overwrites the last slot, then the window shifts left by one (`src/dino_f.py:511-512`).
  - The reference `sample_unroll` equals our manual PCA-space loop for 3 steps: |Δ| = 0 (`rollout.unroll3_equals_manual_pca_loop`).
  - After k steps the 5-slot window holds `[ctx…, p1…pk, pk]`, with the last slot duplicated. `sample_unroll` returns the whole window in 3072-d, shape `[1, 5, 32, 64, 3072]` (`rollout.unroll3_output_shape`). It does not return a stack of predictions.
- **For F3:** FLIR 30 FPS with stride 5 = 0.167 s. That is inside the training range and 5 % from the evaluation step. **Stride 5 is confirmed.** F3 still has to check that the FLIR video really is 30 FPS from the timestamps.

## (d) Spatial flexibility

- The positional embeddings are **factorised tables**: time `d_0[5]`, height `d_1[H]`, width `d_2[W]`.
- With `high_res_adapt=True`, `on_load_checkpoint` bilinearly interpolates `d_1`/`d_2` to the new grid (`src/dino_f.py:231-239`). The high-res checkpoint itself was made this way, from 16×32 tables.
- Separable space–time attention does not depend on the grid size.
- **A 32×40 grid works:**
  - Policy (c), native 448×560, loads with interpolated tables and forecasts at a model/copy-last ratio of 0.648.
  - The uncropped 2:1 reference gets 0.607 (R5 probe).
- See *Decisions → R5*.

## (e) Detection support

**None.** The only heads are DPT heads (segmentation, depth, normals) in `Downstream/` and `dpt.py`. They consume PCA-reconstructed 3072-d features. There is no Mask2Former, DETR or instance code in the release (`grep -ri "mask2former|detr|instance"` finds nothing). **F7 (DINO-DETR port) is required as planned.**

## (f) GPU smoke test

Run on a V100S, fp32 unless stated (`rollout.*`):

| Check | Result | Pass? |
|---|---|---|
| 1-step on real frames: model vs copy-last | SmoothL1 **0.508** vs 0.837 (ratio 0.607); cosine **0.864** vs 0.658; MSE 0.62 vs 1.72 | ✓ |
| Our loss equals the reference loss | `reference_loss` = model SmoothL1 = 0.5079 | ✓ |
| Public `sample` vs our 1-step path | |Δ| = 0 | ✓ |
| 3-step `sample_unroll` vs manual PCA-space loop | |Δ| = 0 | ✓ |
| Rollout does not collapse | cosine between consecutive predicted steps 0.82, 0.85 | ✓ |
| Masked target slot has no influence (zeros / noise) | |Δ| = 0 / 0 | ✓ |
| Determinism (two runs) | |Δ| = 0 | ✓ |
| Dummy PCA-space batch of 2 | `[2, 32, 64, 1152]`, finite | ✓ |
| fp16 autocast vs fp32 | max |Δ| 0.129, mean |Δ| 0.0018; SmoothL1 0.50795 vs 0.50792 | reported |
| Speed | 0.58 s per 1-step forward (median, fp32, batch 1) | reported |

---

## Paper / docs vs code

| Topic | Docs (paper, README, our TechStack draft) | Code / checkpoint (wins) |
|---|---|---|
| PCA layers | README PCA command: `--layers 2,5,9,11` | `2,5,8,11` everywhere else (training commands, PCA filename, `saved_args.d_layers`) |
| Temporal step | TechStack: "stride 3 (0.176 s)" as *the* step | Fixed stride 3 only at evaluation; training samples 2–4 at random |
| DINOv2 version | Not stated | `torch.hub` `main` at runtime, unpinned |
| Final norm on intermediate blocks | Not stated | Applied (DINOv2 default) |
| PCA | "PCA to 1152" | Standardise, then centred PCA, no whitening |
| Downstream heads | Paper reports instance segmentation (Mask2Former) | Not released; only DPT heads |
| Checkpoint args | — | Missing `sliding_window_inference`, `high_res_adapt` (older `train.py`) |

## TechStack corrections (applied)

- **§3.1:**
  - DINOv2 hub code pinned to `dinof_release` `85a24602`; weights from the official `.pth`.
  - Final LayerNorm applied to blocks 3/6/9/12; CLS and 4 registers dropped.
  - Standardise before PCA, no whitening.
  - `PCABasis` is the implementation.
- **§3.3:**
  - Training stride is random (2–4 frames at 17 Hz) and evaluation uses 3; FLIR stride 5 is confirmed.
  - The rollout overwrites the last slot and shifts, in PCA space.
  - Inference precision is fp16 autocast, as in the reference `16-mixed`.
  - Positional tables are interpolated for grids other than 32×64.
  - The weights licence is MIT.
- **§8 Component Registry:** the F1 components are marked ✓.

## Decisions

- **R5: resolution policy. Still OPEN (user decision, 2026-10-02), decided before F5.**
  - Probe: one Cityscapes clip, centre-cropped to 5:4, 1 step. Ratio = model ÷ copy-last SmoothL1:
    - uncropped 2:1 reference: 0.607
    - **(a) stretch: 0.662**
    - (b) crop: 0.551
    - (c) native 32×40: 0.648
  - All policies keep a 34–45 % advantage over copy-last.
  - (b)'s win is most likely an easier central band (it beats the uncropped reference), and it pays with 37.5 % of the field of view.
  - (a) and (c) are within noise of each other.
  - **Caveat:** this is one clip, it measures forecasting only and says nothing about small-object detection, and it is in-domain.
  - Provisional preference: (a). F4 implements (a), (b) and (c) as a config option. Before F5, repeat the probe on a few FLIR RGB and IR clips.
- **R7: cache contents. Decided:**
  - Cache **PCA 1152-d fp16 features only**: about 4.7 MB per frame, about 160 GB in total.
  - Full 3072-d features are not cached (about 420 GB). They are encoded on the fly for a subset to measure the PCA loss per modality (F9, F13).
  - The in-domain baseline for that comparison is the 3.1 % relative reconstruction error and 90.5 % explained variance on Cityscapes (`recipe.*`).
- **DINOv2 hub commit:** `dinof_release` (`85a24602`, the last commit before DINO-Foresight's release). The weights are bit-identical to the checkpoint's copy. The code version was not compared (deferred `hub` section).
- **Inference precision:** fp16 autocast, matching the reference `16-mixed`. The SmoothL1 cost is 4e-5 (0.01 %). For equivalence tests (F4, F10), compare in fp32, where the reference is exact.
- **Frame stride for F3:** 5 at 30 FPS (0.167 s).

## Risks for F4 / F10

1. **Recipe drift in F4.**
   - The encoder must apply the final norm, drop 1 + 4 tokens and standardise before PCA.
   - Enforce this with an fp32 equivalence test against `Dino_f.extract_features` + `pca_transform` on the demo clip. The target is |Δ| = 0, as achieved here.
2. **Hub code version unverified.** Only the code is unpinned, not the weights. If the F10 RGB sanity check is weak, run `sections=[hub]` first.
3. **Rollout window semantics.**
   - F10's `rollout()` should collect each step's prediction directly, as in the manual loop, rather than slicing the `sample_unroll` window, whose last slot is a duplicate.
   - Equivalence test: manual loop vs `sample_unroll`, |Δ| = 0 in fp32.
4. **Grid other than 32×64:** if R5 picks (c), load with `img_size=(448,560)` so `high_res_adapt` interpolates the tables. `inference_args` already handles this.
5. **Domain gap (R9).** Every number above is in-domain Cityscapes. FLIR RGB is out of domain and IR even more so. The F10 sanity check (error grows with horizon, beats copy-last) is the first real test.
6. **Thermal 1→3 channel replication (R4)** is untested by this audit.
7. **Compute (F11):** about 0.58 s per step in fp32 on a V100S, faster with autocast. 6 steps × a few thousand clips × 2 modalities ≈ a few GPU-hours. Not a constraint.
