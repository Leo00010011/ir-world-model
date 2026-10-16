# F4 · DINOv2-Reg encoder wrapper: Plan

## Context and Design Decisions

- **The backbone is DINOv2 *with registers*, `dinov2_vitb14_reg` (ViT-B/14, 4 registers), never plain `dinov2_vitb14`.** The reference builds it as `'dinov2_' + args.dinov2_variant` (`src/dino_f.py:139`), and every README command passes `--dinov2_variant "vitb14_reg"`. Plain ViT-B/14 has no registers: it would emit 2049 tokens instead of 2053, give different features, and fail FR8. Use "DINOv2-Reg" in names, docstrings and the results note. Source: F1 `audit.md` (b); user remark, 2026-10-16.
- **Split into a CPU `Preprocess` and a GPU encoder** (user decision). Preprocessing runs per image in DataLoader workers, because RGB frames come in five native sizes (F2 `stats.md`) and cannot be batched before the resize. The encoder takes normalised `[B, 3, H, W]` batches on the GPU. `Preprocess` returns a `ResizeGeometry`, so F8 maps boxes with exactly the image's transform and inverts it for predictions. Rejected alternatives:
  - an encoder that takes raw images, because GPU resizing is not the reference's PIL resize, which breaks exact equivalence;
  - leaving preprocessing to F8, because Roadmap F4 owns the resolution policy and F5/F9 would need it too.
- **Resize uses the reference's own transform: torchvision `Resize` on a PIL image, then `ToTensor` → `Normalize`** (`src/data.py:162-165`, mirrored by `irwm.worldmodel.reference.load_demo_frames`). This is required for |Δ| = 0 in FR8. Thermal goes through `Image.convert("RGB")` before the resize, exactly as the reference opens every frame. That *is* the 1→3 replication of TechStack §3.1.
- **`stretch` is the reference behaviour; `crop` and `native` are deviations.** `T.Resize((448, 896))` with a tuple resizes to that exact size whatever the aspect ratio, so the reference itself would stretch a 5:4 frame. This matters for the Mission's "IR or our modifications?" question, and `results.md` must say it.
- **`native` = centre crop to 5:4, then 448×560 (32×40), for every source frame** (user decision, 2026-10-16). It keeps the aspect ratio (no distortion) and gives one fixed grid. That keeps F8 free of batches grouped by size, and gives both modalities the same interpolated world-model grid, so a grid difference cannot confound RGB vs IR. 5:4 is thermal's own aspect, so **thermal is never cropped**. RGB loses 4.4 % / 6.25 % on test and up to 10 % on train (16.7 % for the 47 frames at 720×480). Rejected alternatives:
  - stretch to 448×560 without a crop: distorts RGB by up to 6 %;
  - aspect-kept per-image grids (H = 448, width rounded to ×14): 5 different RGB grids, so F8 would need to group batches by size;
  - crop everything to the narrowest source aspect (9:8, from RGB train 1800×1600 → 448×504): thermal would lose 10 % and RGB 1024×768 15.6 %.
- **Positional embeddings for 32×40 are interpolated by existing code, in two places.**
  - DINOv2-Reg interpolates its own positional embedding (pretrained on a 37×37 grid) on every forward pass to whatever grid it receives. It already does this at 32×64. No encoder code is needed. The implementer cites the function and `file:line` in the pinned hub copy in `results.md`.
  - DINO-Foresight's factorised height/width tables are interpolated bilinearly at load time when `high_res_adapt=True` (`src/dino_f.py:231-239`). `inference_args(img_size=(448, 560))` turns this on (`lib/irwm/worldmodel/reference.py`).
  - Both mechanisms are approximations the model was never trained on. The R5 probe measures their cost.
- **`crop` = centre 2:1 band via `center_crop_box`** (`lib/irwm/worldmodel/reference.py`). This is the same helper the F1 probe used, so FLIR and Cityscapes probe numbers are comparable. It applies to every source size. `native` uses the same helper with aspect `(5, 4)`.
- **Only deterministic centre crops in F4; no random crops** (user decision, 2026-10-16). The encoder and the world model are frozen, so no crop choice can make them memorise anything. At evaluation the crop must be fixed and identical for all frames of a clip: a crop that changed between frames would look like camera motion to the world model, and a random one would make results non-reproducible. Random crops are a *training* augmentation for the detector, and Roadmap F8 owns that decision.
  - **Note for F8:** a random-position 5:4 crop does nothing to thermal, which is already 5:4. Random crops must also vary the scale, or the two modalities get different augmentation.
  - **Note for Phase 4:** the reference trained the world model with random feature crops (`--random_crop` in every README training command; `crop_feats`, `src/dino_f.py:268-274`).
- **Constraint passed to F6/F9/F12:** under `crop` and `native`, GT boxes in the cropped margins are not visible to any model. Evaluation must score only the visible region, using `ResizeGeometry.apply_boxes` `keep`, and the same way for oracle, copy-last and predictions.
- **Default precision is fp16 autocast over backbone *and* PCA, not just the backbone.** Lightning `16-mixed` (every reference command: `README.md:88-151`, wired in `train.py:151`) wraps the whole forward pass, including `Dino_f.preprocess` → `extract_features` + `pca_transform`. Running the PCA in fp32 after an fp16 backbone would be a small deviation of our own. Fidelity first (Mission, "Desired behaviours"). F1 decided fp16 autocast for inference; this spec extends it to the encoder.
- **Every equivalence test compares like with like: ours and the reference at the same precision** (user decision; table in FR8).
  - **Test A (fp32 vs fp32):** proves the recipe, with no tolerance (F1 `audit.md` Risks 1).
  - **Test B (fp16 autocast vs fp16 autocast):** proves the production path. Both sides run the same fp16 operations in the same order on the same GPU, and those kernels are deterministic, so the target is |Δ| = 0. The fallback ≤ 1e-2 exists because a batch-size difference can change cuBLAS kernel selection. Both sides therefore get the identical 5-frame batch.
  - **Test C (our fp16 vs our fp32):** informational only. It is *expected* to be non-zero (F1 saw mean |Δ| ≈ 2e-3 at the world-model output).
  - fp32 remains a config option for diagnostics. It explicitly disables autocast, so an outer `16-mixed` context cannot silently override it.
- **Outputs are cast to fp32 after encoding.** fp16 → fp32 is lossless, and it gives F8's head and the metrics one dtype. Storage stays fp16 per TechStack §7 if F5 is ever built.
- **`torch.no_grad()`, not `torch.inference_mode()`.** Inference-mode tensors cannot be saved for backward ("Inference tensors cannot be saved for backward"), and F8 feeds the features straight into a trainable head.
- **`train()` is overridden to stay in eval.** Lightning calls `.train()` on every child module at the start of each epoch. Without the override, the backbone's drop-path or other train-mode behaviour would switch on (TechStack §3.1 "frozen").
- **CLS and register tokens are dropped by the DINOv2 hub code itself.** `get_intermediate_layers(..., reshape=False)` returns patch tokens only (2053 → 2048 at 448×896; F1 `recipe.patch_tokens`). The encoder must not slice again. The stub backbone in the tests returns patch tokens only, to mirror this.
- **The backbone loader is reused, not rewritten.** `load_dinov2_backbone` (`lib/irwm/worldmodel/reference.py`) already builds the model from the pinned `dinof_release` hub code, loads the official weights strictly, and freezes it. The PCA comes from `PCABasis.load(dinof_dir() / PCA_EXPORT)` (`lib/irwm/features/pca.py`, `assets.py`). Both are offline.
- **The R5 probe is part of F4** (user decision). It covers **all 330 clips** of `flir_test_fs5_c4f6_cs10`, both modalities, the 3 policies, and **1 step only** (user decisions). Using all clips addresses F1's single-clip noise. 1 step keeps the numbers comparable to F1 and leaves rollout logic to F10. The probe uses **our** encoder (validated by FR8) plus the reference model's `one_step`, so it also exercises the full F4 path on FLIR. The WM forward runs under fp16 autocast, the pipeline default. Clips overlap (F3), so frames are re-encoded per clip. The extra cost is minutes, and it avoids an in-memory cache.
- **Aggregates are reported per sequence, not only per clip.** Clips overlap, so per-clip statistics overstate certainty (F3, R2). The probe reports per-`rgb_video_id` means and how many of the 8 sequences each policy wins.
- **Throughput is measured both GPU-only and end-to-end** (user decision). The end-to-end part needs FLIR train on ctcomp3, which F8 needs anyway. Step 7 does that copy.
- **`one_step` and `pca_metrics` move to `lib`.** Both `dinof_audit` and `feature_extraction` now need them (TechStack §2: shared logic goes to `lib/irwm`). They are moved unchanged.
- **`encoder` becomes a shared Hydra group** in `lib/irwm/conf/`, because F8 and F9 will also read it (TechStack §2, Hydra wiring). `policy: stretch` is only the provisional R5 default.
- **Boxes are COCO `xywh`**, the format `FlirRecord.boxes` uses (`lib/irwm/data/flir.py:86-107`). This avoids a conversion at F8's boundary. Boxes outside a `crop` region return `keep=False`. F8 decides any minimum-visibility filtering.
- **Deferred: policy (d), two windows** (Roadmap F4). Not implemented. `Preprocess` raises `ValueError` for any policy other than the three.
- **Not decided here: whether F8 keeps the encoder out of its checkpoints.** The encoder adds ~87 M parameters (~350 MB) to every `state_dict` it is part of. F4 does not hide its state. F8 decides.

## Step 1 — Move shared reference helpers
Cut `one_step` and `pca_metrics` from `projects/dinof_audit/audit.py:108-126` and paste them unchanged into `lib/irwm/worldmodel/reference.py`. `torch.nn.functional` is already a light import. In `audit.py`, import them from `irwm.worldmodel.reference`. Run `uv run pytest tests/test_dinof_audit.py` to confirm nothing broke. Covers: FR10.

## Step 2 — Preprocessing and geometry
Create `lib/irwm/features/preprocess.py` with `Policy`, `POLICY_SIZE`, `ResizeGeometry` and `Preprocess` per FR1–FR3.
- **Crop box:** `center_crop_box(w, h, (2, 1))` for `crop`, `center_crop_box(w, h, (5, 4))` for `native`, and `(0, 0, w, h)` for `stretch`. Skip `img.crop` when the box is the full frame (always the case for thermal under `native`).
- **`apply_boxes`:** convert to xyxy, clip to `[left, left+cw] × [top, top+ch]`, subtract `(left, top)`, multiply x by `W_out/cw` and y by `H_out/ch`, then back to xywh. `keep = (w_clipped > 0) & (h_clipped > 0)`.
- **`Resize` is created once in `__init__`.** It must receive the `(H, W)` tuple. A single int would preserve the aspect ratio, which is not the reference behaviour.
- **Picklable:** no lambdas, so DataLoader workers can pickle it.

Covers: FR1, FR2, FR3.

## Step 3 — Encoder
Create `lib/irwm/features/encoder.py` with `encoder_tag` and `DinoV2RegEncoder` per FR4–FR7. Core path:

```python
def _features(self, x):                      # x [B,3,H,W]
    self._check(x)
    h, w = x.shape[-2] // 14, x.shape[-1] // 14
    with torch.no_grad(), self._precision_ctx(x):
        f = torch.cat(self.backbone.get_intermediate_layers(x, n=list(self.LAYERS), reshape=False, norm=True), dim=-1)
        z = self.pca.transform(f) if pca else f
    return z.float().reshape(x.shape[0], h, w, -1)
```

`_precision_ctx` behaves as follows:
- `fp16` + CUDA → `torch.autocast("cuda", dtype=torch.float16)`;
- `fp16` + CPU → `contextlib.nullcontext()` plus a one-time `UserWarning`;
- `fp32` → `torch.autocast(x.device.type, enabled=False)`.

`pca_inverse` runs under `torch.autocast(..., enabled=False)`. Register `backbone` and `pca` as submodules so `.to(device)` moves both. Pass the layers as a list, as the reference does (`d_layers` is a list). Covers: FR4, FR5, FR6, FR7.

## Step 4 — Hydra group
Create `lib/irwm/conf/encoder/default.yaml` (FR9 keys, with a comment pointing to R5) and add `- encoder: default` after `flir` in `lib/irwm/conf/irwm_base.yaml`. Covers: FR9.

## Step 5 — Unit tests (CPU, no weights)
Create `tests/test_encoder.py` covering validation.md Groups 1–4. The encoder tests use a stub backbone: an `nn.Module` with an `nn.Linear` parameter and `get_intermediate_layers(x, n, reshape, norm)` that returns, for each `k` in `n`, a deterministic `[B, (H/14)·(W/14), 768]` tensor (for example, the per-patch mean pixel times `k+1`, broadcast). It must record its arguments so the tests can assert `n == [2, 5, 8, 11]`, `reshape is False`, `norm is True`. The PCA is a random `PCABasis` with `dim=3072`, `n_components=1152` built from QR (the pattern in `tests/test_dinof_audit.py`). The demo-frame preprocessing tests (Group 1) use `irwm.worldmodel.reference.demo_frame_paths()`; they need the submodule but no weights. Covers: FR1–FR7, FR9, FR10.

## Step 6 — Weight-dependent equivalence tests
Add `@pytest.mark.data` tests to `tests/test_encoder.py` (Group 5): they skip unless the weights exist and CUDA is available, and the fp32 test also runs on CPU. They load `load_reference(dinof_dir() / DINOF_HIGHRES, pca_ckpt=dinof_dir() / PCA_EXPORT)` and `DinoV2RegEncoder.from_assets(...)` on the same device, then:
- **Test A:** with no autocast, compare `model.extract_features(frames)` and `model.pca_transform(...)` on the flattened `[5, 3, 448, 896]` demo clip against our `precision="fp32"` `encode_full`/`encode_pca`.
- **Test B:** the same reference calls inside `torch.autocast("cuda", torch.float16)`, against our `precision="fp16"` encoder, on the identical tensor.
- **Test C:** our fp16 vs our fp32, recording max/mean |Δ| and the mean token cosine.

Each test prints its max |Δ| so Step 9 can copy the numbers. Covers: FR4, FR8.

## Step 7 — Copy FLIR to ctcomp3 (user)
1. From ctdesk236: `rsync -a --info=progress2 "$IRWM_DATA_ROOT/teledyne_flir_adas_v2/" ctcomp3:<ctcomp3 IRWM_DATA_ROOT>/teledyne_flir_adas_v2/`. Use the value of `IRWM_DATA_ROOT` from ctcomp3's `.env`.
2. On ctcomp3: `uv run pytest -m data tests/test_flir_data.py`.

The user returns the pytest summary line (expected: 8 passed, as on ctdesk236 in F2). Covers: FR11, FR12 (prerequisite).

## Step 8 — Project `feature_extraction`: probe and throughput
Create `projects/feature_extraction/`:
- `pyproject.toml`: name `irwm-feature-extraction`, `dependencies = ["irwm[reference]"]`, `[tool.uv] package = false`, workspace source as in `projects/dinof_audit/pyproject.toml`. Run `uv lock`.
- `conf/r5_probe.yaml` (`defaults: [irwm_base, _self_]`, `project: feature_extraction`, `policies: [stretch, crop, native]`, `modalities: [rgb, thermal]`, `clip_set: flir_test_fs5_c4f6_cs10`, `device: auto`).
- `conf/throughput.yaml` (`policy: ${encoder.policy}`, `num_workers: ${oc.env:SLURM_CPUS_PER_TASK,8}`, `batch_sizes: [1, 16, 64]`, `sizes: [[448, 896], [448, 560]]`, `precisions: [fp16, fp32]`, `e2e_batch_size: 32`, `e2e_warmup: 10`, `e2e_batches: 100`).
- `r5_probe.py` per FR11:
  - load the `stretch`/`crop` reference model once and the `native` one once, freeing each after its policy;
  - encode the 5 frames of a clip as one batch;
  - write JSON with `irwm.utils.tracking.git_info`, and log to W&B with `make_wandb_logger`.
- `throughput.py` per FR12.

Both call `irwm.utils.paths.load_env()` before `@hydra.main` (TechStack §2). Smoke-run both locally on CPU only if the weights are present. Otherwise, check config resolution with `--cfg job`. Covers: FR11, FR12.

## Step 9 — Run on ctcomp3 (user)
1. `uv run pytest -m data tests/test_encoder.py -s`, on a GPU node (`srun` or an interactive allocation per the cluster config).
2. `bash slurm/submit.sh ctcomp3 projects/feature_extraction/throughput.py cluster.timeout_min=60`
3. `bash slurm/submit.sh ctcomp3 projects/feature_extraction/r5_probe.py cluster.timeout_min=120`

The user returns:
- the pytest output (the printed max |Δ| values);
- the two run dirs, or the files `throughput.json` and `r5_probe.json`;
- the W&B run ids.

Covers: FR8, FR11, FR12.

## Step 10 — Results note and constitution updates
Write `spec/2026-10-16-dinov2-reg-encoder-wrapper/results.md` per FR13, from the returned files. Then follow the Roadmap's done convention:
- Roadmap F4 → ✓ DONE with a short outcome note;
- add the probe table to R5 and rewrite R5 option (c) as "centre crop to 5:4, then 448×560" (it no longer keeps the full RGB frame). The decision stays open, for the user;
- TechStack §8: register `preprocess.py`, `encoder.py`, the `encoder` conf group, and `projects/feature_extraction/`; update the TechStack §2 project comment ("build the DINOv2 caches" → "F4 R5 probe + throughput; F5 cache if built").

Covers: FR13.

## Implementation Order
1. Move shared reference helpers
2. Preprocessing and geometry
3. Encoder
4. Hydra group
5. Unit tests → `uv run pytest tests/test_encoder.py tests/test_dinof_audit.py`, then the full `uv run pytest`
6. Weight-dependent equivalence tests
7. Project `feature_extraction`: probe and throughput (Step 8)
8. Copy FLIR to ctcomp3 (user, Step 7; can happen in parallel with steps 1–7)
9. Run on ctcomp3 (user, Step 9)
10. Results note and constitution updates
