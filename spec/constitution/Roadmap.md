# Roadmap

Features are numbered **F0, F1, …**. Each one should fit in a single session through `/create-spec` → `/build-spec`, and each lists its dependencies. The roadmap is updated as we go: split, merge or reorder features whenever reality disagrees with the plan. When a feature is done, mark it ✓ DONE here and update the Component Registry in `TechStack.md`.

Status legend: ☐ todo · ◐ in progress · ✓ DONE · ⛔ blocked

> **2026-10-01 pivot:** the world model is now **DINO-Foresight** (DINOv2), replacing FlowWM (DINOv3), which has no public weights. No feature had been started, so the change only touches the plan. See `Mission.md` → Change log.

---

## Phase 0: Foundations and gates

### F0 · Repository scaffold ✓ DONE (2026-10-02)
> Spec: `spec/2026-10-01-repository-scaffold/`. 36 tests pass on the laptop and on ctcomp3. The env is built with `slurm/setup_env.sh`, and the GPU smoke job was run on ctcomp3 through `slurm/submit.sh`. Includes multi-cluster support (`cluster/<site>.yaml`), with the W&B key read from `~/.wandb_api_key`.

The uv workspace (`pyproject.toml`, `lib/irwm` editable package, `projects/*` members) with `git init` and `.gitignore`. It also covers:
- env-var path resolution (`irwm.utils.paths`)
- base Hydra configs (`paths`, `cluster`, `wandb` groups)
- one `slurm/` sbatch template and a submitit launcher config
- pytest setup (with a `data` marker)
- DINO-Foresight (`Sta8is/DINO-Foresight`) added as a pinned git submodule under `third_party/`

**Done when:** `uv sync` works locally and on the server (uv installed there with `pip install --user uv`), `py -m pytest` passes, and a dummy Hydra entry point runs locally and through SLURM with W&B logging.
**Depends on:** none.

### F1 · DINO-Foresight audit (GATE) ◐
> **Status 2026-10-02.** Spec: `spec/2026-10-02-dino-foresight-audit/`. The code is built and its 19 CPU tests pass (55 in the whole suite). Not committed yet.
> - **Built:** `irwm.worldmodel.assets` (pinned HF revision `9174a31e` and DINOv2 hub commits, SHA-256 manifests); `irwm.features.pca.PCABasis` (pure-torch PCA, no sklearn); `irwm.worldmodel.reference` (runs the unmodified `Dino_f` offline from the pinned DINOv2 copy); `projects/dinof_audit/` (`download.py`, `pca_export.py`, `audit.py` with 7 sections).
> - **Found in the code read** (to be confirmed by the audit run):
>   - `torch.hub` loads the unpinned DINOv2 `main` at runtime.
>   - The final LayerNorm is applied to every intermediate block, and CLS + 4 registers are dropped.
>   - PCA standardises each dimension before projecting, with no whitening.
>   - The rollout feeds back in PCA space.
>   - Training sampled strides of 2–4 frames (17 Hz) at random; evaluation uses stride 3, frames 7,10,13,16 → 19.
>   - Positional embeddings are factorised tables that can be interpolated.
>   - The only heads are DPT heads (segm/depth/normals), applied to PCA-reconstructed 3072-d features. No detection head.
>   - The README's PCA layers `2,5,9,11` are wrong; the code uses `2,5,8,11`.
> - **Found from the checkpoint itself:**
>   - The saved args come from an older `train.py`: `sliding_window_inference` and `high_res_adapt` are missing. They are filled from the released `train.py` defaults and recorded in the diff.
>   - Saved args: 448×896, `d_layers 2,5,8,11`, `sequence_length 5`, `simple_replace` masking with `full_mask`, `16-mixed` precision, `no_timestep_augm=False` (the high-res fine-tune also used random strides).
> - **Weights and recipe** (sections 1–3, from an unauthorised local CPU run, not re-run on the cluster):
>   - manifests OK; 0 missing or unexpected keys; 258 M world-model params;
>   - checkpoint DINOv2 and PCA tensors are identical to the official files (|Δ| = 0); `PCABasis` = reference (|Δ| = 0);
>   - 2048 tokens × 3072 dims; intermediate layers = block hooks + final norm (|Δ| = 0);
>   - demo z-score stds average 1.00; PCA explains 90.5 % of demo variance.
> - **Scope reduced (user decision, 2026-10-02).** Only the R5 probe was run on the cluster. The rollout/precision/determinism checks, the DINOv2 hub-version comparison and the qualitative segmentation check are **deferred**, to be run only if an issue appears. They are already implemented: `audit.py sections=[...]`. Decisions in the meantime:
>   - **Precision and rollout:** follow the reference (`16-mixed`, rollout in PCA space).
>   - **DINOv2 hub code:** pinned to `dinof_release` (`85a24602`) without comparing versions. The weights are identical to the official ones; only the hub *code* is unpinned upstream.
> - **R5 probe** (ctcomp3, job 5523595, W&B run `a1oln6go`, `~/irwm/outputs/dinof_audit/2026-10-02/13-56-09/audit_results.json`) → see **R5** below. **The decision is deferred to training time** (before F5 builds the cache).
Read the released code and check it against the paper before building anything on top of it. The output is an audit note in the spec folder that corrects `TechStack.md §3.1/§3.3` wherever the code disagrees with the paper. Questions to answer:
- **(a) Checkpoints.** Download `dinof_highres.ckpt`, `dinof_lowres.ckpt` and the PCA file from Hugging Face, and confirm that they load. Note the licence of the weights.
- **(b) Feature recipe, exactly:**
  - how DINOv2 is loaded (torch.hub or a vendored copy, the weights source)
  - block indices (`d_layers 2,5,8,11` is 0-indexed)
  - whether the final LayerNorm is applied to intermediate blocks
  - CLS and register handling, normalisation and resize
  - how PCA is applied (centring, whitening, component order)
- **(c) Temporal protocol in their Cityscapes loader:** frame stride, context length, and how the autoregressive rollout feeds predictions back (in PCA space or full space).
- **(d) Spatial flexibility.** Do the positional embeddings and separable attention accept a token grid other than 32×64? This decides the FLIR resolution policy (R5).
- **(e) Detection support.** Is there any detection or instance head we could reuse, such as the Mask2Former head from the paper?
- **(f) Smoke test.** Run one forward pass and a 3-step rollout on dummy features on a GPU.

**Done when:** the audit note answers (a)–(f), the feature-cache contents are decided (R7), and TechStack is updated. *(2026-10-02: R5 was taken out of F1's gate. The probe data is recorded in R5, and the decision is made at training time, before F5.)*
**Depends on:** F0. **Blocks:** F4, F10.

---

## Phase 1: Data and features

### F2 · FLIR ADAS v2 readers and class mapping ☐
Read-only readers for the `train`/`val` image splits and the `video_*_test` split for both modalities, returning image + COCO annotations. Includes the canonical class mapping (`irwm/data/classes.py`: 15 FLIR classes → 3 coarse classes and the full set) and a dataset-statistics script giving counts per split, modality and class, the size distribution (S/M/L), and image sizes.
**Depends on:** F0, and the user having downloaded the dataset to `$IRWM_DATA_ROOT`.

### F3 · Video clip index ☐
Parse the video test split into sequences and recover frame order and timestamps. Subsample with the **frame stride chosen to match DINO-Foresight's step**: Cityscapes uses 17 Hz with stride 3 (≈0.176 s), so the default for FLIR's 30 FPS is stride 5 (0.167 s); F1 confirms it. Build every valid clip of **4 context + 6 future steps** (stride between clips is configurable) per modality. The index is stored as a small file (JSON/Parquet). The feature also reports how many clips and sequences each modality has, and checks the RGB↔thermal map for consistency.
**Done when:** the clip index exists for both modalities and the clip counts are recorded in the spec.
**Depends on:** F2 (and F1 for the stride).

### F4 · DINOv2 encoder wrapper ☐
`irwm.features.encoder`: a frozen DINOv2-Reg ViT-B/14 that follows the recipe confirmed in F1 (blocks 3/6/9/12 concatenated → 3072-d, then DINO-Foresight's PCA → 1152-d), with the resolution policy as a config option (R5 is still open: implement (a) stretch, (b) crop and (c) native; (d) two windows only if R5 picks it), with 1→3 channel replication for thermal. Exposes both `encode_full` (3072-d) and `encode_pca` (1152-d), plus `pca_inverse`. Must produce **numerically identical** features to DINO-Foresight's own extraction on the same image (within an fp tolerance).
**Depends on:** F0, F1.

### F5 · Feature cache builder ☐
A SLURM-friendly, resumable job that encodes every frame (train, val, video test; RGB and IR) into the HDF5 layout in TechStack §4.2. It includes a reader API (lookup by `image_id`, a Lightning-compatible dataset) and a storage/throughput report. At 32×64 tokens the PCA cache is about 4.7 MB per frame, so about **160 GB** in total (see R7).
**Depends on:** F2, F4.

---

## Phase 2: Detection on real features (H1)

### F6 · Detection evaluator ☐
`irwm.eval`, built as pure functions with tests on synthetic boxes:
- COCO-eval wrapper (AP, AP50, APS/M/L) over the 3 coarse classes and over all FLIR classes
- per-horizon breakdown
- bootstrap CIs over clips (R2)

Results are written as JSON.
**Depends on:** F0 (independent of the data; can run in parallel with Phase 1).

### F7 · DINO-DETR head port ☐
DINO-DETR as a plain `nn.Module` on `[B, C, H', W']` features (C = 1152 PCA channels by default; see TechStack §3.2): query selection, denoising queries, Hungarian matcher and criterion. Architecture hyperparameters are copied from the detrex reference config. Unit tests cover shapes and loss, and the head must overfit a tiny synthetic set.
**Depends on:** F0.

### F8 · Detector training pipeline ☐
A `LightningModule` and a `LightningDataModule` over cached features, with Hydra configs for `det_rgb` and `det_ir`. The model is trained on `train` and selected on `val` with W&B logging. An optional Optuna sweep config can be added. Produces the two frozen detector checkpoints.
**Depends on:** F5, F6, F7.

### F9 · Oracle evaluation → H1 ☐
Evaluate `det_rgb` on RGB video-test features and `det_ir` on IR video-test features, on the primary (AP, AP50) and secondary metrics, and also per static `val` split. Produces **Result Table 1: does IR beat RGB on real DINOv2 features?** It also includes a check on a `val` subset using full 3072-d features (encoded on the fly) to show how much the Cityscapes PCA costs each modality (R7).
**Depends on:** F8.

---

## Phase 3: Zero-shot world model (H2, H3)

### F10 · DINO-Foresight adapter ☐
`irwm.worldmodel`: load the checkpoint, and `rollout(context_feats, n_steps)` → `[n_steps, H', W', 1152]`, autoregressive exactly as in the reference code and deterministic. As a **sanity check**, run it on a handful of FLIR RGB clips: error should be lowest at step 1, grow with the horizon, and beat copy-last.
**Depends on:** F1 (gate passed), F5.

### F11 · Rollout job and prediction cache ☐
A resumable SLURM job that rolls out every clip from F3 for both modalities for 6 steps, stored in the HDF5 layout in TechStack §4.3. It reports the storage size and GPU-hours.
**Depends on:** F3, F10.

### F12 · Prediction evaluation → H2 ☐
Apply the frozen `det_rgb` and `det_ir` heads to the predicted features. Report AP and AP50 for each modality at short-term (1 step), mid-term (3 steps) and every step up to 6, alongside these baselines on the same frames:
- **oracle**: real features
- **copy-last-context-frame**: real features, so representation is perfect but the forecast is naive

Produces **Result Table 2: does the IR advantage survive prediction?**
**Depends on:** F9, F11.

### F13 · Decoupling analysis → H3 ☐
Separate forecasting error from representation error. Candidate analyses, to be finalised in the spec:
- PCA truncation loss per modality: reconstruction error of the Cityscapes PCA on RGB vs IR, and its effect on AP
- oracle-minus-prediction gap as a function of horizon, extrapolated to t→0
- autoregressive drift: the gap at step k under teacher-forced vs fully autoregressive context
- feature-space distance between predictions and ground truth (cosine, SmoothL1 in PCA space)
- distance of predictions to the real-feature manifold (kNN distance to a bank of real features, or a density estimate)
- feature statistics: norms, per-channel moments, spectrum, spatial high-frequency energy (blur)
- detector confidence calibration on real vs predicted features

Each is compared across RGB and IR. Produces the figures and tables for the first thesis claim.
**Depends on:** F12.

---

## Phase 4: Unsupervised adaptation (H4). Draft, to be refined after Phase 3

### F14 · Adaptation protocol ☐
Set up adaptation without leaking the test set:
- split the video sequences into disjoint *adapt* and *test* subsets, or find more unlabeled thermal video
- fix the evaluation harness
- reproduce the Phase 3 numbers on the new test subset

**Depends on:** F12.

### F15+ · Adaptation techniques ☐
One feature per technique, each evaluated with the F14 harness. Candidates:
- **PCA re-fit on target-domain (unlabeled) features.** This needs either a mapping into the old basis or an adapted model.
- **Feature-manifold projection:** map predicted features onto the real-feature manifold with a denoiser or autoencoder trained only on real (unlabeled) features.
- **Statistics alignment:** match the per-channel moments of predicted features to real IR features (AdaBN-style).
- **Unsupervised fine-tuning of DINO-Foresight on unlabeled IR video:** its own masked-feature SmoothL1 loss, with no labels.
- **Test-time adaptation** of the world model on context frames.

Generalisation check: apply the best technique to RGB and, if feasible, to another domain.

---

## Open questions and risks
| # | Item | Resolved in |
|---|---|---|
| R1 | ~~FlowWM checkpoint not public~~. Resolved by switching to DINO-Foresight, whose checkpoints are public. Still open: verify that they load and check the licence of the weights. | F1 |
| R2 | The FLIR video test split is small (~3.7k frames per modality, a few sequences), so there may be few clips and wide confidence intervals. Use bootstrap CIs over clips. | F3, F6 |
| R3 | FLIR has few *large* objects. The primary metric is now AP/AP50 over all sizes, but small objects on a 14-px patch grid may give low absolute AP. | F2 stats, F9 |
| R4 | 1→3 channel replication for thermal is a design choice that may itself hurt DINOv2 IR features. | F4 / F9 |
| R5 | **Aspect ratio. OPEN: decide at training time, before F5** (it fixes the cache grid and both detectors, so changing it later means re-encoding everything and retraining). DINO-Foresight was trained at 448×896 (2:1 → 32×64 tokens of 14 px). FLIR thermal is 640×512 (5:4) and RGB varies. Options, with px of the original FLIR frame per token: **(a)** stretch to 448×896: full frame, trained size, ~1.6× distortion, 10×16 px/token. **(b)** crop the central 2:1 band, then resize to 448×896: no distortion, trained size, but drops 96 rows top and 96 bottom (37.5 % of the frame; boxes lost or truncated there), 10×10 px/token. **(c)** native 448×560: full frame, no distortion, but an untrained 32×40 grid (37.5 % fewer tokens, positional embeddings interpolated 64→40) and 16×16 px/token. **(d)** two overlapping 2:1 windows (top and bottom) merged: full frame, no distortion, trained grid, but 2× cache and compute (reference `sliding_window_inference` code exists). Not tested. **F1 probe** (one Cityscapes demo clip center-cropped to 5:4; 1-step; ratio = model SmoothL1 ÷ copy-last SmoothL1 in PCA space, lower is better): uncropped 2:1 reference 0.607; (a) 0.662; (b) 0.551; (c) 0.648. **Reading:** every policy beats copy-last by 34–45 %, so neither the stretch nor the 32×40 grid breaks the model. (a) and (c) are tied within noise (one clip). (b)'s win is likely because the central band is easier: it even beats the uncropped reference, and it pays with the lost field of view. The metric measures forecasting only, not small-object detection (R3). **Before deciding:** repeat the probe on a few FLIR RGB/IR clips (no labels needed), and weigh field of view and px/token against detection needs. Provisional preference: (a). | before F5 |
| R6 | Phase 4 needs unlabeled IR video that is disjoint from test. | F14 |
| R7 | **PCA basis fit on Cityscapes RGB.** It may truncate IR features more than RGB, which would confound H1 and H2. Caching full 3072-d features would cost about 420 GB, against about 160 GB for the PCA cache. Default: cache PCA features, and measure the PCA loss on a subset with full features. | F1, F9, F13 |
| R8 | DINO-Foresight is **deterministic** (SmoothL1 regression), so predictions may be blurred toward the mean. That is part of what H3 measures, but it also means no sampling and no best-of-N. | F13 |
| R9 | Double domain gap: the model was trained on Cityscapes (German cities, daytime RGB). FLIR RGB is already out of domain, so the RGB sanity check matters before any IR conclusion is drawn. | F10 |
