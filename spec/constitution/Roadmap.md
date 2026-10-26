# Roadmap

Features are numbered **F0, F1, …**. Each one should fit in a single session through `/create-spec` → `/build-spec`, and each lists its dependencies. The roadmap is updated as we go: split, merge or reorder features whenever reality disagrees with the plan. When a feature is done, mark it ✓ DONE here and update the Component Registry in `TechStack.md`.

Status legend: ☐ todo · ◐ in progress · ✓ DONE · ⛔ blocked · ⏸ deferred

> **2026-10-01 pivot:** the world model is now **DINO-Foresight** (DINOv2), replacing FlowWM (DINOv3), which has no public weights. No feature had been started, so the change only touches the plan. See `Mission.md` → Change log.

> **2026-10-15 re-scope (current milestone):** the next goal is a **working DINO-DETR detector trained on DINOv2-Reg features**, for RGB and IR, as fast as possible. The critical path is **F4 → F7 → F8**:
> - **F5 (feature cache) is deferred.** F8 encodes features on the fly with the frozen F4 encoder. The cache is built later only if it turns out to be needed (see F5).
> - **F6 (evaluator) and F9 (oracle evaluation) are deferred.** They come back after the detector exists, before H1 is claimed. F8 selects checkpoints with an off-the-shelf val mAP instead.
> - Feature numbers are kept stable, so the execution order is no longer the numeric order.
> - R5 (aspect-ratio policy) must now be decided **before F8**, because it fixes the token grid the detectors are trained on.

> **2026-10-25 pipeline efficiency (user):** the goal is unchanged: measure the effect of domain shift on world models (Mission). Compute is limited, so an efficient pipeline is a means to that end. Three optimisations were measured on the worst-case GPU (L4) under F7: fused SDPA attention in the encoder, data staged on node-local scratch, and detrex's compiled CUDA MSDA (`spec/2026-10-23-dino-detr-head-port/results.md` §4, §6). They feed F8's budget, and the thesis can report them as supporting engineering results.

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

### F1 · DINO-Foresight audit (GATE) ✓ DONE (2026-10-07)
> Spec and audit note: `spec/2026-10-02-dino-foresight-audit/` (`audit.md`). Run on ctcomp3 (V100S) at commit `3783f8f`.
> - **Gate passed.** Loading, weights, recipe and rollout match the reference:
>   - The checkpoint loads with 0 missing/unexpected keys. Its DINOv2 and PCA tensors are identical to the official files.
>   - Blocks 2,5,8,11 + final norm, CLS and 4 registers dropped, standardise-then-PCA with no whitening.
>   - 1-step on real frames beats copy-last: SmoothL1 0.508 vs 0.837, cosine 0.864 vs 0.658.
>   - The 3-step rollout equals a manual PCA-space loop, and the model is deterministic (|Δ| = 0).
>   - Weights licence: MIT.
> - **Decisions:**
>   - DINOv2 hub code pinned to `dinof_release` (`85a24602`).
>   - Inference in fp16 autocast (reference `16-mixed`).
>   - FLIR stride 5 confirmed.
>   - R7: cache PCA features only.
>   - R5 stays open until before F5.
> - **Deferred (user decision):** the hub-version comparison and the qualitative segmentation check (`audit.py 'sections=[hub,segm]'`). Run them only if F4 or F10 shows an issue.

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

### F2 · FLIR ADAS v2 readers and class mapping ✓ DONE (2026-10-09)
> Spec and stats note: `spec/2026-10-08-flir-readers-class-mapping/` (`stats.md`), visual summary in `notebooks/flir_stats.ipynb`. Stats and real-data tests run on ctdesk236 (CPU) at commit `a4d65ef`; 96 default tests + 8 `data` tests pass.
> - **Label space:** 6 FLIR classes (person, bike, car, motor, light, sign), labels `0..5` by ascending FLIR id, no coarse mapping. All 10 other classes with boxes are ignore regions; `FlirSplit.to_coco_gt` emits each ignore box once per kept class with `iscrowd: 1`. Size buckets use areas normalised to 640×512.
> - **API:** `FlirSplit` (metadata only, per (modality, split); `image_id` is unique only within a split), `FlirImageDataset` (native-resolution `uint8`, thermal `[H, W]` mode `L`, RGB `[H, W, 3]`), `load_rgb_thermal_map`, `video_tags`. Data root `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/` (folder renamed by the user from the misspelt download name).
> - **Data facts:** 30,787 frames (RGB 10318 / 1085 / 3749, thermal 10742 / 1144 / 3749), all JPEGs match `coco.json` size and mode, 1 orphan RGB train JPEG. Kept test boxes: RGB 81,221, thermal 58,685. Test = 8 time-synced sequences per modality (208–1033 frames), **no per-frame timestamps**; frame-index gaps of 271 + 178 in one sequence and 1253 in another. Thermal frames are all 640×512; RGB test frames are 1224×1024 (2151) or 1024×768 (1598).
> - **Findings that affect later features:** ~86–89 % of kept test boxes are small (R3); thermal test has far fewer sign/light/bike boxes than RGB on the same frames (R10); bike has only 113 thermal test boxes (R2).
> - **Deferred:** copying the dataset to ctcomp3 (needed before F5). *Done during F4 (2026-10-19): 8/8 `data` tests pass on ctcomp3, but take 174 s there vs 9 s on ctdesk236 (R11).*

Read-only readers for the `train`/`val` image splits and the `video_*_test` split for both modalities, returning image + COCO annotations. Includes the canonical class mapping (`irwm/data/classes.py`; planned as 15 FLIR classes → 3 coarse classes, decided as 6 FLIR classes + ignore regions, see the note above) and a dataset-statistics script giving counts per split, modality and class, the size distribution (S/M/L), and image sizes.
**Depends on:** F0, and the user having downloaded the dataset to `$IRWM_DATA_ROOT`.

### F3 · Video clip index ✓ DONE (2026-10-14)
> Spec and results note: `spec/2026-10-14-video-clip-index/` (`clips.md`). Built on ctdesk236 (CPU); 145 default tests + 4 `data` tests pass.
> - **Clip set `flir_test_fs5_c4f6_cs10`** (frame stride 5, 4 context + 6 future, clip stride 10), committed at `lib/irwm/data/clip_sets/` (SHA-256 pinned in tests; `.gitattributes` `-text` keeps its bytes exact): 8 pairs, 11 gap-free segments, **330 clips, 304 evaluable** (kept future boxes ≥ 1 in both modalities; 7 clips empty in RGB, 19 in thermal, 0 in both). Two builds are byte-identical.
> - **One index shared by both modalities:** one row = one time window with the RGB and the thermal `image_id`s of the same time-synced frames, so `clip_id` 15 is the same window in both. Ids follow RGB `video_id` order (paired videos sort differently), anchored at each segment start, and are assigned before the evaluable flag, so they never depend on labels. References only: no pixels or annotations are copied.
> - **API:** `ClipIndex.load()` (needs no data or env var), `Clip.context_ids/future_ids(modality)`, `ClipIndex.evaluable()`. Builder: `projects/clip_index/build.py install=true`; a differing clip set is never overwritten, so new parameters mean a new `clip_set` name.
> - **Data facts:** the RGB↔thermal map is verified (bijection over 3749 frames, equal `frame_index`, 8↔8 videos). 30 FPS comes from the README and cannot be verified from the data. Clips overlap (stride 10 < span 46 frames), so F6 must resample by `rgb_video_id`/`segment_index`, not by clip.

Parse the video test split into sequences and recover frame order from `FlirRecord.video_id`/`frame_index` (F2). Subsample with the **frame stride chosen to match DINO-Foresight's step**: **stride 5 at 30 FPS (0.167 s)**, confirmed by F1 (evaluation step 0.176 s, training range 0.118–0.235 s). The test split has **no per-frame timestamps** (F2), so 30 FPS cannot be verified from the data: take it from the README and record that. **Clips must not span a `frame_index` gap** (F2 found gaps of 271 + 178 in one sequence per modality and 1253 in another). Build every valid clip of **4 context + 6 future steps** (stride between clips is configurable) per modality. The index is stored as a small file (JSON/Parquet). The feature also reports how many clips and sequences each modality has, and checks the RGB↔thermal map for consistency.
**Done when:** the clip index exists for both modalities and the clip counts are recorded in the spec.
**Depends on:** F2 (and F1 for the stride).

### F4 · DINOv2 encoder wrapper ✓ DONE (2026-10-19)
> Spec, results note and notebook: `spec/2026-10-16-dinov2-reg-encoder-wrapper/` (`results.md`), `notebooks/encoder_r5_results.ipynb`. GPU runs on ctcomp3 node hpc-gpu5 (**NVIDIA L4, 24 GB**) at commit `4a577ae`; 179 default tests + 7 `data` encoder tests pass. W&B `ei06fokp` (throughput), `xpo3w3xh` (R5 probe).
> - **Equivalence:** our encoder equals `Dino_f.extract_features` + `pca_transform` with |Δ| = 0 in fp32 **and** in fp16 autocast (same 5-frame batch); deterministic. fp16 vs fp32 drift: mean token cosine 0.999995.
> - **API:** `Preprocess(policy, modality)` → (`float32 [3, H, W]`, `ResizeGeometry` with `apply_boxes`/`invert_boxes` on COCO xywh); `DinoV2RegEncoder.from_assets()` with `encode_full`/`encode_pca`/`pca_inverse`, always eval, `no_grad` (not inference mode), fp16 autocast default; `encoder_tag(policy)`; shared Hydra group `encoder` (`policy: stretch` provisional). Policies: `stretch` (= reference), `crop` (centre 2:1), `native` (centre 5:4 → 448×560; never crops thermal). `one_step`/`pca_metrics` moved to `irwm.worldmodel.reference`.
> - **Encoding is slow, and it is the training bottleneck.** L4, fp16: **17 frames/s at 448×896, 36 at 448×560** (fp32 ≈ half). End to end with real JPEGs and only 2 workers, encoding is 100 % of wall time, so data loading is not the limit: **≈ 10 min of encoding per epoch per modality** under `stretch`/`crop` (603 s RGB, 628 s thermal), ≈ 5 min under `native`. Batch 64 at 448×896 does not fit in 24 GB (failed validation item, kept open by user decision: the **L4 is the worst-case GPU** to plan for).
> - **Tests are slow on the cluster (R11).** The `data` tests that only read files take minutes there: FLIR readers **174 s** (9 s on ctdesk236, same code and data) and encoder equivalence **92 s**, almost all of it loading (`coco.json` parsing, the full `dinof_highres.ckpt`, two backbone builds). These are one-off startup costs per process, not per epoch.
> - **R5 FLIR probe** (330 clips × 2 modalities × 3 policies, 1 step): see R5. Native is 2.1× faster for a 0.5–1.9 % higher forecast ratio. **Decision still open (user).**
> - **Also measured:** ImageNet normalisation on thermal is fine (R4). The FLIR dataset is now on ctcomp3 (deferred from F2).
`irwm.features.encoder`: a frozen DINOv2-Reg ViT-B/14 that follows the recipe confirmed in F1 (blocks 3/6/9/12 concatenated → 3072-d, then DINO-Foresight's PCA → 1152-d), with the resolution policy as a config option (R5 is still open: implement (a) stretch, (b) crop and (c) native; (d) two windows only if R5 picks it), with 1→3 channel replication for thermal. Exposes both `encode_full` (3072-d) and `encode_pca` (1152-d), plus `pca_inverse`. Uses the DINOv2 hub code pinned in `irwm.worldmodel.assets` (`dinof_release`) and `irwm.features.pca.PCABasis`. Must produce **numerically identical** features to DINO-Foresight's own extraction (`Dino_f.extract_features` + `pca_transform`) on the demo clip, in fp32 (F1 reached |Δ| = 0). The recipe is in `audit.md` (b): final norm on every block, drop 1 + 4 tokens, standardise before PCA.
Because F8 now encodes on the fly, the wrapper must also be usable **inside a training step**: batched GPU input, fp16 autocast, frozen (`eval()`, `requires_grad_(False)`, no grad), and a throughput figure (frames/s on ctcomp3) recorded in the spec so F8 can budget epochs.
**Depends on:** F0, F1.

### F5 · Feature cache builder ⏸ deferred (2026-10-15)
> **Deferred:** not needed for the current milestone. F8 encodes on the fly. Revisit if (a) on-the-fly encoding dominates F8 training time *(F4: likely; encoding alone is ≈ 10 min per epoch per modality on an L4 at 448×896, ≈ 5 min at 448×560, i.e. ≈ 8.5 GPU-hours per 50-epoch detector)*, (b) Phase 3 (F10–F13) needs the same features many times, or (c) R7 asks for full-feature analyses at scale. *(F7, 2026-10-26: (a) no longer holds. With SDPA the encoder runs at ≈ 89 frames/s and is ≈ 13 % of a native training step; the head dominates.)* The design below is kept as the starting point.

A SLURM-friendly, resumable job that encodes every frame (train, val, video test; RGB and IR) into the HDF5 layout in TechStack §4.2. It includes a reader API (lookup by `image_id`, a Lightning-compatible dataset) and a storage/throughput report. At 32×64 tokens the PCA cache is about 4.7 MB per frame, so about **145 GB** for the 30,787 frames counted in F2 (see R7). Frames come from `irwm.data.flir.FlirImageDataset`. The dataset is on ctcomp3 (copied during F4).
**Depends on:** F2, F4.

---

## Phase 2: Detection on real features (H1)

**Current focus: F8** (F7 ✓ 2026-10-26). The goal of this phase in the current milestone is a DINO-DETR detector that takes DINOv2-Reg features as input and is trained on FLIR, one per modality. Rigorous evaluation (F6) and the H1 comparison (F9) come after it. Execution order: **F7 → F8**, then F6 → F9.

### F7 · DINO-DETR head port ✓ DONE (2026-10-26)
> Spec, results note and notebook: `spec/2026-10-23-dino-detr-head-port/` (`results.md`), `notebooks/bench_short_runs.ipynb`. detrex @ `e244e6c` (+ detectron2 @ `70f4543`) is an unmodified, editable submodule with compiled CUDA ops (nvcc 12.4 + gcc 8.5.0, 801 s on the ctcomp3 login node). GPU runs on L4 (hpc-gpu5/6) at commit `9b5a7ff` (+ changes later committed in `ca02d8e`); 213 default tests + 6 `gpu` tests pass. W&B `4c11kfnm` (head bench, CUDA MSDA), `64y1udph` (PyTorch MSDA), `6o47ocr3` / `xlop65eg` (encoder, BeeGFS / staged).
> - **Model:** detrex's own `dino_vitdet`, with the ViT replaced by `FeatureStem` (LN + Linear 1152→768); 23,001,286 params. **detrex and detectron2 are runtime dependencies** (user decision 2026-10-23, superseding "no runtime dependency" below). Equivalence = `state_dict` structure match against stock detrex; deviations D1–D8 only.
> - **Overfit:** passes at 200 iterations (loss ratio 0.086, min IoU 0.951, min score 0.704; 32×64, fp32, B = 4, 0.7 s/step, 7.6 GiB).
> - **Head cost (32×40, L4):** train ≈ 8 img/s fp32 / **13 img/s fp16** (B = 8 fp16: 9.1 GiB), infer 29 / 50 img/s at B = 8. Compiled MSDA is 1.5–1.9× faster in training and 1.9–2.8× in inference than detrex's PyTorch fallback, at ≈ 0.6× the memory. **The head, not the encoder, bounds F8's step.**
> - **Pipeline efficiency (limited compute):** fused SDPA (`set_attention(backbone, "sdpa")`) gives **2.44×** (37.0 → 90.4 fps, fp16, B = 64) at −65 % memory, with drift below fp16-vs-fp32 (mean token cosine 0.9999982). Staging FLIR on scratch (283 s copy) speeds up the loader but not end to end: the GPU stays the limit. Full detector inference (thermal): **79.6 → 31.2 ms per frame (2.55×)**.
> - **Amendments:** head bench at 32×40 only, with infer mode and an MSDA switch; `.gitignore` anchored to `/feature_extraction/`; `opencv-python-headless`; analytic GIoU test expectation.
> **2026-10-23: the reference is FlowWM's detector.** FlowWM ([`facebookresearch/Flow-World-Models`](https://github.com/facebookresearch/Flow-World-Models) @ `c150625`, arXiv:2606.29059 §5.1) runs DINO-DETR from **detrex** (`projects/dino`, Apache-2.0) on frozen DINOv3 features. Its trainer loads a detectron2 `LazyConfig` (`detrex/projects/dino/configs/models/dino_vitdet_dinov3_multi_layer_pooler_512_w_layer_norm.py`) and calls `detector(batch, inference_from_backbone=True)`, so the world-model latents `[D, H, W]` go straight into the detector's neck and skip the image backbone. That config, the backbone modification and the detector checkpoints are **not released**: the repo only imports detectron2/detrex lazily. The name and the paper ("average pool token features from layers 3, 6, 9, 12") suggest a multi-layer pooling stem with LayerNorm in front of detrex's stock **`dino_vitdet`** model. We have no source for that stem, so this reading is our inference. **We follow the same route.** We reuse detrex's DINO + ViTDet code as closely as we can. Our frozen F4 features take the place of the ViT backbone output, and we keep the rest of the model and its hyperparameters as they are.

**Goal:** port detrex's `dino_vitdet` model (`projects/dino/configs/models/dino_vitdet.py` on top of `dino_r50.py`, detrex @ `e244e6c`) into `irwm/detection/` as a plain `nn.Module` that takes our features `[B, C, H', W']` in place of the ViT `last_feat`. C = 1152 PCA channels by default. It is configurable, so 3072-d full features can be tried (TechStack §3.2). Default grid 32×64. The port keeps these parts of detrex:
- **ViTDet simple feature pyramid** on the single map: scale factors (2, 1, ½) plus a max-pool top level, so 4 levels of 256 channels with LayerNorm. This settles the multi-scale question.
- **`ChannelMapper` neck**, sine position embedding, and the **DINO transformer**: deformable encoder and decoder, 6 + 6 layers, 256-d, 8 heads, FFN 2048, 4 feature levels, two-stage query selection, 900 queries.
- **Contrastive denoising**: 100 dn queries, label noise 0.5, box noise 1.0.
- **Hungarian matcher and `DINOCriterion`**: focal loss (α 0.25, γ 2), costs and weights class 2 / L1 5 / GIoU 2 (loss class weight 1), with auxiliary and encoder losses.
- **Inference post-processing**: top-k over sigmoid scores, giving `(boxes, scores, labels)` in encoder-input pixels. F8 maps them back to FLIR pixels with `ResizeGeometry.invert_boxes`.

The training schedule comes from `dino_vitdet_base_4scale_{12,50}ep.py` and belongs to F8. There is **no detectron2/detrex runtime dependency**: we vendor or rewrite the code, keep detrex's module structure and parameter names, and pin the commit. **Decided in the spec:**
- the input stem in front of the pyramid (FlowWM's "pooler with LayerNorm" is unreleased; ours could be LayerNorm + a linear projection from C)
- the multi-scale deformable attention implementation (detrex's CUDA op or its pure-PyTorch fallback)
- how to replace detrex's `square_pad=1024`, since our grid is not square
- whether to check numerical equivalence against a detrex install with shared weights, or only match the config line by line

**Done when:**
- unit tests cover shapes, matcher and loss on CPU
- the head overfits a tiny synthetic set (loss → ~0, predicted boxes match GT)
- every hyperparameter is listed next to the detrex config line it comes from, and every deviation from detrex or FlowWM is written down with its reason

**Depends on:** F0.

### F8 · Detector training pipeline ☐ (current focus)
A `LightningModule` and a `LightningDataModule` that train the F7 head on **features encoded on the fly** by the frozen F4 encoder (no cache): FLIR images from `FlirImageDataset` (F2) → resolution policy (R5) → frozen DINOv2-Reg + PCA → F7 head. Box targets are mapped to the encoder input frame with the same transform as the image, and ignore regions are handled as decided in the spec (e.g. dropped from targets; matched predictions not penalised). Hydra configs for `det_rgb` and `det_ir`. Trained on `train` and selected on `val` with W&B logging, using an **off-the-shelf val mAP** (e.g. torchmetrics `MeanAveragePrecision`, or `pycocotools` on `FlirSplit.to_coco_gt`; choice in the spec). The rigorous evaluator (F6) is not required. SLURM-ready with auto-requeue (TechStack §5). Whether to use image augmentation (possible now that features are not cached; must be identical for both modalities) is decided in the spec. An optional Optuna sweep config can be added.
Steps: (1) overfit a few hundred real `train` frames end-to-end to prove the pipeline; (2) full RGB and IR runs on ctcomp3 (user step); (3) record val AP/AP50, per-class AP, training time and encoder share of step time in a results note.
**Done when:** two frozen checkpoints `det_rgb` and `det_ir` exist with their val AP/AP50 logged in W&B and in the spec, and both are clearly above zero on person and car (R3, R10). If encoding is the bottleneck, the note says so and F5 is revisited.
**Depends on:** F2, F4 ✓, F7 ✓, and R5 ✓ (`native`). *(Before 2026-10-15: F5, F6, F7.)* The dataset is on ctcomp3 (copied during F4).
**Budget (F4, updated by F7 2026-10-26):** plan for an **NVIDIA L4 (24 GB) as the worst case**. At `native` (448×560, 32×40 tokens), fp16, SDPA encoder + compiled MSDA: the encoder runs at ≈ 89 frames/s end to end on real FLIR with 8 DataLoader workers (loader not the limit), the head trains at ≈ 13 img/s (B = 8, 9.1 GiB). A sequential estimate is **≈ 15 min per epoch per modality, ≈ 87 % of it head** (F7 bench used 5 boxes per image; real FLIR has more). With plain attention the encoder alone is ≈ 5 min per epoch. Use SDPA (decide in the spec, since it is not bit-exact) and compiled MSDA. Expect a few minutes of startup per job on the cluster (R11).

### F6 · Detection evaluator ⏸ deferred (2026-10-15)
> **Deferred:** not needed to get a working detector (F8 uses an off-the-shelf val mAP). Needed before F9.

`irwm.eval`, built as pure functions with tests on synthetic boxes:
- COCO-eval wrapper (AP, AP50, APS/M/L, per-class AP) over the 6 F2 classes, on the GT from `FlirSplit.to_coco_gt` (ignore regions as crowd GT, APS/M/L on 640×512-normalised areas). Adds `pycocotools`.
- per-horizon breakdown
- bootstrap CIs over clips (R2)

Results are written as JSON.
**Depends on:** F0 (independent of the data).

### F9 · Oracle evaluation → H1 ⏸ deferred (2026-10-15)
> **Deferred:** comes after F8 and F6. Video-test features are encoded on the fly unless F5 has been built by then.

Evaluate `det_rgb` on RGB video-test features and `det_ir` on IR video-test features, on the primary (AP, AP50) and secondary metrics, and also per static `val` split. Produces **Result Table 1: does IR beat RGB on real DINOv2 features?** It also includes a check on a `val` subset using full 3072-d features (encoded on the fly) to show how much the Cityscapes PCA costs each modality (R7).
**Depends on:** F6, F8.

---

## Phase 3: Zero-shot world model (H2, H3)

### F10 · DINO-Foresight adapter ☐
`irwm.worldmodel`: load the checkpoint, and `rollout(context_feats, n_steps)` → `[n_steps, H', W', 1152]`, autoregressive exactly as in the reference code (the prediction overwrites the masked last slot, then the window shifts, all in PCA space) and deterministic. Collect each step's prediction directly rather than slicing the `sample_unroll` window, whose last slot is a duplicate. Equivalence test: equal to `sample_unroll` in fp32 (F1: |Δ| = 0). Inference in fp16 autocast. As a **sanity check**, run it on a handful of FLIR RGB clips: error should be lowest at step 1, grow with the horizon, and beat copy-last.
**Depends on:** F1 (gate passed), F4 (F5 if it has been built by then).

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
| R1 | ~~FlowWM checkpoint not public~~. **Closed (F1):** DINO-Foresight checkpoints load cleanly, and the weights are MIT. | F1 ✓ |
| R2 | The FLIR video test split is small (3749 frames per modality in 8 sequences, F2), so there may be few clips and wide confidence intervals. **F3:** 330 clips (304 evaluable) in 8 sequences / 11 segments; clips overlap, so bootstrap by sequence or segment, not by clip. Bike is the weakest class: 113 thermal / 446 RGB test boxes. | F3, F6 |
| R3 | FLIR has few *large* objects. The primary metric is now AP/AP50 over all sizes, but small objects on a 14-px patch grid may give low absolute AP. **F2:** 89 % (RGB) and 86 % (thermal) of kept test boxes are S (< 32² on the 640×512-normalised area); light and sign are ≥ 93 % S in every split. | F2 stats, F9 |
| R4 | 1→3 channel replication for thermal is a design choice that may itself hurt DINOv2 IR features. **F4:** ImageNet normalisation is kept. FLIR thermal pixels (150 train frames) have mean 0.516, std 0.236 → normalised channel means 0.13/0.27/0.49, std ≈ 1.04, no further from DINOv2's input distribution than FLIR RGB (0.35/0.48/0.61). The remaining risk is content (no colour, thermal contrast), which H1 measures. | F4 ✓ / F9 |
| R5 | **Aspect ratio. DECIDED 2026-10-25 (user): (c) native** (centre crop to 5:4, then resize to 448×560, 32×40 tokens), for the 2.1× faster encoding at a 0.5–1.9 % higher forecast ratio (F4 probe below). `encoder.policy` defaults to `native`; benchmark configs default to the 448×560 / 32×40 grid. TechStack §3.1 amended 2026-10-26. Original text: (it fixes the token grid of both detectors, and of the cache if F5 is built, so changing it later means retraining and re-encoding). DINO-Foresight was trained at 448×896 (2:1 → 32×64 tokens of 14 px). FLIR thermal is 640×512 (5:4) and RGB varies. Options, with px of the original FLIR frame per token: **(a)** stretch to 448×896: full frame, trained size, ~1.6× distortion, 10×16 px/token. **(b)** crop the central 2:1 band, then resize to 448×896: no distortion, trained size, but drops 96 rows top and 96 bottom (37.5 % of the frame; boxes lost or truncated there), 10×10 px/token. **(c)** native: centre crop to 5:4, then 448×560 (32×40 tokens; F4): no distortion, thermal never cropped, RGB loses 4–17 % at the edges (4.4 %/6.25 % on test), untrained 32×40 grid (37.5 % fewer tokens, positional embeddings interpolated 64→40), 16×16 px/token, **2.1× faster encoding**. **(d)** two overlapping 2:1 windows (top and bottom) merged: full frame, no distortion, trained grid, but 2× cache and compute (reference `sliding_window_inference` code exists). Not tested. **F1 probe** (one Cityscapes demo clip center-cropped to 5:4; 1-step; ratio = model SmoothL1 ÷ copy-last SmoothL1 in PCA space, lower is better): uncropped 2:1 reference 0.607; (a) 0.662; (b) 0.551; (c) 0.648. **Reading:** every policy beats copy-last by 34–45 %, so neither the stretch nor the 32×40 grid breaks the model. (a) and (c) are tied within noise (one clip). (b)'s win is likely because the central band is easier: it even beats the uncropped reference, and it pays with the lost field of view. The metric measures forecasting only, not small-object detection (R3). **Before deciding:** repeat the probe on a few FLIR RGB/IR clips (no labels needed), and weigh field of view and px/token against detection needs. Provisional preference: (a). **F4 FLIR probe** (all 330 clips of `flir_test_fs5_c4f6_cs10`, 8 sequences, L4; mean ratio, lower is better): RGB crop 0.728 / stretch 0.740 / native 0.754; thermal crop 0.799 / stretch 0.808 / native 0.812. Every policy beats copy-last on every clip (worst 0.958). Policy differences are **1–3 %**, not the 17 % of F1's single clip, and smaller than the spread between sequences. By sequence: crop beats stretch in 6/8 RGB and 4/8 thermal; stretch beats native in 8/8 RGB and 6/8 thermal. Per clip, native − stretch has median +0.014 (RGB) and +0.003 (thermal). **Trade-off for the decision:** native encodes 2.1× faster (≈ 5 vs 10 min per epoch, F8 budget) for a 0.5–1.9 % higher ratio; crop is marginally best but drops 37.5 % of every thermal frame and is no faster. The probe measures forecasting, not small-object detection (R3). | before F8 |
| R6 | Phase 4 needs unlabeled IR video that is disjoint from test. | F14 |
| R7 | **PCA basis fit on Cityscapes RGB.** It may truncate IR features more than RGB, which would confound H1 and H2. Caching full 3072-d features would cost about 390 GB, against about 145 GB for the PCA cache (30,787 frames, F2). **Decided (F1):** cache PCA features only, and measure the PCA loss per modality on a subset with full features encoded on the fly. In-domain baseline: 3.1 % relative reconstruction error, 90.5 % explained variance on Cityscapes. | F9, F13 |
| R8 | DINO-Foresight is **deterministic** (SmoothL1 regression), so predictions may be blurred toward the mean. That is part of what H3 measures, but it also means no sampling and no best-of-N. | F13 |
| R10 | **Annotation asymmetry between modalities (F2).** On the time-synced test frames, thermal has far fewer boxes than RGB for sign (ratio 0.33), light (0.38) and bike (0.25), while person and car are ≈ 1. Train/val ratios are much closer to 1. So per-class AP for sign/light/bike compares different labelling densities, not just modalities, and an RGB detector is credited for objects the thermal GT never labels. H1/H2 conclusions should be checked on person + car alone, and per-class results reported. | F6, F9, F12 |
| R11 | **Slow tests and startup on the cluster (F4). Cause: BeeGFS. Mitigation ready.** Read-only `data` tests take minutes on ctcomp3: FLIR readers 174 s vs 9 s on ctdesk236 (same code and data), encoder equivalence 92 s. **Staging test** (job 5525147, hpc-node8, 8 CPUs, commit `3a0916d`, `slurm/stage_data_test.sbatch`): the readers' `FlirSplit.load` (6 × `coco.json`, ≈ 260 MB) took **176 s from BeeGFS vs 5.7 s from node-local `$LOCAL_SCRATCH`** (whole run 201 s → 29 s), so the shared filesystem is the cause, not the CPU count. Copying all of FLIR (12 GB, 46 450 files, 6 parallel `tar` streams) to scratch takes **276 s (44 MB/s)**. That is longer than the slow load, so staging does not pay off for a one-off metadata read. It does pay off for anything that reads images every epoch (F8: full read from BeeGFS at that rate is ≥ 4.6 min per pass, and probably much slower with random access). Separate cost: a cold `import torch` from the BeeGFS `.venv` takes **40 s**, and about 20 s of the 29 s scratch run is still imports. That cost is per process and stays even with staged data. **Use:** `source slurm/stage_data.sh` after `slurm/env.sh` at job start. It copies to `$LOCAL_SCRATCH/irwm_data/` (wiped when the job ends), checks the file count and sets `IRWM_DATA_ROOT`. Each requeue pays the copy again. **Still open:** when to stage by default (F8), copying only the annotation JSONs for metadata-only jobs, and moving the `.venv` (or `$IRWM_CKPT_ROOT` weights) to scratch. **Encoder benchmark (F7, jobs 5525665/5525666, hpc-gpu5, 2026-10-25):** staging via `IRWM_STAGE_DATA=1` copied 46,450 files (12.2 GB) in **283 s (42 MB/s)**. Loader-only with 8 workers: RGB 349 → 478 fps, thermal 952 → 1537 fps (BeeGFS → staged, page cache). End to end with the SDPA encoder: ≈ 89 fps on both, encode share ≥ 0.998. **On an L4, 8 workers on BeeGFS already outpace the encoder 4–11×, so staging gives no end-to-end gain**; it pays only for faster GPUs, fewer workers, or the metadata load at job start (276–283 s copy vs ≈ 170 s saved). | F8 |
| R9 | Double domain gap: the model was trained on Cityscapes (German cities, daytime RGB). FLIR RGB is already out of domain, so the RGB sanity check matters before any IR conclusion is drawn. | F10 |
