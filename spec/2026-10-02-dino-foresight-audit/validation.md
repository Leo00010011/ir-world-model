# F1 · DINO-Foresight audit: Validation

## Code Correctness (`tests/test_dinof_audit.py`, CPU, no weights)

### Group 1 — Assets and manifests
- [ ] `dinof_dir()` == `$IRWM_CKPT_ROOT/dino_foresight/9174a31e`; `dinov2_code_dir("dinof_release")` == `$IRWM_CKPT_ROOT/torch_hub/dinov2@85a24602`; `dinov2_weights_path().name == "dinov2_vitb14_reg4_pretrain.pth"`.
- [ ] `DINOV2_HUB_PINNED in DINOV2_HUB_COMMITS`; all shas are 40 lowercase hex chars; `DINOF_FILES` has the 6 weight files.
- [ ] `sha256_file` on a tmp file equals `hashlib.sha256(bytes).hexdigest()`. Files larger than one 8 MiB chunk hash correctly.
- [ ] `write_manifest` then `verify_manifest` → `[]`. After one byte of a file is changed → `[that file]`. After a file is deleted → `[that file]`.

### Group 2 — `PCABasis`
- [ ] For synthetic `mean, std>0, pca_mean, components` (orthonormal rows, 3072→1152, seeded), `transform(x)` equals the explicit formula `((x-mean)/std - pca_mean) @ C.T` (`atol=1e-5`) for `x` of shape `[2, 4, 3072]` and `[3072]`.
- [ ] `inverse(transform(x))` equals `x` when `x - mean` lies in the span (built as `inverse(z)`) to `atol=1e-4`. For full-rank synthetic `C` (k = d = 64), the round trip is exact to `atol=1e-5`.
- [ ] Wrong last dim → `ValueError` for both maps. Buffers have `requires_grad == False`. `load` round-trips a saved dict and preserves `sha256`.

### Group 3 — Reference-args handling (pure, on a fake Namespace)
- [ ] `inference_args(saved, pca_ckpt=p)` sets `eval_mode=False, eval_modality=None, head_ckpt=None, evaluate=False, vis_attn=False, pca_ckpt=str(p), high_res_adapt=False`, and does not mutate `saved`. The diff lists exactly the changed fields.
- [ ] `inference_args(saved, pca_ckpt=p, img_size=(448,560))` → `img_size == (448,560)`, `high_res_adapt is True`. With `img_size == saved.img_size` → `high_res_adapt is False`.

### Group 4 — Demo inputs
- [ ] `load_demo_frames((448, 896))` → shape `[1, 5, 3, 448, 896]`, float32, per-channel mean in `[-2.5, 2.5]` (ImageNet-normalised). Frames are in sorted filename order (017089 … 017101).
- [ ] `crop_aspect=(5, 4)` on 2048×1024 frames crops to 1280×1024 (centered: left offset 384) before resizing. `load_demo_frames((448, 560), crop_aspect=(5,4))` → `[1, 5, 3, 448, 560]`.
- [ ] `lib/irwm` imports without `timm`/`einops`/`sklearn` installed (`irwm.worldmodel.reference` imports them only inside functions), checked by an import-graph scan for top-level imports.

### Group 5 — Real weights (`@pytest.mark.data`, run on ctcomp3 after download)
- [ ] `verify_manifest(dinof_dir()) == []` and `verify_manifest(dinov2_hub_dir()) == []`.
- [ ] `checkpoint_args(highres)` has `img_size == (448, 896)`, `d_layers == [2,5,8,11]`, `sequence_length == 5`, `hidden_dim == 1152`, `layers == 12`, `masking == "simple_replace"`, `seperable_attention is True`.
- [ ] `PCABasis.load(pca_1152.pt)` matches the reference `pca_transform`/`pca_inverse_transform` on random `[2, 2048, 3072]` input (`atol=1e-5`).

## Data Validity (from `audit_results.json`, reviewed in `audit.md`)
- [ ] **Weights identity:** checkpoint `dino_v2.*` vs official weights max |Δ| = 0 (or the difference is documented). Checkpoint PCA tensors vs `.pth` max |Δ| = 0.
- [ ] **Feature recipe:** 2048 tokens × 3072 dims. `get_intermediate_layers` equals hooks+norm to |Δ| < 1e-6. Standardised demo features have |mean| < 0.5 and std in [0.5, 2] on average across dims (in-domain sanity). PCA explained variance on the demo frames is reported.
- [ ] **1-step on real frames:** model SmoothL1 < copy-last SmoothL1 and model cosine > copy-last cosine on the demo clip. If this fails, the loading or the recipe is wrong, and the gate does not pass.
- [ ] **3-step:** `sample_unroll` equals the manual PCA-space loop (|Δ| = 0, fp32). Output shapes are as specified.
- [ ] **Masked slot:** changing the target-frame pixels changes nothing (|Δ| = 0).
- [ ] **Determinism:** |Δ| = 0 across two runs. fp32 vs fp16-autocast |Δ| is reported, and the chosen inference precision is stated.
- [ ] **Hub version:** feature |Δ| between the three commits is reported. The pinned commit is justified by the 1-step SmoothL1.
- [ ] **R5:** for each of (a)/(b)/(c), the ratio model/copy-last is reported, together with the uncropped 2:1 reference ratio. The decision follows the rule "lowest ratio; stretch (a) if within 5 % of the best", with the caveat that this is a single clip.
- [ ] **Qualitative check:** the segmentation PNG from predicted features looks like a plausible street scene (reviewed by eye).

## Data Architecture Integrity
- [ ] `audit.py` runs with networking disabled after `download.py` (`HF_HUB_OFFLINE=1`; the hub dir is pre-populated). Any network attempt fails the run.
- [ ] The submodule is unmodified after the run (`git -C third_party/DINO-Foresight status --porcelain` is empty).
- [ ] Every file under `$IRWM_CKPT_ROOT/dino_foresight` and `$IRWM_CKPT_ROOT/torch_hub` is listed in a manifest with a matching sha256. `pca_1152.pt` records the sha256 of its source `.pth`.
- [ ] `audit_results.json` contains `git.commit`, the resolved config, the HF revision and the pinned hub sha, so every number traces back to its inputs.
