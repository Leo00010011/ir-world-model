# F4 · DINOv2-Reg encoder wrapper: Validation

## Code Correctness (`tests/test_encoder.py`, CPU, no weights, no dataset)

### Group 1 — Preprocess (FR1, FR2)
- [ ] **Output shapes:** synthetic `uint8` thermal `[512, 640]` and RGB `[1024, 1224, 3]` and `[768, 1024, 3]`, under each policy → tensor `float32` `[3, 448, 896]` (`stretch`, `crop`) or `[3, 448, 560]` (`native`).
- [ ] **Thermal replication:** thermal input → the 3 output channels differ only by the per-channel ImageNet normalisation. Undoing `Normalize` gives 3 identical channels (|Δ| = 0).
- [ ] **Normalisation:** a constant RGB image of value 255 under `stretch` → channel `c` equals `(1 − mean[c]) / std[c]` to `atol=1e-6`.
- [ ] **Stretch equals the reference transform:** the 5 demo PNGs as `uint8` arrays → `Preprocess("stretch", "rgb")` stacked equals `load_demo_frames((448, 896))[0]`, |Δ| = 0.
- [ ] **Crop equals the F1 probe transform:** the same frames → `Preprocess("crop", "rgb")` equals `load_demo_frames((448, 896), (2, 1))[0]`, |Δ| = 0.
- [ ] **Invalid input:** thermal given `[H, W, 3]`, RGB given `[H, W]`, a `float32` array, or policy `"windows"` → `ValueError`.
- [ ] **Picklable:** `pickle.loads(pickle.dumps(Preprocess("native", "thermal")))` gives the same output on a fixed input (|Δ| = 0).

### Group 2 — ResizeGeometry (FR3)
- [ ] **Stretch scaling:** a thermal 640×512 frame under `stretch` and box `[64, 51.2, 64, 51.2]` → `[89.6, 44.8, 89.6, 44.8]` (x ×1.4, y ×0.875), `keep=True`.
- [ ] **Crop offset:** a thermal 640×512 frame under `crop` gives crop box `(0, 96, 640, 320)`. Box `[100, 200, 50, 40]` → `[140, 145.6, 70, 56]` (offset −96 on y, then ×1.4 both axes).
- [ ] **Crop clipping and drop:** under `crop` on 640×512, box `[10, 80, 20, 30]` (y 80–110, partly above the band) → clipped to y 96–110 → `[14, 0, 28, 19.6]`, `keep=True`. Box `[10, 10, 20, 30]` (entirely above) → `keep=False` and a zero row.
- [ ] **Native crop boxes:** under `native` → `crop_box` equals `(0, 0, 640, 512)` for 640×512, `(0, 22, 1224, 979)` for 1224×1024, `(32, 0, 960, 768)` for 1024×768, `(0, 80, 1800, 1440)` for 1800×1600, `(64, 0, 1920, 1536)` for 2048×1536, `(60, 0, 600, 480)` for 720×480, and the full frame for 1280×1024. `out_size == (560, 448)` in every case.
- [ ] **Native thermal is untouched:** a thermal 640×512 frame → `Preprocess("native", "thermal")` equals `Resize((448, 560))` → `ToTensor` → `Normalize` applied to the uncropped `convert("RGB")` image (|Δ| = 0). Box `[64, 51.2, 64, 51.2]` → `[56, 44.8, 56, 44.8]` (×0.875 on both axes).
- [ ] **Native on RGB 4:3:** a 1024×768 frame, box `[0, 0, 1024, 768]` → clipped to x 32–992 → `[0, 0, 560, 448]`, `keep=True`. Box `[0, 0, 20, 20]` (inside the cropped left margin) → `keep=False`.
- [ ] **Native aspect preserved:** for every size in the crop-box item above, `crop_w / crop_h` is within 0.1 % of 1.25.
- [ ] **Round trip:** 100 random boxes inside the crop box, for each policy and source size → `invert_boxes(apply_boxes(b)[0])` equals `b` to `atol=1e-3`.
- [ ] **Empty input:** `[0, 4]` → outputs of shape `[0, 4]` and `[0]`.

### Group 3 — Encoder with stub backbone (FR4–FR7)
- [ ] **Call contract:** `encode_full` on `[2, 3, 448, 896]` → the stub records `n == [2, 5, 8, 11]`, `reshape is False`, `norm is True`. Output `[2, 32, 64, 3072]` `float32`.
- [ ] **Block order:** the stub's block `k` returns a constant `k+1` → `encode_full` channels `[0:768]` = 3, `[768:1536]` = 6, `[1536:2304]` = 9, `[2304:3072]` = 12.
- [ ] **Row-major token layout:** the stub sets token `i`'s value to `i` → `encode_full(x)[0, r, c, 0] == r·64 + c` for all `r, c`.
- [ ] **PCA path:** `encode_pca(x)` equals `pca.transform(encode_full(x))` (|Δ| ≤ 1e-5, CPU fp32). Shape `[2, 32, 64, 1152]`. On `[1, 3, 448, 560]` → `[1, 32, 40, 1152]`.
- [ ] **Inverse:** `pca_inverse(encode_pca(x))` has shape `[2, 32, 64, 3072]` and equals `pca.inverse(pca.transform(f))` to `atol=1e-5`.
- [ ] **Frozen:** after construction, `all(not p.requires_grad for p in enc.parameters())` and `enc.training is False`. After `enc.train()` and `enc.train(True)`, still `enc.training is False` and `enc.backbone.training is False`. `enc.train()` returns `enc`.
- [ ] **No graph, usable downstream:** `y = encode_pca(x.requires_grad_())` → `y.requires_grad is False`. `nn.Linear(1152, 4)(y).sum().backward()` runs without error, which proves the output is not an inference tensor.
- [ ] **Input validation:** `[3, 448, 896]` (3-D), `[1, 1, 448, 896]` (C = 1), and `[1, 3, 450, 896]` (not ÷14) → `ValueError`. `precision="bf16"` → `ValueError`.
- [ ] **CPU fp16 fallback:** `precision="fp16"` on a CPU tensor → exactly one `UserWarning` across two calls, and output equal to the `precision="fp32"` encoder's (|Δ| = 0).
- [ ] **fp32 ignores an outer autocast:** on CPU, `with torch.autocast("cpu", dtype=torch.bfloat16): encode_pca(x)` on a `precision="fp32"` encoder → equals the result without the outer context (|Δ| = 0).
- [ ] **`from_assets` without weights:** with `irwm_env` pointing at empty tmp roots → `FileNotFoundError` whose message names `download.py`.

### Group 4 — Config, tags and refactor (FR9, FR10)
- [ ] **Encoder tags:** `encoder_tag("stretch") == "dinov2_vitb14reg_448x896_l3-6-9-12_pca1152"`, `encoder_tag("crop") == "dinov2_vitb14reg_448x896crop_l3-6-9-12_pca1152"`, `encoder_tag("native") == "dinov2_vitb14reg_448x560_l3-6-9-12_pca1152"`.
- [ ] **Hydra group:** composing `irwm_base` (Hydra `compose` with `pkg://irwm.conf`) → `cfg.encoder == {"policy": "stretch", "precision": "fp16", "hub_version": "dinof_release"}`.
- [ ] **Moved helpers:** `from irwm.worldmodel.reference import one_step, pca_metrics` works. `projects/dinof_audit/audit.py` contains no `def one_step` or `def pca_metrics`. `pca_metrics` on known tensors (pred = gt + 1, beta 1) → `smoothl1 == 0.5`, `mse == 1.0`.
- [ ] **Lightweight imports:** `test_no_heavy_toplevel_imports_in_lib` (in `tests/test_dinof_audit.py`) still passes. The full default suite `uv run pytest` passes (≥ 145 + new tests, 0 failures).

### Group 5 — Equivalence with real weights (`@pytest.mark.data`, run on ctcomp3 GPU)
All three tests use the same `[5, 3, 448, 896]` demo tensor from `load_demo_frames((448, 896))`, as one unsplit batch, on the same GPU (FR8 table).
- [ ] **Test A, fp32 full features:** our `precision="fp32"` `encode_full` vs reference `Dino_f.extract_features` with no autocast, reshaped to `[5, 32, 64, 3072]` → max |Δ| = 0.
- [ ] **Test A, fp32 PCA features:** our `precision="fp32"` `encode_pca` vs reference `pca_transform(extract_features(...))` with no autocast → max |Δ| = 0.
- [ ] **Test B, fp16 full and PCA features:** our `precision="fp16"` `encode_full`/`encode_pca` vs the same reference calls inside `torch.autocast("cuda", torch.float16)` → max |Δ| = 0. If not 0, max |Δ| ≤ 1e-2 **and** `results.md` names the differing op (user decision).
- [ ] **Test C, fp16 drift (information only):** our fp16 vs our fp32 `encode_pca` → max/mean |Δ| and the mean token cosine are printed and copied to `results.md`. No threshold.
- [ ] **Native grid end-to-end:** a thermal 640×512 synthetic frame through `Preprocess("native", "thermal")` → `encode_pca` → `[1, 32, 40, 1152]`, all finite. This exercises DINOv2-Reg's own positional-embedding interpolation on a real backbone.
- [ ] **Assets offline:** `DinoV2RegEncoder.from_assets()` succeeds inside the Step 9 jobs on a ctcomp3 compute node. The hub load uses `source="local"` with `pretrained=False`, and the weights come from `dinov2_weights_path()`, so no download is attempted. `enc.pca.sha256` is non-empty.
- [ ] **Register variant:** the loaded backbone has `num_register_tokens == 4` and `embed_dim == 768`. A 448×896 forward of `get_intermediate_layers` gives 2048 tokens per block.

## Data Validity (`r5_probe.json`, `throughput.json`, pytest output from ctcomp3)

- [ ] **Encoder drift, Test C:** `results.md` records the Test C numbers. Expected mean cosine ≥ 0.999. This is informational and not a gate, because fp16 is the reference behaviour.
- [ ] **Probe completeness:** `r5_probe.json` holds 330 clips × 2 modalities × 3 policies = **1980** per-clip rows, all with finite metrics, and `n_clips == 330` in each of the 6 summaries.
- [ ] **Probe sanity:** for every (modality, policy), the model beats copy-last on average (mean ratio < 1.0). If any is ≥ 1.0, report it prominently in `results.md`. It is a finding for R5/R9, not a bug to hide, but check first that the encoder input really is the `Preprocess` output for that modality.
- [ ] **Probe grid:** the per-clip `x` grid is `32×64` for `stretch`/`crop` and `32×40` for `native` (recorded in the JSON).
- [ ] **Throughput, GPU-only:** `throughput.json` has frames/s for 2 sizes × 2 precisions × 3 batch sizes = 12 entries (`"oom"` allowed only for fp32 at B = 64). Expected: fp16 frames/s > fp32 at B = 16 for both sizes.
- [ ] **Throughput, end-to-end:** frames/s, encode share and projected epoch seconds for RGB train (10318 frames) and thermal train (10742 frames) are recorded in `results.md`. If the encode share is < 50 %, the note says that data loading, not encoding, is the bottleneck for F8.
- [ ] **Results note:** `results.md` contains all of the following:
  - the FR8 numbers;
  - both throughput tables;
  - the probe table (mean, median and summed ratio per modality × policy);
  - the sequence win counts;
  - the comparison with the F1 Cityscapes probe;
  - an explicit sentence that `stretch` equals the reference transform and `crop`/`native` are deviations (both centre crops; `native` never crops thermal);
  - the positional-embedding interpolation sites (DINOv2-Reg hub `file:line`; DINO-Foresight `high_res_adapt`);
  - the statement that R5 is still the user's decision.

## Data Architecture Integrity
- [ ] **Submodule untouched:** `git -C third_party/DINO-Foresight status --porcelain` is empty after all runs.
- [ ] **Determinism:** `encode_pca` run twice on the same demo batch (GPU, fp16) → |Δ| = 0.
- [ ] **Provenance:** both JSONs record the git commit (clean tree), the resolved config, the GPU name and the `encoder_tag`s. `r5_probe.json` also records the clip-set SHA-256, which equals the one pinned in `tests/test_clip_index.py`.
- [ ] **Allowed paths only:** the runs write only under `$IRWM_OUTPUT_ROOT/feature_extraction/` and W&B. Nothing is written under `$IRWM_DATA_ROOT`, `$IRWM_CKPT_ROOT` or `$IRWM_CACHE_ROOT`. A grep of `lib/irwm/features/` and `projects/feature_extraction/` finds no absolute paths (TechStack §6).
- [ ] **Naming:** the new code, configs and `results.md` call the backbone "DINOv2-Reg" / `vitb14_reg`, never plain ViT-B/14 (`grep -n "vitb14[^_]"` on the new files finds nothing).
