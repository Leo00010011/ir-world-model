# F7 · DINO-DETR head port: Validation

## Code Correctness (`tests/test_detection_head.py`)

Groups 1–6 run on CPU with no data or weights (ctdesk236, `uv run pytest tests/test_detection_head.py`), in < 60 s in total. Groups 7–8 are `@pytest.mark.gpu` and run on ctcomp3 (plan.md Step 9).

### Group 1 — Submodule and build (CPU)
- [x] `git submodule status third_party/detrex` → starts with `e244e6c3da3e84566728c52c21fb061d23ce0e2f`, no `+`/`-`/`U` prefix; `git -C third_party/detrex submodule status detectron2` → starts with `70f454304e1a38378200459dd2dbca0f0f4a5ab4` (FR1).
- [x] `git status --porcelain` is empty in `third_party/detrex` and `third_party/detrex/detectron2` after the build (FR1).
- [x] `import detectron2, detrex; from detrex import _C` succeeds; `detectron2.__version__ == "0.6"`; `import pkg_resources` succeeds (FR2).
- [x] `uv lock --check` passes without building detectron2/detrex: `dependency-metadata` is declared for both (FR2; checked once by the implementer, reviewed by eye).

### Group 2 — detrex adapter (CPU)
- [x] `detrex_root()` → `repo_root() / "third_party" / "detrex"`; `DETREX_COMMIT`/`DETECTRON2_COMMIT` equal the submodule SHAs of Group 1 (FR3).
- [x] `detrex_root()` with `repo_root` monkeypatched to an empty tmp dir → `FileNotFoundError` whose message contains `git submodule update --init --recursive` (FR3).
- [x] Calling `ensure_detrex_projects()` twice → `sys.path.count(str(detrex_root())) == 1`, and the entry is not at index 0 (FR3).
- [x] `importlib.import_module("projects.dino.modeling").__file__` is under `third_party/detrex/projects/dino/modeling/`; `detrex.__file__` is under `third_party/detrex/detrex/` (editable install) (FR3).
- [x] `load_dino_vitdet_config().model.num_queries == 900` and `.backbone.square_pad == 1024`, read from the file and not from irwm code (FR3).
- [x] With `torch.cuda.is_available` monkeypatched to return `False`, `cuda_ops_available()` → `False` without raising (FR3).
- [x] The existing `test_lib_does_not_import_projects` still passes (plan.md, adapter decision).

### Group 3 — Feature stem (CPU)
- [x] `FeatureStem()(randn(2, 1152, 32, 64))["last_feat"].shape == (2, 768, 32, 64)`; with `in_channels=3072` → `randn(1, 3072, 32, 40)` gives `(1, 768, 32, 40)` (FR4).
- [x] `FeatureStem().output_shape() == {"last_feat": ShapeSpec(channels=768, stride=16)}` (FR4, D5).
- [x] Equal to the manual formula: `F.linear(F.layer_norm(x.permute(0,2,3,1), (1152,), w_n, b_n, 1e-6), W, b).permute(0,3,1,2)` → |Δ| = 0 (FR4).
- [x] Input `[2, 1024, 32, 64]` or `[1152, 32, 64]` → `ValueError` naming the expected `1152` / 4-D shape (FR4).

### Group 4 — Config, structure equivalence and Hydra group (CPU)
- [x] `build_dino_config()` → `_target_ is FeatureDINO`, `backbone.net._target_ is FeatureStem` with `in_channels=1152, embed_dim=768, out_feature="last_feat"`, `num_classes == 6`, `device == "cpu"` (FR6).
- [x] Every row marked "yes" in the plan.md hyperparameter table is asserted on the built config. This covers: SFP `out_channels` 256, `scale_factors` (2.0, 1.0, 0.5), `norm` "LN"; neck `in_features` p3–p6, `num_outs` 4, `kernel_size` 1; encoder/decoder `num_layers` 6, `embed_dim` 256, `num_heads` 8, `feedforward_dim` 2048, dropouts 0.0; `num_feature_levels` 4; `num_queries` 900; `dn_number` 100; `label_noise_ratio` 0.5; `box_noise_scale` 1.0; matcher costs 2.0/5.0/2.0; base weights 1/5.0/2.0; focal α 0.25 γ 2.0; `two_stage_binary_cls` False (FR6).
- [x] With no overrides, the recomputed `criterion.weight_dict` equals the one in detrex's loaded config exactly (D8).
- [x] **Structure match:** consider `instantiate(load_dino_vitdet_config().model)` with `num_classes=6, device="cpu"` (stock, ViT-B) and `DINODetector().detr`. Outside `backbone.net.*`, their `state_dict` key sets are identical and every shape is equal. Our `backbone.net.*` keys are exactly `{norm.weight [1152], norm.bias [1152], proj.weight [768, 1152], proj.bias [768]}` (user decision: equivalence check).
- [x] Parameter count: `DINODetector()` → **23,001,286**; `DINODetector(in_channels=3072)` → **24,479,686** (spec-time probe; FR7).
- [x] `overrides={"transformer.decoder.num_layers": 3}` → the weight dict has the suffixes `_enc`, `_0`, `_1` and no `_2`; `overrides={"no.such.key": 1}` → `KeyError`; `overrides={"num_classes": 3}` and `{"backbone.net.embed_dim": 512}` → `ValueError` (FR6).
- [x] `initialize_config_dir(lib/irwm/conf)` + `compose(config_name="irwm_base")` (the pattern in `tests/test_repo_scaffold.py`) → `cfg.detector == {in_channels: 1152, stem_dim: 768, num_classes: 6, overrides: {}}`, equal to the `DINODetector.__init__` defaults read with `inspect.signature`. `DINODetector(**OmegaConf.to_container(cfg.detector))` builds (FR8).

### Group 5 — Wrapper on CPU: eval, shapes, errors
- [x] Full default config, `eval()`, `feats = randn(1, 1152, 32, 64)` → one dict. `boxes` `float32 [K, 4]` with K ≤ 300; all `0 ≤ x0 ≤ x1 ≤ 896`, `0 ≤ y0 ≤ y1 ≤ 448`; `scores` in [0, 1]; `labels` `int64` in 0..5 (FR7).
- [x] Same with `feats = randn(1, 1152, 32, 40)` and default `image_size` → boxes within 560×448. With `image_size=(448, 560)` given explicitly → |Δ| = 0 against the default (FR7).
- [x] Pyramid (forward hook on `detr.neck`): 32×64 input → 4 maps `[1, 256, 64, 128]`, `[1, 256, 32, 64]`, `[1, 256, 16, 32]`, `[1, 256, 8, 16]`; 32×40 → `64×80, 32×40, 16×20, 8×10` (plan.md, pyramid sizes).
- [x] Two eval calls on the same input → |Δ| = 0 for boxes, scores and labels (FR7).
- [x] Reduced config (plan.md Step 7) with batch 3 at 8×16 → a list of 3 dicts (FR7).
- [x] Train mode on CPU with valid targets → `RuntimeError` whose message contains `.cuda()`; train mode with `targets=None` or 2 targets for B = 3 → `ValueError` (FR7).
- [x] Default config (900 queries) on a 4×8 grid (170 tokens) → `ValueError` naming 170 and 900 (FR7).
- [x] `FeatureDINO.preprocess_image` on 2 inputs `[1152, 8, 16]` → `ImageList` with `tensor.shape == (2, 1152, 8, 16)`, values identical to the inputs (no normalisation) (FR5).
- [x] `FeatureDINO` defines only `preprocess_image`: `set(vars(FeatureDINO)) − set(vars(object)) − {"__module__", "__doc__", "__qualname__"} == {"preprocess_image"}` (FR5).

### Group 6 — Matcher and set criterion (CPU)
- [x] `HungarianMatcher` from the built config, on `pred_logits [1, 10, 6]` (−10 everywhere, +10 at `(q, label)` for queries 7, 2, 5) and `pred_boxes` with queries 7, 2, 5 equal to GT boxes 0, 1, 2 → indices `([2, 5, 7], [1, 2, 0])` (sorted by query) (roadmap: matcher on CPU).
- [x] Same matcher with a target with 0 boxes → `(tensor([], int64), tensor([], int64))` (empty-image case).
- [x] `TwoStageCriterion.forward` (the parent of `DINOCriterion`, which uses no `.cuda()`) on the perfect outputs above, without aux/enc → `loss_bbox == 0` (atol 1e-6), `loss_giou == mean(1e-6 / (A_i + 1e-6))` over the GT areas `A_i` (rtol 1e-3; detrex's `box_iou` divides by `union + 1e-6`, `detrex/layers/box_ops.py:81`, so a perfect box keeps that residue; amended 2026-10-23, user decision), `loss_class < 1e-3`. With the matched logits moved to a wrong class → `loss_class` at least 10× larger (roadmap: loss on CPU).

### Group 7 — CUDA ops, training losses, fp16 (`@pytest.mark.gpu`, ctcomp3 L4)
- [x] `cuda_ops_available()` → `True` (FR2, FR3).
- [x] Random CUDA inputs for 4 levels: `MultiScaleDeformableAttnFunction.apply` (the CUDA op) vs `multi_scale_deformable_attn_pytorch`. Forward close to `rtol=1e-4, atol=1e-5`. Grads w.r.t. `value`, `sampling_locations` and `attention_weights` close to `rtol=1e-3, atol=1e-4` (user decision: CUDA op).
- [x] Full default config, `train()`, batch 2 at 32×64 on CUDA, image 0 with 3 boxes and image 1 with 0 boxes → exactly 39 loss keys: `{loss_class, loss_bbox, loss_giou}` × `{"", _0…_4, _enc, _dn, _dn_0…_dn_4}`. All finite. `sum(losses.values()).backward()` gives a non-zero gradient on `detr.backbone.net.proj.weight` (FR7).
- [x] The same with both images empty → 39 keys, all finite, and every `*_dn*` loss is 0 (`dn_criterion.py:99-104`) (FR7).
- [x] The same as the first item under `torch.autocast("cuda", dtype=torch.float16)` → all losses finite; eval boxes finite (plan.md, MSDA fp16 handling).

### Group 8 — Overfit (`@pytest.mark.gpu`, ctcomp3 L4)
- [x] The synthetic set of plan.md Step 7 (4 images, 11 boxes), with the full default config, fp32, batch 4, AdamW lr 1e-4, wd 1e-4 and clip 0.1. Within ≤ 1500 iterations: in eval mode, every GT box has a detection with the same label, IoU ≥ 0.8 and score ≥ 0.5, **and** the total weighted loss is ≤ 10 % of its step-0 value. The test prints iterations, loss ratio, min IoU, min score, seconds and peak GiB (user decision: overfit bar; roadmap Done when).

## Data Validity (`results.md`, from the user's ctcomp3 runs)
- [x] **CUDA build on ctcomp3:** `setup_env.sh` built both packages with `FORCE_CUDA=1`. `nvcc`/`gcc` versions and build seconds are recorded. If it fails, F7 is not done: report the compiler error instead of falling back (user decision: speed).
- [x] **GPU suite:** `pytest -m gpu` on an L4 → all Group 7–8 tests pass. The output summary is in `results.md`.
- [x] **Head cost (FR10):** `head_bench.json` exists, with 2 modes (train, infer) × 1 grid (32×40) × 4 batch sizes × 2 precisions = 16 entries *(amended 2026-10-26, user decision: was 2 grids × train only)*, each either timed or `"oom": true`. `results.md` tabulates the median step seconds, images/s and peak GiB. It also compares images/s with F4's encoder throughput (17 frames/s at 448×896, 36 at 448×560) to state whether the head or the encoder bounds F8's step.
- [x] **Deviation list complete:** `results.md` confirms that D1–D8 in plan.md are the only differences from detrex, or adds any found during implementation with its reason (roadmap Done when).
- [x] **Results note complete (FR11):** `results.md` holds every FR11 item: the build, the GPU test summary, the overfit numbers, the bench table, the parameter count (expected 23,001,286) and the deviation confirmation.

## Data Architecture Integrity
- [x] **detrex unmodified:** Group 1's clean `git status` holds after every build. No patch files exist (TechStack §2). *(2026-10-26, user decision: the CUDA-build check on ctcomp3 was not captured; accepted on the clean ctdesk236 trees, detrex's `.gitignore` covering `*.so`/`build/`/`version.py`, and the unprefixed `e244e6c` pin; results.md §8.)*
- [x] **No network at runtime:** building `DINODetector` and running Groups 2–6 needs no network. `LazyConfig.load` reads only `third_party/detrex` (checked once with networking disabled, or by reviewing that nothing downloads; reviewed by eye).
- [x] **No hard-coded paths:** `detrex_root()` derives from `repo_root()`. `head_bench.py` writes only under the Hydra run dir (`$IRWM_OUTPUT_ROOT`) (TechStack §6).
- [x] **Provenance:** `head_bench.json` contains `git_info()` (commit, dirty flag), the GPU name, `cuda_ops_available()` and the resolved config (TechStack §7).
- [x] **Default suite unaffected:** `uv run pytest` (default `-m 'not data and not gpu'`) passes on ctdesk236, with the gpu tests deselected and not failing (FR9).
