# F7 · DINO-DETR head port: Plan

## Context and Design Decisions

**Deviation from the constitution (user decision, 2026-10-23).** TechStack §3.2 and the earlier Roadmap F7 text say "no detectron2/detrex runtime dependency". The user reversed this: detrex and detectron2 are installed and used at runtime, as FlowWM does. Step 10 amends TechStack §3.2 accordingly. Until then, this spec is the authority on that point.

### How FlowWM does it (the route we copy)
- **FlowWM installs detectron2 + a modified detrex and calls them directly.** `Flow-World-Models/trainer/trainer.py:354-371` (@ `c150625`) runs `LazyConfig.load(model_cfg_path)` → `instantiate(cfg.model)` → `DetectionCheckpointer(model).load(...)`, with detectron2 imported lazily. `trainer.py:530-561` builds detectron2-style dicts `{"image": latents [D, H, W], "height", "width", "image_id"}` and calls `detector(batch, inference_from_backbone=True)`. That kwarg does not exist in stock detrex (`projects/dino/modeling/dino.py:151` takes only `batched_inputs`), so FlowWM runs an unreleased fork. Its config and checkpoints are unreleased too (Roadmap F7 note).
- **We reach the same behaviour without forking.** Stock `DINO.forward` (`dino.py:151-307`) works on feature tensors once `preprocess_image` stops normalising pixels. A probe during spec writing confirmed this end to end on CPU: detrex built CPU-only on ctdesk236 in 53 s (detectron2) + 31 s (detrex) with gcc 15.2 and torch 2.2.0. `dino_vitdet` was loaded with LazyConfig, `backbone.net` was swapped for a stub stem, and `num_classes=6` gave **23,001,286 parameters** (4,331,904 in stem + pyramid). One eval forward on `[1152, 32, 64]` took 1.5 s and returned 300 boxes inside 896×448.

### detrex form and loading
- **detrex is a pinned submodule, unmodified, installed editable (user decision).** This mirrors the DINO-Foresight pattern (TechStack §2). Rejected alternatives:
  - git dependencies plus copied files: that is vendoring again;
  - vendoring with patches: not FlowWM's route.
- **detrex's DINO classes and configs are not pip-installable.** They live in `projects/dino/...`, and `projects/` has no `__init__.py`. `setup.py` uses `find_packages(exclude=("configs", "tests"))`, so they are left out. `ensure_detrex_projects()` therefore puts the submodule root on `sys.path`.
  - It **appends** rather than inserts, so an installed `detrex`/`detectron2` always wins over any same-named source directory.
  - It is permanent and idempotent rather than a context manager. The namespace package `projects` recomputes its path when `sys.path` changes, and later LazyConfig loads need `projects.dino.modeling` importable.
  - Our own repo's `projects/` folder has no `__init__.py` either. If the repo root is on `sys.path` (`python -c` from the root), the two merge as namespace portions, and `projects.dino` exists only in detrex. A test pins the resolved file location (validation Group 2).
- **`TechStack §2`'s "lib never imports from projects" test stays valid.** It greps for literal `import projects` / `from projects` lines in `lib/`. The adapter uses `importlib.import_module("projects.dino.modeling")` and `LazyConfig.load`, never such a statement. Keep it that way.
- **Editable builds leave both trees clean.** In the probe, `git status --porcelain` was empty in detrex and detectron2 after `uv pip install --no-build-isolation -e`. `build/`, `*.egg-info`, `detrex/version.py` (written by `setup.py:46-50`) and the in-tree `_C*.so` are all gitignored. So "unmodified" is checkable with `git status --porcelain`.
- **`setuptools<81` is required.** `detrex/config/config.py:18` does `import pkg_resources`. With setuptools 84.2.1 the import failed, and pinning `<81` fixed it (probe).
- **Hyperparameters come from detrex's own file, not a copy.** `build_dino_config` loads `dino_vitdet.py`, which itself imports `dino_r50.py`. That makes every value detrex's by construction, and the "equivalence" check reduces to a structure match (below). The `in_feature="${.net.out_feature}"` interpolation (`dino_vitdet.py:42`) requires the stem's LazyCall to pass `out_feature="last_feat"` **explicitly**: LazyCall records only the arguments it is given (probe: `InterpolationKeyError` without it).

### Architecture decisions
- **Stem = `LayerNorm(C, eps=1e-6)` + `Linear(C → 768)` (user decision).** Everything from the pyramid on then has exactly ViTDet-B's shapes (`embed_dim = 768`, `dino_vitdet.py:12`), and the same stem works for C = 1152 and C = 3072.
  - eps 1e-6 matches the ViT's `norm_layer` (`dino_vitdet.py:26`).
  - The layers use PyTorch's default init. detrex's DINO init only touches neck convs (`dino.py:122-125`).
  - FlowWM's stem ("multi_layer_pooler_512_w_layer_norm") is unreleased. The rejected options were LN + Linear → 512 (FlowWM's name, but non-ViTDet shapes) and LN only (SFP shapes would depend on C).
- **The stem declares a nominal stride of 16, not 14.** `SimpleFeaturePyramid` derives its level names from `stride / scale` (detectron2 `vit.py:404,451,459`). Stride 16 gives `p3`–`p6`, the names detrex's neck config expects (`dino_vitdet.py:51-57`). Stride 14 would give `p2`…, and the neck would not find its inputs. DINO never uses strides numerically: positions and boxes are normalised per map. The real token pitch is 14 px, which only enters through `image_size`.
- **Pyramid sizes.**
  - Default grid 32×64 → p3 64×128, p4 32×64, p5 16×32, p6 8×16, which is 10,880 tokens.
  - Native grid 32×40 → 64×80, 32×40, 16×20, 8×10.
  - The 0.5 level is `MaxPool2d(2, 2)` (`vit.py:425-426`) and p6 is `max_pool2d(kernel_size=1, stride=2)` (detectron2 `fpn.py:188-200`).
- **`square_pad=1024` needs no replacement: it is inert for DINO.** `DINO.preprocess_image` calls `ImageList.from_tensors(images)` with no `padding_constraints` (`dino.py:502-505`), and only `padding_constraints` reads `square_pad` (`vit.py:470-475`). It stays in the config untouched.
- **No padding masks.** All frames of one policy have the same grid (`irwm.features.preprocess.POLICY_SIZE`), so a batch is one tensor. In train mode `dino.py:178-183` sets `img_masks[:, :img_h, :img_w] = 0` using `instances.image_size` in pixels on a token-sized mask. Pixels ≥ tokens, so the slice covers the whole map and every mask is all-valid, which is correct for unpadded input. In eval mode the masks are zeros (`dino.py:185-186`).
- **Box frames.** Targets are xyxy pixels of the encoder input. The wrapper wraps them in `Instances(image_size, gt_boxes=Boxes(...), gt_classes=...)`, and `DINO.prepare_targets` normalises them to cxcywh in [0, 1] (`dino.py:550-559`). In eval, `DINO.inference` scales normalised boxes by the token-grid size `images.image_sizes` (`dino.py:542-544`). `detector_postprocess(results, height, width)` then rescales to `(H, W)` = `image_size`, clips and drops empty boxes (detectron2 `postprocessing.py:41-58`). So outputs are xyxy encoder-input pixels and K ≤ 300. F8 converts xyxy → xywh and maps them to FLIR pixels with `ResizeGeometry.invert_boxes` (`irwm/features/preprocess.py:64-71`). The output uses xyxy because detrex's `Instances.pred_boxes` is xyxy; we chose it so the boxes pass through without conversion.
- **Device handling.** `DINO.__init__` moves `pixel_mean` to `device` at construction (`dino.py:111-114`). `device="cuda"` (`dino_r50.py:109`) would therefore fail on a CPU-only machine. We build with `device="cpu"` and set `self.detr.device = feats.device` before every call. `prepare_targets` and `instances.to(self.device)` read it (`dino.py:205,554`). `pixel_mean`/`pixel_std` are never used, because `preprocess_image` is overridden.
- **Training needs CUDA; detrex stays unmodified (user decision).** `prepare_for_cdn` (`dino.py:370-458`) and `DINOCriterion.compute_dn_loss` (`dn_criterion.py:79-104`) hard-code `.cuda()`/`"cuda"`. A train-mode forward on CPU fails (probe: device-mismatch `RuntimeError`). We do not override those methods. Every training-forward test is `@pytest.mark.gpu`, and the wrapper raises a clear `RuntimeError` for CPU training. CPU tests still cover the matcher and the set-criterion losses, which have no `.cuda()` (`criterion.py`, `matcher.py`), through `TwoStageCriterion.forward`.
- **MSDA uses detrex's compiled CUDA op (user decision: speed matters, compile time is acceptable).** On CUDA tensors detrex calls `_C.ms_deform_attn_forward`, and on CPU tensors the `grid_sample` fallback (`multi_scale_deform_attn.py:341-353`). The CUDA op casts fp16 `value` to fp32 internally and casts the output back (`:343,355-356`). If `detrex._C` fails to import, detrex replaces the whole layer with a dummy that raises (`:420-428`). A CPU-only build has `_C` but no CUDA kernels, so CUDA use then fails loudly: we never get a silent slow path.
- **CUDA build settings.** `FORCE_CUDA=1` is needed because login nodes have no GPU (`setup.py:106-109`). `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"` covers the V100S (sm_70) and L4 (sm_89) on ctcomp3 plus common intermediate cards, so one shared `.venv` serves every node. These variables hold no paths, so TechStack §6 allows them. CUDA 12.1's `nvcc` accepts host gcc ≤ 12. ctdesk236 has gcc 15.2 and no `nvcc`, so ctdesk236 stays CPU-only (user decision: GPU tests run on ctcomp3 only).
- **Lock without building.** `uv lock` would otherwise run `setup.py` to get metadata, which imports torch and fails without build isolation. Declare both packages in `[[tool.uv.dependency-metadata]]` with:
  - `detectron2`: version `0.6` (`detectron2/__init__.py:10`), `requires-dist` = `detectron2/setup.py:162-196`;
  - `detrex`: version `0.3.0` (`setup.py` `setup(version=...)`), `requires-dist` = `third_party/detrex/requirements.txt`.

  Add `no-build-isolation-package = ["detectron2", "detrex"]`. Install in two phases: `uv sync --no-install-package detectron2 --no-install-package detrex`, then `uv sync`. The second phase builds against the installed torch.
- **Core dependency, not an extra (user decision).** Every `uv sync` builds both packages. This includes the Windows laptop still listed in TechStack §1, where it needs MSVC Build Tools (flagged to the user, not handled here).

### Equivalence and tests
- **Equivalence = structure match against stock detrex (user decision).** Our model and detrex's stock `dino_vitdet` are built from the same file. The stock one keeps its ViT-B and gets `num_classes=6`, `device="cpu"`. Outside `backbone.net.*` their `state_dict` key sets and shapes must be identical, and the key hyperparameters are asserted (table below). The DINO code itself is detrex's, so this proves that the only differences are the stem and the listed deviations. A golden-output fixture was dropped: it guarded vendored code, and we vendor none.
- **Overfit bar (user decision).** GPU, full default config, 4 fixed synthetic feature maps `[4, 1152, 32, 64]`, 1–5 boxes each, the detrex 12ep optimiser (AdamW lr 1e-4, betas (0.9, 0.999), wd 1e-4, clip-grad max_norm 0.1, L2: `dino_vitdet_base_4scale_12ep.py:28-30,37-39`), fp32, batch 4. It passes within ≤ 1500 iterations when both hold:
  - in eval mode, every GT box has a detection with the same label, IoU ≥ 0.8 and score ≥ 0.5;
  - the total weighted loss is ≤ 10 % of its step-0 value.
- **GPU tests run on ctcomp3 only, as a (user) step (user decision).** The implementer writes them and runs the CPU suite locally.

### Notes for F8 (recorded, not built here)
- **Ignore regions (user decision, 2026-10-23):** "Ignored classes are like they don't exist in the dataset; they shouldn't matter on training or eval. The model shouldn't know about them." F8 drops them from targets. F7's criterion is stock, with no ignore hook. Their evaluation treatment stays as F2 fixed it (crowd GT, neither TP nor FP).
- **detrex's optimiser gives every parameter whose name contains `backbone` 0.1× lr** (`dino_vitdet_base_4scale_12ep.py:40`), which was meant for a pretrained ViT. Our stem and pyramid are new layers named `detr.backbone.*`. F8 decides whether the factor applies.
- **Training schedule:** `dino_vitdet_base_4scale_{12,50}ep.py`: total batch 16 (`:48`), 90k iterations for 12ep (`:16`).

### Hyperparameter table (detrex @ `e244e6c`, `projects/dino/configs/models/` unless noted)

| Item | Value (ours) | detrex source | Same? |
|---|---|---|---|
| Image backbone `backbone.net` | `FeatureStem(C, 768)`: LN(eps 1e-6) + Linear | `dino_vitdet.py:16-41` ViT-B/16 (768-d, 12 layers) | **No** (D1) |
| SFP `in_feature` / `out_feature` | `last_feat` | `dino_vitdet.py:40,42` | yes |
| SFP `out_channels` | 256 | `dino_vitdet.py:43` | yes |
| SFP `scale_factors` | (2.0, 1.0, 0.5) | `dino_vitdet.py:44` | yes |
| SFP `top_block` | `LastLevelMaxPool` (k 1, s 2) | `dino_vitdet.py:45`; detectron2 `fpn.py:188-200` | yes |
| SFP `norm` | `"LN"` (channel LN, eps 1e-6) | `dino_vitdet.py:46`; detectron2 `batch_norm.py:333-353` | yes |
| SFP `square_pad` | 1024 (inert) | `dino_vitdet.py:47`; unused per `dino.py:502-505` | yes |
| Neck `ChannelMapper` inputs | p3–p6, 256 ch each | `dino_vitdet.py:51-57` | yes |
| Neck `num_outs` / `out_channels` | 4 / 256 | `dino_vitdet.py:58`, `dino_r50.py:44` | yes |
| Neck `kernel_size` / norm | 1 / `GroupNorm(32, 256)` | `dino_r50.py:46-47` | yes |
| Position embedding | Sine, 128 feats, T 10000, normalize, offset −0.5 | `dino_r50.py:31-36` | yes |
| Encoder | 6 layers, 256-d, 8 heads, FFN 2048, dropout 0/0, no post-norm, no checkpoint | `dino_r50.py:50-60` | yes |
| Decoder | 6 layers, 256-d, 8 heads, FFN 2048, dropout 0/0, return_intermediate, no checkpoint | `dino_r50.py:61-71` | yes |
| Decoder `look_forward_twice` | True | `dino_transformer.py:120` (default) | yes |
| MSDA points / im2col step | 4 / 64 | `detrex/layers/multi_scale_deform_attn.py:162-163` (defaults) | yes |
| `num_feature_levels` | 4 | `dino_r50.py:72`, `dino_vitdet.py:59` | yes |
| Two-stage proposals / learnt init query | = `num_queries` / True | `dino_r50.py:73`, `dino_transformer.py:255` | yes |
| `embed_dim` | 256 | `dino_r50.py:75` | yes |
| `num_classes` | **6** | `dino_r50.py:76` (80) | **No** (D2) |
| `num_queries` | 900 | `dino_r50.py:77` | yes |
| `aux_loss` | True | `dino_r50.py:78` | yes |
| Matcher costs class / bbox / giou | 2.0 / 5.0 / 2.0, focal cost α 0.25 γ 2 | `dino_r50.py:81-88` | yes |
| Loss weights class / bbox / giou (+ `_dn`) | 1 / 5.0 / 2.0 | `dino_r50.py:89-96` | yes |
| Class loss | focal, α 0.25, γ 2.0; `two_stage_binary_cls` False | `dino_r50.py:97-100` | yes |
| Aux / enc loss weights | copies of base weights for `_enc` and `_0`…`_4` | `dino_r50.py:113-121` | yes (recomputed, D8) |
| Denoising | `dn_number` 100, label noise 0.5, box noise 1.0 | `dino_r50.py:102-104` | yes |
| `pixel_mean` / `pixel_std` | unused | `dino_r50.py:105-106` | n/a (D3) |
| `device` | `"cpu"` at build, synced per call | `dino_r50.py:109` (`"cuda"`) | **No** (D4) |
| `vis_period` | 0 | `dino_r50.py:107` | yes |
| `select_box_nums_for_evaluation` | 300 | `dino.py:72` (default) | yes |

### Deviations from detrex / FlowWM (complete list)
- **D1 — Stem instead of ViT-B.** Our inputs are frozen F4 features. The stem is LN + Linear C → 768 (user decision). FlowWM's stem is unreleased.
- **D2 — `num_classes = 6`.** TechStack §3.4.
- **D3 — `preprocess_image` without pixel normalisation or padding.** The inputs are features, not pixels. FlowWM reaches the same with its unreleased `inference_from_backbone` kwarg.
- **D4 — `device="cpu"` at construction, synced to the input per call.** This lets the model build on CPU and follow Lightning's device placement.
- **D5 — Nominal stride 16 in `FeatureStem.output_shape()`.** It keeps the level names `p3`–`p6`; the real pitch is 14 px.
- **D6 — Boxes are scaled to the caller's `image_size`.** It is passed as `height`/`width` to `detector_postprocess`, the same mechanism FlowWM uses with original image sizes.
- **D7 — Wrapper API.** A tensor in, tensors out (`DINODetector`) around detrex's list-of-dicts interface. Pure conversion.
- **D8 — Aux weight dict recomputed after `overrides`.** For the default config the result is identical to `dino_r50.py:113-121`. With a changed decoder depth it stays correct; the config file's dict is computed for 6 layers.

## Step 1 — detrex submodule
Add `third_party/detrex` as a submodule of `https://github.com/IDEA-Research/detrex.git` at `e244e6c3da3e84566728c52c21fb061d23ce0e2f`, then `git -C third_party/detrex submodule update --init detectron2` (gitlink `70f45430…`). Commit `.gitmodules` and the gitlink. Covers: FR1.

## Step 2 — Dependencies and build
- `lib/pyproject.toml` dependencies gain `"detectron2"`, `"detrex"` and `"setuptools<81"`, with a one-line comment each (why `<81`: `detrex/config/config.py:18`).
- Root `pyproject.toml`:
  - `[tool.uv.sources]` gains `detectron2 = { path = "third_party/detrex/detectron2", editable = true }`, `detrex = { path = "third_party/detrex", editable = true }` and `irwm-detection = { workspace = true }`;
  - `[tool.uv]` gains `no-build-isolation-package = ["detectron2", "detrex"]`;
  - add two `[[tool.uv.dependency-metadata]]` tables (decision above).
- Run `uv lock`, then the two-phase sync on ctdesk236 (CPU ops).
- `slurm/setup_env.sh`: in its sync step, run the two-phase sync. Before phase 2, if `command -v nvcc` succeeds, export `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"` and pass `--reinstall-package detectron2 --reinstall-package detrex`. Print `nvcc --version | tail -1`, `gcc --version | head -1` and the elapsed seconds. Otherwise `warn` that the ops are CPU-only.
- Its verify step adds `python -c "import detectron2, detrex; from detrex import _C"`.

Covers: FR2.

## Step 3 — detrex adapter
Create `lib/irwm/detection/detrex_adapter.py` per FR3.
- `cuda_ops_available()` returns `False` on CPU-only machines without raising. On CUDA it builds `MultiScaleDeformableAttention(embed_dim=256, num_heads=8, num_levels=1, num_points=4, batch_first=True)` and calls it once. Arguments: `query = randn(1, 4, 256)`, `value = randn(1, 16, 256)`, `reference_points = rand(1, 4, 1, 2)`, `spatial_shapes = tensor([[4, 4]])`, `level_start_index = tensor([0])`, all on `cuda`. It returns `False` on any exception.
- Module docstring: the spec path, the commits, and why `sys.path` is appended rather than inserted.

Covers: FR3.

## Step 4 — Feature stem
Create `lib/irwm/detection/stem.py` per FR4.
- Forward: `x.permute(0, 2, 3, 1)` → `self.norm` → `self.proj` → `permute(0, 3, 1, 2).contiguous()`.
- Set `self._out_features = ["last_feat"]` (honouring `out_feature`), `_out_feature_channels`, and `_out_feature_strides = {out_feature: 16}` with a comment pointing to D5.

Covers: FR4.

## Step 5 — `FeatureDINO`, `build_dino_config`, `DINODetector`
Create `lib/irwm/detection/detector.py`:
- **Import `DINO`.** At module level: `ensure_detrex_projects()`, then `DINO = importlib.import_module("projects.dino.modeling").DINO`.
- **`FeatureDINO`** per FR5.
- **`build_dino_config`** per FR6:
  - apply overrides with `OmegaConf.select`/`OmegaConf.update` on the `model` node, after checking each key exists with `OmegaConf.select(model, key, throw_on_missing=False)`;
  - recompute the aux weights: `base = {k: v for k, v in weight_dict.items() if not k.endswith(("_enc",) + tuple(f"_{i}" for i in range(64)))}`. The base keys are exactly the 6 listed at `dino_r50.py:89-96`; assert that. Then rebuild `_enc` and `_0`…`_{n-2}`.
- **`DINODetector`** per FR7:
  - Train inputs: `[{"image": feats[i], "instances": inst_i, "height": H, "width": W}]`, where `inst_i = Instances((H, W)); inst_i.gt_boxes = Boxes(boxes); inst_i.gt_classes = labels`.
  - Eval inputs: `[{"image": feats[i], "height": H, "width": W}]`.
  - Eval outputs: convert `out["instances"]` → `{"boxes": pred_boxes.tensor, "scores": scores, "labels": pred_classes}`.
  - Token check: `sum(h_l · w_l)` over the 4 levels, with `h, w` → `(2h, 2w), (h, w), (⌊h/2⌋, ⌊w/2⌋), (⌈⌊h/2⌋/2⌉, ⌈⌊w/2⌋/2⌉)`. It is computed from the pyramid rule, not from a forward pass, and compared with `self.detr.num_queries` before the call.
- **`lib/irwm/detection/__init__.py`** re-exports the public names (Public API Summary). Keep `irwm.detection` importable with no CUDA and no data (`test_irwm_subpackages_import` imports it).

Covers: FR5, FR6, FR7.

## Step 6 — Hydra group
Create `lib/irwm/conf/detector/default.yaml` per FR8. Its header comment names the spec and says that every detrex hyperparameter comes from `dino_vitdet.py` and is changed only through `overrides`. Add `- detector: default` to `irwm_base.yaml`. Covers: FR8.

## Step 7 — Tests
Register the `gpu` marker and the new `addopts` in root `pyproject.toml`. Create `tests/test_detection_head.py`, covering validation.md Groups 1–8.
- **Reduced config for fast CPU tests:** `overrides={"num_queries": 100, "dn_number": 10, "transformer.encoder.num_layers": 1, "transformer.decoder.num_layers": 2}` on an 8×16 grid (680 tokens ≥ 100). Full-config CPU tests use one image at 32×64.
- **Synthetic overfit data** (Group 8): `torch.Generator().manual_seed(0)`. `feats = randn(4, 1152, 32, 64)`. Per image `n ∈ {1, 2, 3, 5}` boxes, with centres uniform in the frame, w and h uniform in [32, 200] px, clipped to 896×448, and labels `randint(0, 6)`. `torch.manual_seed(0)` before training (denoising noise uses the global RNG). Evaluate every 100 iterations and stop at the first pass.
- **GPU tests skip cleanly** with `pytest.mark.skipif(not cuda_ops_available(), reason=...)` in addition to the marker.

Covers: FR9.

## Step 8 — Head benchmark entry point
Create `projects/detection/pyproject.toml` (`name = "irwm-detection"`, `dependencies = ["irwm"]`, `[tool.uv] package = false`, `irwm = { workspace = true }`), `projects/detection/conf/head_bench.yaml` and `projects/detection/head_bench.py` per FR10. Structure them like `projects/feature_extraction/throughput.py` and `conf/throughput.yaml`: `load_env()` at import, `make_wandb_logger`, JSON to the Hydra run dir. Config keys: `grids: [[32, 64], [32, 40]]`, `batch_sizes: [1, 2, 4, 8]`, `precisions: [fp32, fp16]`, `warmup: 3`, `timed: 10`, `in_channels: ${detector.in_channels}`. Call `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.synchronize()` around each timed block. **Reminder (TechStack §6):** no hard-coded paths; the run dir comes from Hydra. Covers: FR10.

## Step 9 — CUDA build and GPU runs on ctcomp3 (user)
1. On the ctcomp3 login node, make a CUDA 12.x `nvcc` available (for example `module load cuda/12.1`) together with a host `gcc` that `nvcc` supports. Check `nvcc --version` and `gcc --version`.
2. `git pull && bash slurm/setup_env.sh` (the CUDA build may take tens of minutes; this is acceptable, user decision).
3. `srun --gres=gpu:1 --pty bash -c 'source slurm/env.sh && uv run pytest tests/test_detection_head.py -m gpu -v'` on an L4 node (hpc-gpu5).
4. `bash slurm/submit.sh ctcomp3 projects/detection/head_bench.py cluster.timeout_min=60`.

The user returns:
- the `setup_env.sh` lines with the `nvcc`/`gcc` versions and the build seconds;
- the full pytest output, including the overfit test's printed summary (iterations, loss ratio, min IoU, min score, seconds, peak GiB);
- `head_bench.json` and the W&B run id.

## Step 10 — Results note and constitution updates
- Write `spec/2026-10-23-dino-detr-head-port/results.md` per FR11 from the returned files.
- Then follow the Roadmap done convention: mark F7 ✓ DONE with a result note, and update the Component Registry (detection rows, the detrex submodule row, the Hydra group, `projects/detection`).
- Amend TechStack:
  - §3.2: detrex/detectron2 are runtime dependencies via the pinned submodule (this spec's deviation, user decision 2026-10-23), and validation is the structure match, not "comparable AP on a small run";
  - §2: add `third_party/detrex` to the layout;
  - §7: add the `gpu` marker;
  - §1: add the build toolchain (`nvcc`, `FORCE_CUDA`) and note that `pycocotools` now arrives with detectron2.

Covers: FR11.

## Implementation Order
1. Step 1 — detrex submodule
2. Step 2 — Dependencies and build (CPU ops on ctdesk236)
3. Step 3 — detrex adapter
4. Step 4 — Feature stem
5. Step 5 — `FeatureDINO`, `build_dino_config`, `DINODetector`
6. Step 6 — Hydra group
7. Step 7 — Tests → `uv run pytest tests/test_detection_head.py -v`, then the full suite `uv run pytest`
8. Step 8 — Head benchmark entry point (check locally that it exits non-zero with a clear message without CUDA ops)
9. Step 9 — CUDA build and GPU runs on ctcomp3 (user)
10. Step 10 — Results note and constitution updates
