# F7 · DINO-DETR head port: Requirements

## Goal

Deliver a DINO-DETR detection head that consumes our frozen DINOv2-Reg features `[B, C, H', W']` (C = 1152 PCA coefficients by default, 3072 full features optionally) in place of an image backbone. It follows the route FlowWM takes (Roadmap F7 note): detrex's own `dino_vitdet` model, run from an installed detrex, with the ViT replaced by a small feature stem. The head is what H1 is measured with (Mission), and F8 trains it once per modality (`det_rgb`, `det_ir`); F9 and F12 reuse it frozen. F7 delivers the module, its build, its tests and a GPU cost measurement. It trains nothing on real data.

## Scope

In scope:
- detrex (`IDEA-Research/detrex` @ `e244e6c`, with its nested detectron2 @ `70f4543`) as a pinned, unmodified git submodule, installed into the uv env with its compiled ops (CPU everywhere, CUDA where `nvcc` is available).
- `irwm.detection`: detrex adapter, `FeatureStem`, `FeatureDINO`, `build_dino_config`, `DINODetector`.
- Shared Hydra group `detector`.
- A `gpu` pytest marker, CPU tests and GPU tests (`tests/test_detection_head.py`).
- `projects/detection/head_bench.py`: train-step time and peak memory of the head on the GPU, for F8's budget.
- A results note `spec/2026-10-23-dino-detr-head-port/results.md`.

Out of scope:
- Training on FLIR, the `LightningModule`/`LightningDataModule`, optimiser and schedule, augmentation, val mAP (F8).
- Handling of FLIR ignore regions in targets (F8; see plan.md, user decision on ignore regions).
- The optimiser's `lr_factor_func` (detrex gives `backbone.*` 0.1× lr; our stem and pyramid are named `backbone.*`), and head precision for training (F8).
- The rigorous COCO evaluator (F6).
- Any change to detrex or detectron2 source.

## Functional Requirements

**FR1 — detrex submodule (`third_party/detrex`).**
- `third_party/detrex` is a git submodule of `https://github.com/IDEA-Research/detrex.git` checked out at `e244e6c3da3e84566728c52c21fb061d23ce0e2f`. Its nested submodule `detectron2` is at `70f454304e1a38378200459dd2dbca0f0f4a5ab4` (detrex's own gitlink).
- After a build, `git status --porcelain` is empty in both trees (build artefacts are covered by their `.gitignore`).
- `slurm/setup_env.sh` already runs `git submodule update --init --recursive`. That covers both.

**FR2 — Dependencies and build (`pyproject.toml`, `lib/pyproject.toml`, `slurm/setup_env.sh`).**
- `detectron2` (0.6) and `detrex` (0.3.0) are **core** dependencies of `irwm`, installed as **editable path sources** from `third_party/detrex/detectron2` and `third_party/detrex`.
- `setuptools<81` is a core dependency. detrex imports `pkg_resources` at import time (`detrex/config/config.py:18`), and setuptools ≥ 81 removed it.
- Both packages are built without build isolation, against the env's torch 2.2.0. `uv lock` never builds them: their metadata is declared statically in `[[tool.uv.dependency-metadata]]`.
- `slurm/setup_env.sh` builds the CUDA ops when `nvcc` is on `PATH`: it sets `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"` and reinstalls both packages. It prints the `nvcc --version` and `gcc --version` lines and the build wall time. If `nvcc` is missing, it builds CPU-only ops and prints a warning that GPU training will fail.
- After setup, `python -c "import detectron2, detrex; from detrex import _C"` succeeds.

**FR3 — detrex adapter (`lib/irwm/detection/detrex_adapter.py`).**
- `DETREX_COMMIT = "e244e6c3da3e84566728c52c21fb061d23ce0e2f"`, `DETECTRON2_COMMIT = "70f454304e1a38378200459dd2dbca0f0f4a5ab4"`.
- `detrex_root() -> Path` is `irwm.utils.paths.repo_root() / "third_party" / "detrex"`. It raises `FileNotFoundError` naming `git submodule update --init --recursive` if `projects/dino/modeling/__init__.py` is missing.
- `ensure_detrex_projects() -> None` **appends** `str(detrex_root())` to `sys.path` once (idempotent). After it runs, `importlib.import_module("projects.dino.modeling")` resolves to a file under `third_party/detrex/projects/`.
- `load_dino_vitdet_config() -> omegaconf.DictConfig` returns `detectron2.config.LazyConfig.load(detrex_root() / "projects/dino/configs/models/dino_vitdet.py")` after `ensure_detrex_projects()`. The config is not copied anywhere in `irwm`.
- `cuda_ops_available() -> bool` is `True` iff CUDA is available and detrex's `MultiScaleDeformableAttention` forward runs on a `[1, 4, 256]`-query CUDA tensor without raising.

**FR4 — Feature stem (`lib/irwm/detection/stem.py`).**
- `class FeatureStem(detectron2.modeling.Backbone)` with `__init__(self, in_channels: int = 1152, embed_dim: int = 768, out_feature: str = "last_feat")`.
- `forward(x: Tensor[B, C, H', W']) -> {"last_feat": Tensor[B, embed_dim, H', W']}`: per-token `nn.LayerNorm(in_channels, eps=1e-6)` over channels, then `nn.Linear(in_channels, embed_dim)`. Submodule names are `norm` and `proj`.
- `output_shape()` returns `{"last_feat": ShapeSpec(channels=embed_dim, stride=16)}`. The stride is **nominal** (see plan.md).
- `x.ndim != 4` or `x.shape[1] != in_channels` → `ValueError` naming the expected and actual shapes.

**FR5 — Feature-input DINO (`lib/irwm/detection/detector.py`).**
- `class FeatureDINO(DINO)`, where `DINO` is detrex's `projects.dino.modeling.DINO`, overrides **only** `preprocess_image(batched_inputs)`. It returns `detectron2.structures.ImageList.from_tensors([x["image"].to(self.device) for x in batched_inputs])`, with no pixel normalisation and no padding.
- Every other method is detrex's, unmodified.

**FR6 — Config builder (`lib/irwm/detection/detector.py`).**
- `build_dino_config(*, in_channels: int = 1152, stem_dim: int = 768, num_classes: int = 6, overrides: Mapping[str, Any] | None = None) -> DictConfig` starts from `load_dino_vitdet_config().model` and applies, in order:
  1. `model._target_ = FeatureDINO`;
  2. `model.backbone.net = LazyCall(FeatureStem)(in_channels=in_channels, embed_dim=stem_dim, out_feature="last_feat")`;
  3. `model.num_classes = num_classes`, `model.device = "cpu"`;
  4. each `overrides` item as a dotted key relative to `model` (`{"transformer.encoder.num_layers": 2}` → `model.transformer.encoder.num_layers = 2`). An unknown key raises `KeyError`. The keys `backbone.net`, `backbone.net.*`, `num_classes` and `device` raise `ValueError`, because they are set by arguments;
  5. the auxiliary weight dict is recomputed from the base weights with the same rule as `dino_r50.py:113-121`, using the possibly overridden `transformer.decoder.num_layers`.
- With no overrides, every value except those set in items 1–3 equals detrex's file (plan.md, hyperparameter table).

**FR7 — Detector wrapper (`lib/irwm/detection/detector.py`).**
- `class DINODetector(nn.Module)` with `__init__(self, *, in_channels: int = 1152, stem_dim: int = 768, num_classes: int = len(irwm.data.classes.CLASS_NAMES), overrides: Mapping[str, Any] | None = None)`. It instantiates `build_dino_config(...)` into `self.detr` (a `FeatureDINO`), so every parameter key starts with `detr.`.
- `forward(feats: Tensor[B, C, H', W'], targets: list[dict[str, Tensor]] | None = None, image_size: tuple[int, int] | None = None)`. `image_size = (H, W)` is the encoder-input size in pixels; the default is `(14·H', 14·W')`. Behaviour switches on `self.training`, as in detrex (user decision):
  - **train mode** → `dict[str, Tensor]` of weighted losses, exactly what `DINO.forward` returns (39 keys with the default 6-layer decoder). `targets` has one dict per image: `"boxes"` `float32 [N, 4]` **xyxy in encoder-input pixels** and `"labels"` `int64 [N]` in `0..num_classes-1`. `N = 0` is allowed. The total loss is `sum(losses.values())`.
  - **eval mode** → `list[dict]`, one per image: `"boxes"` `float32 [K, 4]` xyxy in encoder-input pixels, clipped to `image_size`; `"scores"` `float32 [K]` in `[0, 1]`; `"labels"` `int64 [K]`. K ≤ 300 (`select_box_nums_for_evaluation`; detectron2 `detector_postprocess` drops boxes that are empty after clipping). `targets` is ignored.
- Before each call it sets `self.detr.device = feats.device`.
- Errors:
  - train mode with `targets is None` or `len(targets) != B` → `ValueError`;
  - train mode with `feats` not on CUDA → `RuntimeError` saying that detrex's denoising code hard-codes `.cuda()` (`dino.py:370-458`), so training needs a GPU;
  - total tokens over the 4 pyramid levels < `num_queries` → `ValueError` (otherwise `torch.topk` fails inside `DINOTransformer.forward`, `dino_transformer.py:430`);
  - `feats` shape or channel mismatch → `ValueError` (FR4).

**FR8 — Hydra group `detector` (`lib/irwm/conf/detector/default.yaml`, `lib/irwm/conf/irwm_base.yaml`).**
- Keys: `in_channels: 1152`, `stem_dim: 768`, `num_classes: 6`, `overrides: {}`. Each key carries a comment citing its source.
- `irwm_base.yaml` defaults gain `- detector: default` after `encoder: default`.
- `DINODetector(**OmegaConf.to_container(cfg.detector))` builds the default head.

**FR9 — Tests (`tests/test_detection_head.py`, `pyproject.toml`).**
- A new marker `gpu: needs CUDA and detrex/detectron2 built with CUDA ops; skipped by default (run with -m gpu)` is registered. `addopts` becomes `-m 'not data and not gpu'`.
- CPU tests cover FR1–FR8 and the matcher and criterion on CPU (validation.md Groups 1–6). They need no data or weights and finish in < 60 s on ctdesk236.
- GPU tests (`@pytest.mark.gpu`, skipped with a reason if `cuda_ops_available()` is false) cover CUDA ops, training losses, fp16 and the overfit (validation.md Groups 7–8).

**FR10 — Head benchmark (`projects/detection/`).**
- New uv workspace member `irwm-detection` (`projects/detection/pyproject.toml`, depends on `irwm`), registered in the root `[tool.uv.sources]`.
- `projects/detection/head_bench.py` (Hydra, config `conf/head_bench.yaml`, `project: detection`) runs on synthetic features for every combination of:
  - grid `[32, 64]` and `[32, 40]`;
  - batch size `1, 2, 4, 8`;
  - precision `fp32` and `fp16` (CUDA autocast).
- Each run does 3 warm-up and 10 timed train steps: forward, `sum(losses).backward()`, then `AdamW(lr=1e-4, weight_decay=1e-4)` step with grad clip `max_norm=0.1`.
- It writes `<run_dir>/head_bench.json` with, per combination: median step seconds, images/s, peak allocated GiB (`torch.cuda.max_memory_allocated`), and `"oom": true` when CUDA OOM is raised (caught, then the next combination runs). It also records `git_info()`, the GPU name and `cuda_ops_available()`, and logs to W&B. Without CUDA it exits with a non-zero status and a clear message.

**FR11 — Results note (`spec/2026-10-23-dino-detr-head-port/results.md`).**
- It records:
  - the ctcomp3 CUDA build: `nvcc`/`gcc` versions, wall time, node;
  - the `pytest -m gpu` output summary;
  - the overfit run: iterations to pass, final loss ratio, min IoU, min score, wall time, peak memory;
  - the `head_bench.json` table;
  - the head's parameter count;
  - a confirmation that the deviation list in plan.md is complete.

## Public API Summary

```python
# lib/irwm/detection/detrex_adapter.py
DETREX_COMMIT: str
DETECTRON2_COMMIT: str
def detrex_root() -> Path: ...
def ensure_detrex_projects() -> None: ...
def load_dino_vitdet_config() -> DictConfig: ...
def cuda_ops_available() -> bool: ...

# lib/irwm/detection/stem.py
class FeatureStem(detectron2.modeling.Backbone):
    def __init__(self, in_channels: int = 1152, embed_dim: int = 768, out_feature: str = "last_feat"): ...
    def forward(self, x: Tensor) -> dict[str, Tensor]: ...   # [B, C, H', W'] -> {"last_feat": [B, 768, H', W']}

# lib/irwm/detection/detector.py
class FeatureDINO(DINO):                                     # detrex DINO; preprocess_image overridden only
    def preprocess_image(self, batched_inputs: list[dict]) -> ImageList: ...
def build_dino_config(*, in_channels: int = 1152, stem_dim: int = 768, num_classes: int = 6,
                      overrides: Mapping[str, Any] | None = None) -> DictConfig: ...
class DINODetector(nn.Module):
    def __init__(self, *, in_channels: int = 1152, stem_dim: int = 768, num_classes: int = 6,
                 overrides: Mapping[str, Any] | None = None): ...
    def forward(self, feats: Tensor, targets: list[dict[str, Tensor]] | None = None,
                image_size: tuple[int, int] | None = None) -> dict[str, Tensor] | list[dict[str, Tensor]]: ...
    # train: {loss_name: Tensor}   eval: [{"boxes": [K,4] xyxy px, "scores": [K], "labels": [K]}]

# lib/irwm/detection/__init__.py re-exports: DINODetector, FeatureStem, FeatureDINO, build_dino_config,
#                                            cuda_ops_available
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| `third_party/detrex` @ `e244e6c` (+ `detectron2` @ `70f4543`) | read (import, config file) | New submodule, Apache-2.0, never modified |
| `projects/dino/configs/models/dino_vitdet.py`, `dino_r50.py` (detrex) | read | Loaded with `LazyConfig.load`; the source of every hyperparameter |
| `irwm.data.classes.CLASS_NAMES` | read (import) | `num_classes` default (TechStack §3.4) |
| `irwm.utils.paths.repo_root`, `irwm.utils.tracking` | read (import) | Submodule location; bench provenance and W&B |
| `$IRWM_OUTPUT_ROOT/detection/...` | write | `head_bench.json` (Hydra run dir) |
| GitHub (`IDEA-Research/detrex`, `facebookresearch/detectron2`) | download | Submodule clone only; no network at runtime |
| New deps | — | `irwm` core: `detectron2`, `detrex` (editable path sources), `setuptools<81`. Pulled in transitively by them: `fvcore`, `iopath`, `pycocotools`, `fairscale`, `opencv-python`, `scipy`, `timm`, `einops`, `submitit`, `yacs`, `tabulate`, `tensorboard`, `matplotlib` and the rest of `detectron2/setup.py:162-196` and detrex `requirements.txt` |
| Build toolchain | — | C++ compiler everywhere; on ctcomp3 also a CUDA 12.x `nvcc` and a host `gcc` that `nvcc` supports (user, plan.md Step 9) |
