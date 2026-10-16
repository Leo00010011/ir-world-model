# F4 · DINOv2-Reg encoder wrapper: Requirements

## Goal

F4 delivers the frozen DINOv2-Reg ViT-B/14 (`dinov2_vitb14_reg`, 4 register tokens) feature encoder, plus the image preprocessing that feeds it. Together they turn a FLIR frame into DINO-Foresight's feature space: 3072-d (blocks 3/6/9/12) or 1152-d PCA coefficients. Both must equal the reference extraction exactly. Every downstream result rests on these features. F8 trains the detectors on them, encoded on the fly. F9 and F10 read them as oracle and world-model input. F5 caches them if it is ever built. If the features differ from the reference, a change in AP could come from our encoder rather than from IR, so equivalence with the reference is the core requirement. F4 also produces the evidence that two open decisions are waiting for:
- **R5:** a FLIR re-run of the F1 resolution probe on all 330 clips of the F3 clip set.
- **F8 budgeting:** encoding throughput on ctcomp3.

## Scope

In scope:
- `irwm.features.preprocess`: resolution policy (`stretch`, `crop`, `native`), thermal 1→3 replication, ImageNet normalisation, and the box geometry that goes with each image.
- `irwm.features.encoder`: `DinoV2RegEncoder` with `encode_full`, `encode_pca`, `pca_inverse`. It is frozen, batched and GPU-ready, with fp16 autocast (default) or fp32.
- Shared Hydra group `encoder` (`lib/irwm/conf/encoder/default.yaml`).
- Moving `one_step` and `pca_metrics` from `projects/dinof_audit/audit.py` into `lib/irwm/worldmodel/reference.py` (now used by two projects).
- Project `projects/feature_extraction/` with two entry points: `r5_probe.py` (FLIR R5 probe) and `throughput.py`.
- Tests in `tests/test_encoder.py`, and a results note `results.md` in this folder.

Out of scope:
- **Policy (d), two overlapping windows.** Implemented only if R5 picks it (Roadmap F4).
- **Deciding R5.** F4 provides the probe numbers. The user decides R5 afterwards, before F8.
- **Feature cache.** F5 is deferred (Roadmap 2026-10-15). F4 writes no features to disk.
- **World-model rollout API.** That is F10. The probe uses the reference model for 1 step only.
- **Image augmentation, including random crops, target filtering and the F8 data pipeline** (F8; user decision: F4 provides centre crops only).
- **Restricting evaluation GT to the visible crop region** (F6/F9/F12).
- **Checkpoint-size handling of the frozen encoder inside a `LightningModule`** (F8).

## Functional Requirements

**FR1 — Resolution policies (`lib/irwm/features/preprocess.py`).**
- `Policy = Literal["stretch", "crop", "native"]`. `POLICY_SIZE: dict[Policy, tuple[int, int]] = {"stretch": (448, 896), "crop": (448, 896), "native": (448, 560)}`, as `(H, W)`.
- **`stretch`:** the full frame is resized to 448×896 with no crop. This is exactly what the reference transform `T.Resize((448, 896))` does to any frame (`third_party/DINO-Foresight/src/data.py:165`).
- **`crop`:** the frame is centre-cropped to 2:1 with `irwm.worldmodel.reference.center_crop_box(w, h, (2, 1))`, then resized to 448×896. The same rule applies to every source size (640×512, 1224×1024, 1024×768).
- **`native`:** the frame is centre-cropped to 5:4 with `center_crop_box(w, h, (5, 4))`, then resized to 448×560 (32×40 tokens). The same rule applies to every source size (user decision). Height is cropped when the frame is narrower than 5:4, and width when it is wider. There is no distortion. Crop boxes for the sizes in F2 `stats.md`:
  - thermal 640×512 → `(0, 0, 640, 512)` (no crop);
  - RGB 1280×1024 → no crop;
  - RGB 1224×1024 → `(0, 22, 1224, 979)` (4.4 % of the height lost);
  - RGB 1024×768 → `(32, 0, 960, 768)` (6.25 % of the width lost);
  - RGB 1800×1600 → `(0, 80, 1800, 1440)` (10 % of the height lost);
  - RGB 2048×1536 → `(64, 0, 1920, 1536)` (6.25 % of the width lost);
  - RGB 720×480 → `(60, 0, 600, 480)` (16.7 % of the width lost).
- **Only centre crops** (user decision). `Preprocess` has no random or caller-supplied crop. Any training augmentation is F8's.
- `crop` and `native` are deviations from the reference input pipeline. `stretch` is not.

**FR2 — `Preprocess` (`lib/irwm/features/preprocess.py`).**
- `Preprocess(policy: Policy, modality: Literal["rgb", "thermal"])` is a picklable callable for `FlirImageDataset(transform=...)`.
- `__call__(img: np.ndarray) -> tuple[Tensor, ResizeGeometry]` takes `uint8 [H, W]` (thermal) or `[H, W, 3]` (RGB) and returns `float32 [3, H_out, W_out]` plus its geometry.
- The pipeline runs in this order, mirroring the reference loader (`src/data.py:162-165`):
  1. `Image.fromarray`, then `.convert("RGB")`. For thermal (mode `L`) this replicates the channel 3×.
  2. If the policy is `crop` or `native`, `img.crop(box)` (skipped when the box is the full frame).
  3. `torchvision.transforms.Resize((H_out, W_out))` on the PIL image (PIL bilinear with antialias, as in the reference).
  4. `ToTensor`.
  5. `Normalize(IMAGENET_MEAN, IMAGENET_STD)`, using the constants from `irwm.worldmodel.reference`.
- Wrong `ndim`, wrong channel count for the modality, or a non-`uint8` dtype → `ValueError`. An unknown policy or modality → `ValueError` at construction.

**FR3 — `ResizeGeometry` (`lib/irwm/features/preprocess.py`).**
- A frozen dataclass with these fields: `src_size: tuple[int, int]` (W, H of the source), `crop_box: tuple[int, int, int, int]` (left, top, w, h in source pixels; the full frame for `stretch`), `out_size: tuple[int, int]` (W, H of the output).
- `apply_boxes(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]` maps COCO `xywh` boxes in source pixels `[N, 4]` → (`xywh` in output pixels `float32 [N, 4]`, `keep: bool [N]`). Steps:
  1. clip the box to the crop box;
  2. subtract the crop offset;
  3. scale by `out/crop` per axis.

  `keep` is `False` when the clipped width or height is ≤ 0, and those rows hold zeros. `N = 0` is allowed.
- `invert_boxes(boxes: np.ndarray) -> np.ndarray` maps `xywh` in output pixels → `xywh` in source pixels (scale, then add the offset; no clipping). F8 uses it to report predictions in image pixels.
- **Round trip:** for any box fully inside the crop box, `invert_boxes(apply_boxes(b)[0]) == b` to `atol=1e-3` px.

**FR4 — `DinoV2RegEncoder` construction and freezing (`lib/irwm/features/encoder.py`).**
- `DinoV2RegEncoder(backbone: nn.Module, pca: PCABasis, *, precision: Literal["fp16", "fp32"] = "fp16")` is an `nn.Module`. `LAYERS = (2, 5, 8, 11)`: 0-indexed, i.e. blocks 3/6/9/12.
- `from_assets(*, precision="fp16", hub_version=DINOV2_HUB_PINNED, device="cpu") -> DinoV2RegEncoder` builds the backbone with `irwm.worldmodel.reference.load_dinov2_backbone(hub_version)` and the PCA with `PCABasis.load(irwm.worldmodel.assets.dinof_dir() / PCA_EXPORT)`. It uses no network. A missing file raises `FileNotFoundError` naming the path and `projects/dinof_audit/download.py`.
- **Frozen:** every parameter has `requires_grad=False`. The module is in `eval()` after construction, and `train(mode)` is overridden so the module stays in eval mode and returns `self`. All encoding runs under `torch.no_grad()`, **not** `torch.inference_mode()`, so F8's trainable head can use the outputs.
- An unknown `precision` → `ValueError`.

**FR5 — `encode_full` (`lib/irwm/features/encoder.py`).**
- `encode_full(x: Tensor) -> Tensor` takes `[B, 3, H, W]` with `H` and `W` multiples of 14 and returns `float32 [B, H/14, W/14, 3072]`, channels-last.
- It computes `torch.cat(backbone.get_intermediate_layers(x, n=LAYERS, reshape=False, norm=True), dim=-1)`. This is exactly the reference `Dino_f.extract_features` (`src/dino_f.py:276-288`): final norm on every block, CLS and the 4 registers dropped by the DINOv2 hub code itself.
- The tokens are reshaped row-major: token `i` → `(i // (W/14), i % (W/14))`, as `einops.rearrange('b (h w) c -> b h w c')` does in the reference (`src/dino_f.py:408`).
- A non-4D input, `C ≠ 3`, or `H`/`W` not divisible by 14 → `ValueError`.

**FR6 — `encode_pca` and `pca_inverse` (`lib/irwm/features/encoder.py`).**
- `encode_pca(x: Tensor) -> Tensor` maps `[B, 3, H, W]` → `float32 [B, H/14, W/14, 1152]`. It applies `PCABasis.transform` to the concatenated features, inside the **same** precision context as the backbone.
- `pca_inverse(z: Tensor) -> Tensor` maps `[..., 1152]` → `[..., 3072]` via `PCABasis.inverse`, in fp32 with no autocast.

**FR7 — Precision.**
- With `precision="fp16"` on a CUDA tensor, backbone and PCA both run inside `torch.autocast("cuda", dtype=torch.float16)`. That is what Lightning `16-mixed` does in every reference training and eval command (`third_party/DINO-Foresight/README.md:88-151`, `train.py:151`). Outputs are cast to `float32` afterwards, which is lossless.
- With `precision="fp16"` on a CPU tensor, the encoder runs in fp32 and emits one `UserWarning` per instance. CPU fp16 autocast is not what the reference does.
- `precision="fp32"` runs with autocast explicitly disabled, so an outer autocast context (for example F8's `16-mixed` trainer) cannot change it.

**FR8 — Equivalence with the reference (demo clip, ctcomp3).**
Every comparison runs our encoder and the reference code **at the same precision**, on the same device, on the **same 5-frame batch**. The batch must not be split or reordered: a different batch size can make cuBLAS pick a different kernel. The input is the 5 demo frames from `load_demo_frames((448, 896))`, flattened to `[5, 3, 448, 896]`.

| Test | Ours | Reference | Required | What it proves |
|---|---|---|---|---|
| A (recipe) | `precision="fp32"` | `extract_features` / `pca_transform` called with no autocast (the reference is not intrinsically fp16; `16-mixed` is only a Lightning Trainer setting) | max \|Δ\| = 0 | Layers, final norm, token drop, token order and PCA are identical |
| B (production path) | `precision="fp16"` | the same reference calls inside `torch.autocast("cuda", dtype=torch.float16)`, which is what `16-mixed` does | max \|Δ\| = 0; if not exactly 0, accepted only with max \|Δ\| ≤ 1e-2 **and** the differing op named in `results.md` (user decision) | The default path used in every real run equals the reference's real path |
| C (information) | `precision="fp16"` | ours, `precision="fp32"` | no threshold; max/mean \|Δ\| and mean token cosine recorded | How much fp16 itself changes the features |

- A and B compare `encode_full` against `extract_features`, and `encode_pca` against `pca_transform(extract_features(...))`, both reshaped to `[5, 32, 64, C]`.
- **Preprocessing.** `Preprocess("stretch", "rgb")` applied to the 5 demo PNGs (loaded as `uint8` arrays) equals `load_demo_frames((448, 896))[0]` with |Δ| = 0. Likewise, `Preprocess("crop", "rgb")` equals `load_demo_frames((448, 896), (2, 1))[0]` with |Δ| = 0.

**FR9 — Hydra group `encoder` (`lib/irwm/conf/encoder/default.yaml`).**
- Keys: `policy: stretch` (provisional R5 preference; the comment says R5 is open), `precision: fp16`, `hub_version: dinof_release`.
- Added to the `irwm_base` defaults after `flir`.
- `encoder_tag(policy: Policy) -> str` lives in `irwm.features.encoder`:
  - `stretch` → `dinov2_vitb14reg_448x896_l3-6-9-12_pca1152` (the TechStack §4.2 tag);
  - `crop` → `dinov2_vitb14reg_448x896crop_l3-6-9-12_pca1152`;
  - `native` → `dinov2_vitb14reg_448x560_l3-6-9-12_pca1152`.

**FR10 — Shared reference helpers (`lib/irwm/worldmodel/reference.py`).**
- `one_step(model, x)` and `pca_metrics(pred, gt, beta)` move from `projects/dinof_audit/audit.py:108-126` into `reference.py` unchanged, with the same signatures, docstrings and behaviour. `audit.py` imports them from there.
- No module-level heavy imports are added: `test_no_heavy_toplevel_imports_in_lib` still passes.

**FR11 — R5 FLIR probe (`projects/feature_extraction/r5_probe.py`).**
- For every clip of `ClipIndex.load()` (`flir_test_fs5_c4f6_cs10`, 330 clips, evaluable or not), for each modality (`rgb`, `thermal`) and each policy (`stretch`, `crop`, `native`):
  1. Take the 5 frames `context_ids(m)` (4) and `future_ids(m)[0]`, read through `FlirSplit.load(m, "test").by_image_id(...).path`.
  2. Apply `Preprocess(policy, m)`, then `DinoV2RegEncoder.encode_pca` with `precision=fp16`.
  3. Stack to `x [1, 5, h, w, 1152]`.
  4. Run `one_step(model, x)` with the reference model under `torch.autocast("cuda", float16)`.
  5. `pca_metrics` of the prediction vs `x[:, -1]`, and of copy-last `x[:, -2]` vs `x[:, -1]`, with `beta = model.args.beta_smoothl1`.
- The reference model is `load_reference(dinof_dir() / DINOF_HIGHRES, pca_ckpt=dinof_dir() / DINOF_PCA, ...)` (the released PCA file; amended 2026-10-16, see plan Amendments) with `img_size=None` for `stretch`/`crop` and `img_size=(448, 560)` for `native`.
- It writes `r5_probe.json` to the Hydra run dir with these contents:
  - per-clip rows: `clip_id`, `rgb_video_id`, `segment_index`, modality, policy, model and copy-last `smoothl1`/`cosine`, and `ratio = model_smoothl1 / copy_last_smoothl1`;
  - per (modality, policy): `n_clips`, mean ratio, median ratio, ratio of summed SmoothL1, and per-`rgb_video_id` mean ratio (8 sequences);
  - per modality: for each policy pair, the number of sequences (out of 8) where the first policy has the lower mean ratio;
  - provenance: git commit, resolved config, clip-set name and SHA-256, `encoder_tag` per policy, GPU name.
- It logs the per-(modality, policy) summary to W&B. It makes no R5 decision.

**FR12 — Throughput (`projects/feature_extraction/throughput.py`).**
- **GPU-only part:** synthetic normalised input `[B, 3, H, W]` for `H×W ∈ {448×896, 448×560}`, `precision ∈ {fp16, fp32}`, `B ∈ {1, 16, 64}`. For each combination: 5 warm-up batches, then 20 timed batches with `torch.cuda.synchronize()`. It records frames/s and peak memory (`torch.cuda.max_memory_allocated`). An OOM is recorded as `"oom"`, not raised.
- **End-to-end part:** for each modality, `FlirImageDataset(FlirSplit.load(m, "train"), transform=Preprocess(policy, m))` in a `DataLoader` with `batch_size=32`, `num_workers=cfg.num_workers` (default `${oc.env:SLURM_CPUS_PER_TASK,8}`) and `pin_memory=True`, followed by `encode_pca` in fp16. It times 10 warm-up and then 100 batches, and records:
  - frames/s;
  - the encode share: time spent in `.cuda()` transfer + `encode_pca` (measured with `torch.cuda.synchronize()` around it) ÷ total wall time of the timed batches;
  - the projected seconds per epoch (`len(split) / frames_per_s`). The policy is `cfg.policy`, default `stretch`.
- It writes `throughput.json` (all numbers, git commit, config, GPU name, CPU count) to the Hydra run dir and logs it to W&B.

**FR13 — Results note (`spec/2026-10-16-dinov2-reg-encoder-wrapper/results.md`).**
- It records:
  - FR8 equivalence numbers (fp32 and fp16 max |Δ|, the run commit and GPU);
  - the fp16-vs-fp32 drift of our encoder (max and mean |Δ| in PCA space);
  - the FR12 tables;
  - the FR11 summary table (mean, median and summed ratio per modality × policy, plus sequence win counts);
  - where positional embeddings are interpolated for 32×40: DINOv2-Reg's own interpolation on every forward pass (cite `file:line` in the pinned `dinof_release` hub code) and DINO-Foresight's `high_res_adapt` at load time (`src/dino_f.py:231-239`);
  - one paragraph comparing with the F1 Cityscapes probe (a 0.662, b 0.551, c 0.648).
- It states that R5 is still the user's decision.

## Public API Summary

```python
# lib/irwm/features/preprocess.py
Policy = Literal["stretch", "crop", "native"]
POLICY_SIZE: dict[Policy, tuple[int, int]]          # (H, W)

@dataclass(frozen=True)
class ResizeGeometry:
    src_size: tuple[int, int]                       # (W, H)
    crop_box: tuple[int, int, int, int]             # (left, top, w, h), source px
    out_size: tuple[int, int]                       # (W, H)
    def apply_boxes(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # xywh [N,4] -> (xywh [N,4], keep [N])
    def invert_boxes(self, boxes: np.ndarray) -> np.ndarray: ...                    # xywh [N,4] -> xywh [N,4]

class Preprocess:
    def __init__(self, policy: Policy, modality: Literal["rgb", "thermal"]) -> None: ...
    def __call__(self, img: np.ndarray) -> tuple[Tensor, ResizeGeometry]: ...      # uint8 [H,W(,3)] -> [3,H',W'] fp32

# lib/irwm/features/encoder.py
def encoder_tag(policy: Policy) -> str: ...

class DinoV2RegEncoder(nn.Module):
    LAYERS: tuple[int, ...] = (2, 5, 8, 11)
    def __init__(self, backbone: nn.Module, pca: PCABasis, *, precision: Literal["fp16", "fp32"] = "fp16") -> None: ...
    @classmethod
    def from_assets(cls, *, precision: str = "fp16", hub_version: str = DINOV2_HUB_PINNED,
                    device: str | torch.device = "cpu") -> "DinoV2RegEncoder": ...
    def encode_full(self, x: Tensor) -> Tensor: ...   # [B,3,H,W] -> [B,H/14,W/14,3072] fp32
    def encode_pca(self, x: Tensor) -> Tensor: ...    # [B,3,H,W] -> [B,H/14,W/14,1152] fp32
    def pca_inverse(self, z: Tensor) -> Tensor: ...   # [...,1152] -> [...,3072]
    def train(self, mode: bool = True) -> "DinoV2RegEncoder": ...  # always stays eval

# lib/irwm/worldmodel/reference.py  (moved from projects/dinof_audit/audit.py)
def one_step(model, x: Tensor) -> tuple[Tensor, float]: ...         # [B,5,h,w,1152] -> ([B,h,w,1152], loss)
def pca_metrics(pred: Tensor, gt: Tensor, beta: float) -> dict: ... # {"smoothl1", "mse", "cosine"}
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| `$IRWM_CKPT_ROOT/torch_hub/` (DINOv2 hub code `dinof_release`, `dinov2_vitb14_reg4_pretrain.pth`) | read | From F1 `download.py`. Present on ctcomp3, **not** on ctdesk236. |
| `$IRWM_CKPT_ROOT/dino_foresight/9174a31e/` (`pca_1152.pt`, `dinof_highres.ckpt`) | read | F1. The checkpoint is needed only by FR8 tests and the probe. |
| `third_party/DINO-Foresight` (`src/dino_f.py`, demo frames) | read (import) | Never modified (TechStack §2). |
| `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/` on ctcomp3 (`video_*_test`, `images_*_train`) | read | Must be copied to ctcomp3 first (deferred from F2; plan Step 7). |
| `lib/irwm/data/clip_sets/flir_test_fs5_c4f6_cs10` | read | F3. |
| Hydra run dir under `$IRWM_OUTPUT_ROOT/feature_extraction/` | write | `r5_probe.json`, `throughput.json`. |
| New deps | — | None in `lib`. `projects/feature_extraction/pyproject.toml` depends on `irwm[reference]` (the probe imports `Dino_f`). |
