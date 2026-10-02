# F1 · DINO-Foresight audit (GATE): Requirements

## Goal

Before anything is built on DINO-Foresight, establish from the **released code and weights**, not the paper, exactly how features are produced and how the model is run. That means downloading and pinning every external artefact (HF checkpoints, PCA, DPT heads, the DINOv2 hub code and weights), running the reference code unchanged on real Cityscapes frames and on dummy inputs, and writing an audit note. The note answers Roadmap F1 (a)–(f), decides R5 (FLIR resolution policy) and R7 (feature-cache contents), and corrects `TechStack.md` §3.1/§3.3 wherever the code disagrees with the paper. F4 (encoder) and F10 (adapter) are gated on this note.

## Scope

In scope:
- A reproducible, offline-capable artefact layout under `$IRWM_CKPT_ROOT`. This covers all HF files at a pinned revision, the DINOv2 hub code at pinned commits, and the DINOv2-Reg ViT-B/14 weights, all with SHA-256 manifests.
- Export of the sklearn PCA pickle to plain tensors, so that nothing downstream needs scikit-learn.
- `irwm.features.pca.PCABasis`: a pure-torch transform/inverse that reproduces `Dino_f.pca_transform` / `pca_inverse_transform` exactly.
- `irwm.worldmodel.reference`: loads the **unmodified** reference `Dino_f` from the submodule with the checkpoint's own hyperparameters. This is the ground truth for F4/F10 equivalence tests.
- `projects/dinof_audit/`: `download.py` (login node) and `audit.py` (GPU job), which write `audit_results.json` and figures.
- The audit note `spec/2026-10-02-dino-foresight-audit/audit.md`, plus TechStack/Roadmap corrections.

Out of scope:
- Our own DINOv2 encoder wrapper (F4) and the rollout adapter/prediction cache (F10/F11). F1 only runs the reference code.
- Any FLIR data. The dataset isn't available yet; the R5 probe uses the 5 Cityscapes demo frames shipped in the submodule.
- Reproducing the paper's Cityscapes numbers (Mission: "We do not reproduce the Cityscapes results").
- Modifying the submodule. Their code is imported as-is.

## Functional Requirements

**FR1 — Pinned artefacts (`irwm.worldmodel.assets`).**
- Constants: `DINOF_HF_REPO = "Sta8is/DINO-Foresight"`, `DINOF_HF_REVISION = "9174a31e86c0ccd4df68c8c51f92ccce3313657a"`, and `DINOF_FILES` covering all 6 weight files (`dinof_highres.ckpt`, `dinof_lowres.ckpt`, `dinov2_pca_448_l[2_5_8_11]_1152.pth`, `head_{segm,depth,normals}_pca1152.ckpt`) plus `README.md`/`config.json`.
- `DINOV2_HUB_COMMITS: dict[str, str]` holds three named DINOv2 hub-code commits:
  - `pre378`: the parent of `e1277af` (before the positional-embedding interpolation fix #378, Feb 2024)
  - `dinof_release`: `85a24602099d397264d5b30461ad7f3bfd726ca1`, the last commit before DINO-Foresight's release on 2024-12-17
  - `main_2026`: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
  
  `DINOV2_HUB_PINNED` names the one the audit selects (FR7.5).
- `dinof_dir() -> Path` = `ckpt_root()/dino_foresight/<rev[:8]>`. `dinov2_hub_dir() -> Path` = `ckpt_root()/torch_hub`. `dinov2_code_dir(name) -> Path` = `dinov2_hub_dir()/dinov2@<sha[:8]>`. `dinov2_weights_path() -> Path` = `dinov2_hub_dir()/checkpoints/dinov2_vitb14_reg4_pretrain.pth`.
- `sha256_file(path) -> str` (streaming). `write_manifest(dir, files) -> Path` writes `manifest.json`, a list of `{file, bytes, sha256, source, revision}`. `verify_manifest(dir) -> list[str]` returns the files that are missing or have a mismatched hash; an empty list means OK.
- Downloads are idempotent: a file is skipped if it is present and its hash matches the manifest.

**FR2 — Download entry point (`projects/dinof_audit/download.py`, Hydra, run on a login node).**
- Fetches `DINOF_FILES` via `huggingface_hub.hf_hub_download(revision=DINOF_HF_REVISION, local_dir=dinof_dir())`.
- Fetches each `DINOV2_HUB_COMMITS` entry as a GitHub zip and extracts it to `dinov2_code_dir(name)`.
- Fetches the DINOv2 weights from `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth`.
- Creates `dinov2_hub_dir()/facebookresearch_dinov2_main` as a copy of the pinned commit. That is the cache dir name `torch.hub.load('facebookresearch/dinov2', ...)` resolves to, so the reference `Dino_f.__init__` runs offline on the pinned code.
- Writes the manifests, then runs FR3. A second run downloads nothing and exits 0.

**FR3 — PCA export.** `export_pca(pth_path, out_path)` (in the audit project, since it needs scikit-learn 1.5.2 to unpickle) writes `dinof_dir()/pca_1152.pt` containing `{"mean": [3072], "std": [3072], "pca_mean": [3072], "components": [1152, 3072]}` as float32 tensors, plus `{"whiten": bool, "explained_variance_ratio": [1152], "source_sha256": str}`. It raises `ValueError` if `whiten` is True, because the reference code does not apply whitening and the audit must flag that.

**FR4 — `irwm.features.pca.PCABasis`.**
- `PCABasis.load(path) -> PCABasis` (frozen buffers, fp32).
- `transform(x[..., 3072]) -> [..., 1152]` = `((x - mean) / std - pca_mean) @ components.T`.
- `inverse(z[..., 1152]) -> [..., 3072]` = `(z @ components + pca_mean) * std + mean`.
- `n_components`, `dim`, `sha256` properties.
- Both maps must equal the reference `Dino_f.pca_transform` / `pca_inverse_transform` on the same tensors, within fp32 tolerance `atol=1e-5` (checked in `audit.py`).
- Shapes are validated: a wrong last dim raises `ValueError`.

**FR5 — Reference loader (`irwm.worldmodel.reference`, lazy imports of `pytorch_lightning`, `timm`, `einops`, `sklearn`).**
- `checkpoint_args(ckpt_path) -> argparse.Namespace` returns `ckpt["hyper_parameters"]["args"]` exactly as saved.
- `inference_args(saved, *, pca_ckpt, img_size=None) -> Namespace` is a pure function on a copy of the args. It sets `eval_mode=False`, `eval_modality=None`, `head_ckpt=None`, `evaluate=False` and `vis_attn=False`, sets `pca_ckpt` to the local path, and sets `high_res_adapt = img_size is not None and img_size != saved.img_size`. If `img_size` is given it overrides it. Every field it changes is listed in a returned diff (`dict[str, (old, new)]`).
- `load_reference(ckpt_path, *, pca_ckpt, img_size=None, device) -> (Dino_f, diff)`:
  1. Prepends the submodule root to `sys.path`.
  2. Calls `torch.hub.set_dir(dinov2_hub_dir())`.
  3. Calls `Dino_f.load_from_checkpoint(ckpt, args=..., strict=False, map_location="cpu")`.
  4. Calls `.eval()`, sets `requires_grad_(False)` and moves the model to `device`.
  
  It records the `missing_keys`/`unexpected_keys`, and raises if any `maskvit.*`, `embed.*` or `replace_vector` key is missing.
- `load_reference` is the only place outside the submodule that imports `src.dino_f`.
- *Build-time changes (2026-10-02):*
  - **DINOv2 routing:** instead of relying on `torch.hub.set_dir` and the `facebookresearch_dinov2_main` cache dir (torch.hub still pings GitHub first, which can hang offline), `pinned_dinov2_hub()` redirects `torch.hub.load('facebookresearch/dinov2', ...)` to `source="local"` on the pinned copy for the duration of the call. The cache-dir alias is kept for running their notebook by hand.
  - **Missing args:** the checkpoint's saved args lack `sliding_window_inference` and `high_res_adapt`. `inference_args` fills fields like these from `train_py_defaults()`, which parses the released `train.py` with `ast` (it is never executed), and reports them as `("<missing>", value)`.
  - **New helpers:** `load_dinov2_backbone(version)` (purges the cached `dinov2` modules so several commits can be compared in one process) and `load_demo_frames(crop_aspect=[...])` (chained crops, needed for R5 policy b).

**FR6 — Demo inputs.** `load_demo_frames(img_size, crop_aspect=None) -> Tensor[1, 5, 3, H, W]` reads the 5 frames in `third_party/DINO-Foresight/.assets/demo/frankfurt_01_017101/` in sorted order (Cityscapes stride 3: 4 context frames + 1 target). It uses the reference eval transform `Resize(img_size) → ToTensor → Normalize(ImageNet)`. With `crop_aspect=(w, h)`, it first center-crops to that aspect (the largest such crop).

**FR7 — Audit run (`projects/dinof_audit/audit.py`, Hydra, GPU via `slurm/submit.sh`; CPU allowed but slow).** It writes `audit_results.json` and figures to the run dir and logs them to W&B. Sections:
1. **Artefacts:** `verify_manifest` is empty for all dirs. It also records the checkpoint's saved args, the parameter count per top-level module, and the state-dict key groups.
2. **Weights identity:**
   - max |Δ| between the checkpoint's `dino_v2.*` tensors and the official DINOv2 weights;
   - max |Δ| between the checkpoint's `pca_*`/`mean`/`std` and the `.pth` PCA;
   - `PCABasis` vs reference transform/inverse (FR4 tolerance).
3. **Feature recipe**, verified on demo frames at 448×896:
   - token count = 32·64 and feature dim = 4·768 = 3072;
   - `get_intermediate_layers(n=[2,5,8,11])` output equals manual hooks on `blocks[i]` followed by `model.norm`, with CLS and registers dropped (proves the final norm is applied and which tokens are kept);
   - z-score statistics of the standardised demo features (mean/std over tokens);
   - PCA reconstruction relative error `‖x − inv(T(x))‖² / ‖x − mean‖²` and explained variance on the demo frames.
4. **Forward and rollout (reference code):**
   - **1-step:** `model.sample(frames)` gives a prediction for frame 5. Report SmoothL1(β=0.1), MSE and mean cosine in PCA space against the true frame-5 features, alongside the **copy-last** baseline (frame-4 features). Check that the shapes are `[1,5,32,64,1152]` (PCA) and `[1,5,32,64,3072]` (postprocessed).
   - **3-step:** `model.sample_unroll(frames, unroll_steps=3)` must equal a manual loop of `forward` + shift in PCA space, with max |Δ| = 0 under fp32, documenting that feedback happens in PCA space.
   - **Masked-slot independence:** replacing the target-frame pixels with zeros or noise leaves the prediction unchanged (|Δ| = 0), which proves there is no leakage and that the adapter may pass any placeholder.
   - **Determinism:** two identical runs give |Δ| = 0.
   - **Precision:** fp32 vs `autocast(float16)` max/mean |Δ| and the change in SmoothL1.
   - **Dummy inputs:** random PCA-space context `[2, 5, 32, 64, 1152]` gives finite outputs of the right shape (batch > 1).
   - **Timing:** GPU time per 1-step forward.
5. **DINOv2 hub version:** for each `DINOV2_HUB_COMMITS` entry, build the backbone from that code with the same weights, then record:
   - max |Δ| of the demo features against `dinof_release`;
   - the 1-step SmoothL1 of the reference model when fed that version's features;
   - the z-score fit.
   
   `DINOV2_HUB_PINNED` is the version with the lowest 1-step SmoothL1. On ties (|Δ| features < 1e-5), it is `dinof_release`.
6. **R5 probe:** center-crop the demo frames to 5:4 (FLIR thermal aspect) and compare three policies. For each, report 1-step SmoothL1, cosine, copy-last SmoothL1 and the ratio `model/copy-last`:
   - (a) stretch to 448×896;
   - (b) crop the 5:4 frame to 2:1, then resize to 448×896;
   - (c) native resize to 448×560 (a 32×40 grid), with the model built at `img_size=(448,560)` so that the reference `on_load_checkpoint` interpolates the positional embeddings.
   
   The same crop and resize are applied to context and target. Also report the uncropped 2:1 reference ratio.
7. **Qualitative check:** the `head_segm_pca1152` DPT head on oracle vs predicted features for frame 5 (PNG, colour-mapped). It is visual only.

**FR8 — Audit note (`audit.md`).**
- One section per Roadmap question (a)–(f). Each answer cites `file:line` in the submodule and the relevant `audit_results.json` key.
- A **Paper vs code** table, and a **TechStack corrections** list that is applied to `TechStack.md`.
- **Decisions:**
  - R5 policy, with the probe numbers and an honest caveat (1 clip);
  - R7 cache contents;
  - the pinned DINOv2 hub commit;
  - inference precision;
  - the frame stride confirmed for F3.
- **Licences:** code MIT; HF weights MIT (model card); DINOv2 Apache-2.0.
- **Risks for F4/F10.**

## Public API Summary

```python
# irwm/worldmodel/assets.py
DINOF_HF_REPO: str; DINOF_HF_REVISION: str; DINOF_FILES: tuple[str, ...]
DINOV2_HUB_COMMITS: dict[str, str]; DINOV2_HUB_PINNED: str
def dinof_dir() -> Path: ...
def dinov2_hub_dir() -> Path: ...
def dinov2_code_dir(name: str) -> Path: ...
def dinov2_weights_path() -> Path: ...
def sha256_file(path: str | Path) -> str: ...
def write_manifest(directory: Path, entries: list[dict]) -> Path: ...
def verify_manifest(directory: Path) -> list[str]: ...

# irwm/features/pca.py
class PCABasis(nn.Module):
    @classmethod
    def load(cls, path: str | Path) -> "PCABasis": ...
    def transform(self, x: Tensor) -> Tensor: ...   # [..., 3072] -> [..., 1152]
    def inverse(self, z: Tensor) -> Tensor: ...     # [..., 1152] -> [..., 3072]
    n_components: int; dim: int; sha256: str

# irwm/worldmodel/reference.py
def checkpoint_args(ckpt_path: str | Path) -> Namespace: ...
def inference_args(saved: Namespace, *, pca_ckpt: str | Path,
                   img_size: tuple[int, int] | None = None) -> tuple[Namespace, dict]: ...
def load_reference(ckpt_path, *, pca_ckpt, img_size=None, device="cpu") -> tuple["Dino_f", dict]: ...
def load_demo_frames(img_size: tuple[int, int], crop_aspect: tuple[int, int] | None = None) -> Tensor: ...
```

## Dependencies

| Item | Direction | Notes |
|---|---|---|
| huggingface.co `Sta8is/DINO-Foresight` @ `9174a31e` | read | ~7.4 GB, login node only |
| github.com `facebookresearch/dinov2` @ 3 commits | read | zip archives, ~1 MB each |
| dl.fbaipublicfiles.com DINOv2-Reg ViT-B/14 weights | read | ~350 MB |
| `third_party/DINO-Foresight` | read (import) | unmodified; demo frames from `.assets/demo/` |
| `$IRWM_CKPT_ROOT/dino_foresight/9174a31e/` | write | HF files, `pca_1152.pt`, `manifest.json` |
| `$IRWM_CKPT_ROOT/torch_hub/` | write | `dinov2@<sha>/`, `facebookresearch_dinov2_main/`, `checkpoints/`, `manifest.json` |
| `$IRWM_OUTPUT_ROOT/dinof_audit/...` | write | `audit_results.json`, figures |
| New deps | — | `irwm[reference]` extra: `timm`, `einops`, `scikit-learn==1.5.2`; `huggingface_hub` in the audit project |
