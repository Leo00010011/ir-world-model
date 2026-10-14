# Tech Stack

This file lists every technology choice, where code lives, and the rules all features must follow. When a feature adds a class, cache layout or config group, it updates the **Component Registry** at the bottom.

## 1. Runtime environment

| Item | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Pinned in `.python-version`; DINO-Foresight is tested on 3.11 |
| Env / packaging | **uv** (workspace) | Root `pyproject.toml` + `uv.lock`. `uv sync` on the laptop and on the cluster gives identical envs. See §1.1 |
| Deep learning | PyTorch (CUDA build on the cluster, CPU build is fine locally) | DINO-Foresight pins torch 2.2.0 / torchvision 0.17.0 (cu121); we start from that and only move if needed |
| Training loop | **PyTorch Lightning** (`lightning`) | Every trainable model is a `LightningModule`; data comes from `LightningDataModule`s |
| Configuration | **Hydra** (`hydra-core`, OmegaConf) | Every entry point is a `@hydra.main` script; no argparse |
| Logging | **Weights & Biases** (`wandb`) through Lightning's `WandbLogger` | Project `ir-world-model`; W&B group = project folder name |
| HPO | **Optuna** through `hydra-optuna-sweeper` | Only when a spec asks for it |
| Cluster | **SLURM** through `hydra-submitit-launcher` plus plain `sbatch` templates in `slurm/` | See §5 |
| Feature storage | **HDF5** (`h5py`) | See §4 |
| Detection metrics | `pycocotools` (COCO eval), with our own wrapper for the 6-class label space with ignore regions (F2), per-horizon breakdown and bootstrap CIs | |
| Testing | `pytest` | Run locally with `py -m pytest tests/...` (Windows) or `uv run pytest` |
| Local OS | Windows 11 (development) | Code must be path-agnostic (`pathlib`) and must not assume POSIX-only tools |
| Cluster OS | Linux + SLURM | |

### 1.1 Installing on the external server
- uv needs no admin rights. Install it in user space with `pip install --user uv` (or the official standalone installer, which writes to `~/.local/bin`).
- Build the env on a node with internet (usually the login node). If compute nodes are offline, they then only use the existing `.venv`.
- PyTorch's CUDA build (cu121, matching DINO-Foresight) is selected through a `[tool.uv.sources]` / index entry in `pyproject.toml`, so `uv sync` installs the CUDA wheels on Linux and the CPU wheels on the Windows laptop.

## 2. Repository layout (monorepo)

```
ir-world-model/
├── pyproject.toml              # uv workspace root (members: lib, projects/*)
├── uv.lock
├── spec/
│   ├── constitution/           # Mission.md, TechStack.md, Roadmap.md (this folder)
│   └── YYYY-MM-DD-<feature>/   # one folder per feature: requirements.md, plan.md, validation.md
├── lib/                        # shared code, installed editable as package `irwm`
│   ├── pyproject.toml
│   └── irwm/
│       ├── data/               # FLIR readers, clip builders, class mappings, transforms
│       ├── features/           # DINOv2 encoder wrapper + PCA, HDF5 feature cache read/write
│       ├── detection/          # DINO-DETR head (nn.Module), losses, matcher, box utils
│       ├── worldmodel/         # DINO-Foresight adapter (load ckpt, rollout API), prediction cache
│       ├── eval/               # COCO eval wrapper, per-horizon metrics, bootstrap CIs
│       └── utils/              # paths/env resolution, seeding, logging, SLURM helpers
├── projects/                   # one folder per research project; entry points + configs only
│   ├── smoke/                  # permanent env/cluster health check (toy Lightning run)
│   ├── data_stats/             # F2: FLIR dataset statistics / class-imbalance report
│   ├── clip_index/             # F3: build the committed video clip set + count report
│   ├── dinof_audit/            # F1: download/pin DINO-Foresight artefacts, audit the reference code
│   ├── feature_extraction/     # build the DINOv2 caches
│   │   ├── conf/               # Hydra configs
│   │   └── *.py                # @hydra.main entry points
│   ├── detection/              # train/eval DINO-DETR heads on cached features
│   ├── wm_eval/                # DINO-Foresight zero-shot rollouts + evaluation
│   └── wm_adaptation/          # Phase 4 unsupervised adaptation experiments
├── third_party/
│   └── DINO-Foresight/         # git submodule (Sta8is/DINO-Foresight), pinned commit, NOT modified in place
├── slurm/                      # sbatch templates shared by all projects
├── notebooks/                  # analysis only; never imported by code
└── tests/                      # pytest, one file per feature: tests/test_<feature>.py
```

Rules:
- **Reusable logic goes in `lib/irwm`.** Project folders hold only Hydra entry points, configs and project-specific glue. If two projects need the same function, it moves to `lib/`.
- A project folder may later get its own `spec/`-style constitution once it grows large enough.
- **Hydra wiring:** shared config groups live in `lib/irwm/conf/`. A project's primary config uses `defaults: [irwm_base, _self_]` and `hydra.searchpath: ["pkg://irwm.conf"]`, and sets `project: <folder name>`. Every entry point calls `irwm.utils.paths.load_env()` at import time, before `@hydra.main`. `run_name` comes from the cluster group (a timestamp locally, `slurm-<job id>` on SLURM, which is stable across requeues). Checkpoints go to `${paths.ckpt_root}/<project>/${run_name}/`, with the trainer's `default_root_dir` set to that dir, and training resumes with `resume_ckpt_path()`.
- **Reference-only deps:** `pytorch_lightning`, `timm`, `einops` and `scikit-learn==1.5.2` are needed only to run DINO-Foresight's own code. They live in the optional extra `irwm[reference]` (used by `projects/dinof_audit`) and are imported lazily, so `lib/irwm` imports without them. A test enforces that there are no module-level imports.
- **Notebooks** (`notebooks/`) need the root dependency group `notebook` (`ipykernel`, `matplotlib`): `uv sync --group notebook`. A plain `uv sync` removes it again. Notebooks read results from `$IRWM_OUTPUT_ROOT` through `irwm.utils.paths`, never from hard-coded paths.
- **Dependencies:** deps shared by all projects (torch, lightning, hydra, wandb, …) are declared in `lib/pyproject.toml`. Each feature adds its own deps there, or to its project's `pyproject.toml` if only that project needs them.
- `lib/irwm` must never import from `projects/`.
- Third-party code (DINO-Foresight) is used only through an adapter in `irwm/worldmodel/`. We do not patch the submodule. If a change is unavoidable, it is kept as a patch file with its reason recorded in the spec.

## 3. Models and fixed hyperparameters

### 3.1 Encoder: DINOv2 (frozen)
Fixed to match DINO-Foresight. Items marked *(F1)* are confirmed by the F1 audit (`spec/2026-10-02-dino-foresight-audit/audit.md`).
- Backbone **DINOv2-Reg ViT-B/14** (`vitb14_reg`), frozen, `eval()`, no grad. Weights are the official `dinov2_vitb14_reg4_pretrain.pth`, bit-identical to the copy inside the DINO-Foresight checkpoint *(F1)*. The model is built from the DINOv2 hub code **pinned to `dinof_release` (`85a24602`)**, because the reference loads the unpinned `main` at runtime *(F1)*. Paths come from `irwm.worldmodel.assets` and env vars (§6).
- Input **448×896** (32×64 patch tokens), the size of the high-res checkpoint. ImageNet mean/std normalisation. How FLIR's 5:4 frames are mapped to the model input (stretch, crop, native grid or two windows) is **still open**. It is decided at training time, before F5, from the F1 probe plus a FLIR check (Roadmap R5). Until then the encoder (F4) must take the policy as a config parameter, not hard-code it. Below, the token grid is written `H'×W'` (default 32×64).
- **Features = channel concatenation of the patch-token outputs of blocks 3, 6, 9, 12** (1-indexed; `d_layers 2,5,8,11` in the reference code) → **3072-d**. The CLS token and the 4 register tokens are dropped (2053 → 2048 tokens). **The final LayerNorm is applied to every one of the four blocks** (DINOv2 `get_intermediate_layers` default `norm=True`) *(F1)*.
- **PCA:** DINO-Foresight's released PCA (`dinov2_pca_448_l[2_5_8_11]_1152.pth`, fit on Cityscapes train) reduces 3072 → **1152-d**. Applied as: standardise each dim by the stored `mean`/`std`, subtract `pca_mean`, then project on `components`. **No whitening** *(F1)*. Implemented in pure torch by `irwm.features.pca.PCABasis` (equal to the reference, |Δ| = 0). This is the space the world model reads and writes. The PCA is part of the frozen encoder and is never re-fit outside Phase 4.
- **Thermal frames** are single-channel and get **replicated to 3 channels** before normalisation. This choice is written down because it affects every IR result.

### 3.2 Detector: DINO-DETR head (ported)
- DINO-DETR (Zhang et al., 2022) is reimplemented or vendored as a plain `nn.Module` in `irwm/detection/`. There is **no detectron2/detrex runtime dependency**. (DINO-Foresight reports instance segmentation with Mask2Former, but FLIR has boxes only, so we use a box detector.)
- Input is cached features `[B, 1152, H', W']` (PCA coefficients, single scale). The PCA inverse is a fixed linear map, so a learned input projection on the coefficients is equivalent to one on reconstructed features, and we skip reconstruction. The backbone is not part of the module.
- Trained with Lightning + Hydra. **One head per modality**: `det_rgb` and `det_ir`.
- The head is trained on `train` and selected on `val`. It is then **frozen** and reused, unchanged, for oracle and world-model evaluation.
- The port is validated once against the detrex reference config: same architecture hyperparameters, and comparable AP on a small run.

### 3.3 World model: DINO-Foresight (frozen, zero-shot)
- Official code from `Sta8is/DINO-Foresight` (MIT), as a submodule. The checkpoint is `dinof_highres.ckpt` from Hugging Face `Sta8is/DINO-Foresight`. **Used zero-shot: never retrained in Phases 0–3.**
- A masked feature transformer (12 layers, hidden 1152, 8 heads, separable space–time attention) that predicts **1 future frame from 4 context frames** in PCA space. It is **deterministic** (SmoothL1 regression).
- Temporal protocol: training sampled a random step of 2–4 frames at 17 Hz (0.118–0.235 s); evaluation uses a fixed step of 3 (≈ **0.176 s**) *(F1)*. For FLIR at 30 FPS we use **stride 5 (0.167 s)**, inside the training range *(F1)*. The 30 FPS comes from the FLIR README and cannot be verified, because the test split has no per-frame timestamps *(F3)*. Longer horizons come from an **autoregressive rollout in PCA space**: the prediction overwrites the masked last slot, then the window shifts by one *(F1)*. Short-term = 1 step, mid-term = 3 steps; we evaluate up to 6 steps.
- Inference precision: **fp16 autocast**, as in the reference `16-mixed` (SmoothL1 changes by 0.01 %). Equivalence tests run in fp32, where the reference is reproduced exactly *(F1)*.
- Positional embeddings are factorised tables (time, height, width). A token grid other than 32×64 is supported by loading with `high_res_adapt=True`, which bilinearly interpolates the height/width tables *(F1)*.
- The adapter exposes `rollout(context_feats[4, H', W', 1152], n_steps) -> [n_steps, H', W', 1152]`.
- **Licences:** DINO-Foresight code is MIT and DINOv2 is Apache-2.0. The released weights are MIT (HF model card) *(F1)*. Research use only; do not redistribute weights.

### 3.4 Canonical label space *(F2)*
One label space in `irwm/data/classes.py`; every project imports it and nobody redefines it.
- **6 classes:** `CLASS_NAMES = ("person", "bike", "car", "motor", "light", "sign")`, labels `0..5` by ascending FLIR category id (`CATEGORY_ID_TO_LABEL = {1: 0, 2: 1, 3: 2, 4: 3, 10: 4, 12: 5}`). FLIR ids and names are always read from `coco.json` (the README's id list is wrong for `scooter`).
- **Ignore regions:** `label_of(id) is None` for every other FLIR class. They are not targets and not background. In COCO GT (`FlirSplit.to_coco_gt`) each ignore box is emitted once per kept category with `iscrowd: 1`, because `COCOeval` matches crowd regions only within a category.
- **No coarse mapping** (vehicle/pedestrian/cyclist) exists anywhere.
- **Size buckets:** `normalised_area = w·h·(640·512)/(W·H)`, then COCO thresholds 32² / 96² (`size_bucket`). Thermal (640×512) is unchanged; RGB frames are larger, so native buckets would compare different objects.

## 4. Data and cache layout

### 4.1 Raw data
FLIR ADAS v2 as downloaded, under `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/` (*F2*; the zip's misspelt folder name was corrected by the user). It is read-only; code never writes there. Only `irwm.data.flir` (`FLIR_DIRNAME`) and `conf/flir/default.yaml` (`flir.root`, in `irwm_base`) name the folder.
```
teledyne_flir_adas_v2/
├── rgb_to_thermal_vid_map.json          # RGB test basename → thermal test basename (3749 entries)
└── FLIR_ADAS_v2/<split_dir>/            # images_{rgb,thermal}_{train,val}, video_{rgb,thermal}_test
    ├── coco.json                        # the annotation source (index.json only for test videos[].tags)
    ├── index.json
    └── data/video-<video_id>-frame-<6 digits>-<uid>.jpg
```
`image_id` is unique only within one (modality, split); lookups go through `FlirSplit`. The 16-bit `analyticsData` TIFFs are not used.

### 4.2 Feature cache (HDF5)
Location: `$IRWM_CACHE_ROOT/features/<encoder_tag>/<modality>/<split>.h5`, where `encoder_tag = dinov2_vitb14reg_448x896_l3-6-9-12_pca1152` (the resolution part changes if F1 picks another policy).

Size check: 32·64·1152 fp16 ≈ 4.7 MB per frame. With 30,787 frames (23,289 annotated image frames plus 7,498 video frames, F2), the total is about **145 GB**. Full 3072-d features would be about 390 GB, so they are **not** cached by default. They are encoded on the fly for the PCA-loss analysis on a subset (Roadmap R7).

Minimal layout (the feature spec fixes the details):
```
/feats        float16  [N, H', W', 1152]  chunks=(1, H', W', 1152)   # PCA coefficients
/image_id     int64    [N]                COCO image id, same order as /feats
/file_name    str      [N]
attrs: encoder_tag, layers, resolution, resize_policy, dinov2_weights_sha, pca_sha, git_commit, created_at
```
Invariants: rows are in a deterministic order (sorted by `image_id`). Every row matches exactly one COCO image. Readers look up by `image_id` and never by position alone.

### 4.3 Prediction cache (HDF5)
Location: `$IRWM_CACHE_ROOT/predictions/<wm_tag>/<modality>/<clip_set>.h5`, with `<clip_set>` a committed clip set read through `irwm.data.clips.ClipIndex.load()` (default `flir_test_fs5_c4f6_cs10`, F3). Rows follow `clip_id`. It stores predicted future features `[n_clips, n_steps, H', W', 1152]` fp16 (deterministic, so there is no sample axis), plus per-clip context and target frame ids so every predicted frame can be matched to its ground-truth annotations. The WM-eval spec defines the exact layout.

### 4.4 Checkpoints and outputs
- Checkpoints go in `$IRWM_CKPT_ROOT/<project>/<run_name>/`. Lightning `last.ckpt` is always saved so SLURM requeues can resume.
- Hydra run dirs go in `$IRWM_OUTPUT_ROOT/<project>/${now:%Y-%m-%d}/${now:%H-%M-%S}`, and multirun sweeps under `.../multirun/`.

## 5. SLURM rules
- Every entry point runs headless, takes no interactive input, and is fully described by its Hydra config.
- Launch through `slurm/submit.sh <site> <entry.py>` (single jobs) or `-m cluster=<site>` (submitit sweeps). Partition, account, GPU type and time come from **one Hydra file per cluster**, `lib/irwm/conf/cluster/<site>.yaml`, which inherits `cluster/slurm.yaml`. `submit.sh` turns it into sbatch flags, so resources are never duplicated in `#SBATCH` lines or code.
- New cluster: add a site file, then run `bash slurm/setup_env.sh` on its login node (installs uv in user space, syncs, verifies).
- Jobs must **resume after preemption**: Lightning `ckpt_path="last"`, requeue signal handling (`SLURMEnvironment(auto_requeue=True)`), and idempotent cache writers that skip or append finished rows.
- Data loading must be efficient on shared filesystems: open HDF5 lazily per worker (not in `__init__`), and use large chunked reads.
- Multi-GPU goes through Lightning DDP only. Code must not assume a single GPU or rank 0.

## 6. Environment variables (only these may hold paths)
| Variable | Meaning |
|---|---|
| `IRWM_DATA_ROOT` | Raw datasets (read-only) |
| `IRWM_CACHE_ROOT` | Feature and prediction caches |
| `IRWM_CKPT_ROOT` | Model checkpoints (ours + downloaded DINOv2 / DINO-Foresight weights and PCA) |
| `IRWM_OUTPUT_ROOT` | Hydra run dirs, eval JSONs, figures |
| `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` | W&B; `WANDB_MODE=offline` is supported for compute nodes without internet |
| `WANDB_API_KEY_FILE` | File holding the W&B key (default `~/.wandb_api_key`); `slurm/env.sh` exports `WANDB_API_KEY` from it when unset |

Hydra configs read these through `${oc.env:IRWM_CACHE_ROOT}`. A local `.env` (git-ignored) sets them on the laptop. **Put `.env` comments on their own line, never after a value**: VS Code's Python extension injects `.env` into Jupyter kernels with a parser that keeps inline comments as part of the value, and `load_env()` does not override variables that are already set.

## 7. Coding constraints
- Type hints on public functions. Document shapes as `[B, C, H, W]` in docstrings.
- Feature tensors are channels-last `[..., H', W', 1152]` on disk and in caches. They are converted to `[B, 1152, H', W']` only at the detector input.
- Storage is fp16. Compute is fp32 unless a spec says otherwise. DINO-Foresight was trained in 16-mixed precision, and the adapter follows whatever F1 finds the reference inference uses.
- Seeds come from Hydra config and are set through `lightning.seed_everything`. World-model inference is deterministic. The adapter sets `torch.backends.cudnn.deterministic` where it matters.
- Every eval writes a machine-readable JSON (metrics + config + git commit) next to its W&B log.
- Tests use tiny synthetic data, run on CPU in under a minute, and do not need the real dataset, cached weights or GPUs. Tests that need real data are marked `@pytest.mark.data` and skipped by default.

## 8. Component Registry
Updated by every feature when it is marked done. Status: ☐ planned · ◐ in progress · ✓ done.

| Component | Location | Feature | Status |
|---|---|---|---|
| uv workspace, `irwm` package skeleton, env-var path resolution | `pyproject.toml`, `lib/irwm/utils/paths.py` | F0 | ✓ |
| Git provenance + W&B logger factory (`git_info`, `make_wandb_logger`) | `lib/irwm/utils/tracking.py` | F0 | ✓ |
| SLURM helpers (`on_slurm`, `cluster_plugins`, `resume_ckpt_path`, `sbatch_args`; CLI `python -m irwm.utils.slurm <site>`) | `lib/irwm/utils/slurm.py` | F0 | ✓ |
| Shared Hydra groups (`irwm_base`, `paths`, `cluster/{local,slurm}`, `wandb`) | `lib/irwm/conf/` (`pkg://irwm.conf`) | F0 | ✓ |
| Smoke project (Hydra + Lightning + W&B + requeue health check) | `projects/smoke/` | F0 | ✓ |
| Cluster scripts: `setup_env.sh` (per-cluster env bootstrap), `submit.sh` (site config → sbatch), `job.sbatch`, `env.sh` (.env + W&B key file) | `slurm/` | F0 | ✓ |
| Per-cluster site configs (template `example_site.yaml`; sites: `ctcomp3`) | `lib/irwm/conf/cluster/<site>.yaml` | F0 | ✓ |
| DINO-Foresight submodule @ `d98f765` | `third_party/DINO-Foresight` | F0 | ✓ |
| FLIR ADAS v2 readers: `FlirSplit`, `FlirRecord`, `FlirImageDataset`, `to_coco_gt` (ignore regions as crowd GT), `load_rgb_thermal_map`, `video_tags` | `lib/irwm/data/flir.py` | F2 | ✓ |
| Canonical label space (6 classes, ignore rule, 640×512-normalised size buckets) | `lib/irwm/data/classes.py` | F2 | ✓ |
| Shared Hydra group `flir` (`flir.root`) | `lib/irwm/conf/flir/default.yaml` | F2 | ✓ |
| Dataset statistics (`compute_stats`, `render_markdown`, `COCO80_NAME`) + entry point | `lib/irwm/data/stats.py`, `projects/data_stats/` | F2 | ✓ (ctdesk236) |
| Stats notebook | `notebooks/flir_stats.ipynb` | F2 | ✓ |
| Video clip index: `ClipIndex`, `Clip`, `check_pairing`, `find_segments`, `build_clip_index`, `dump_clip_index`/`write_clip_index` (never overwrites a differing file), `summarise` | `lib/irwm/data/clips.py` | F3 | ✓ |
| Committed clip set `flir_test_fs5_c4f6_cs10` (330 clips, 304 evaluable; shared RGB/thermal rows) + builder entry point | `lib/irwm/data/clip_sets/`, `projects/clip_index/` | F3 | ✓ (ctdesk236) |
| DINOv2 encoder wrapper + PCA | `lib/irwm/features/encoder.py` | F4 | ☐ |
| HDF5 feature cache | `lib/irwm/features/cache.py` | F5 | ☐ |
| DINO-DETR head | `lib/irwm/detection/` | F7 | ☐ |
| COCO evaluator (6-class AP/AP50 with ignore regions, per-horizon, CIs) | `lib/irwm/eval/` | F6 | ☐ |
| DINO-Foresight adapter + prediction cache | `lib/irwm/worldmodel/` | F10–F11 | ☐ |
| Pinned artefacts (HF rev `9174a31e`, DINOv2 hub commits, SHA-256 manifests, `$IRWM_CKPT_ROOT/{dino_foresight,torch_hub}` layout) | `lib/irwm/worldmodel/assets.py` | F1 | ✓ (manifests verified on ctcomp3) |
| `PCABasis`: DINO-Foresight PCA in pure torch (`transform`/`inverse`, 3072↔1152), loaded from the exported `pca_1152.pt` | `lib/irwm/features/pca.py` | F1 | ✓ (equals the reference, \|Δ\| = 0, ctcomp3) |
| Reference loader: unmodified `Dino_f` with the checkpoint's args, DINOv2 routed to the pinned local copy, `train_py_defaults` for missing args, demo-frame loader | `lib/irwm/worldmodel/reference.py` | F1 | ✓ |
| Audit project (`download.py`, `pca_export.py`, `audit.py`) | `projects/dinof_audit/` | F1 | ✓ (`hub`, `segm` sections implemented but not run) |
