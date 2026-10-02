# F1 · DINO-Foresight audit: Plan

## Context and Design Decisions

- **The code is the source of truth** (Mission: "where the code and the paper disagree, the code wins"). The audit therefore runs the **unmodified** `Dino_f` class from the submodule with the hyperparameters saved inside the checkpoint (`save_hyperparameters()` stores the full argparse namespace). It never reconstructs them from the README, which already disagrees with the code: the README PCA command uses `--layers 2,5,9,11`, while the code, the PCA file name and the training commands all use `2,5,8,11`.
- **What a code read established before writing this spec** (to be confirmed numerically by `audit.py`):
  - DINOv2 is loaded with `torch.hub.load('facebookresearch/dinov2','dinov2_vitb14_reg')`, which pulls the **unpinned `main` branch** at runtime (`src/dino_f.py:139`).
  - Features come from `get_intermediate_layers(x, n=[2,5,8,11], reshape=False)` with DINOv2's default `norm=True`, so the final LayerNorm is applied to every block. CLS and the 4 register tokens are dropped (`dino_f.py:279`).
  - PCA works as follows: standardise by the stored `mean`/`std`, subtract `pca.mean_`, then project onto `components_`. There is no whitening (`dino_f.py:387-398`, `pca.py:128-134`).
  - The rollout writes the prediction into the last slot and shifts. Feedback stays in PCA space (`dino_f.py:511-513`).
  - Training randomised the frame skip over {1,2,3}, i.e. a stride of 2–4 frames at 17 Hz (`data.py:123`). Evaluation uses stride 3 with frames 7,10,13,16 → 19 (`data.py:151-157`).
  - Positional embeddings are factorised tables `d_0[5]`, `d_1[H]`, `d_2[W]`. The high-res checkpoint was itself made by bilinearly interpolating the 16×32 tables and fine-tuning (`dino_f.py:231-239`, `attention_masked.py:267-302`).
  - The only heads are DPT heads (segmentation, depth, normals), which consume PCA-reconstructed 3072-d features. There is no detection or instance head.
  - The checkpoint `state_dict` also contains `dino_v2.*` and the PCA tensors, so the exact weights used in training are recoverable.
- **Pinning DINOv2.** Because the reference resolves `main` at runtime, *which* DINOv2 code produced the training features is unknown. The hub repo changed positional-embedding interpolation in Feb 2024 (#378), and `main` has gained 14 commits since DINO-Foresight's release. The audit builds the backbone from three commits with identical weights and picks the one under which the reference model predicts best on real frames. A pre-populated `facebookresearch_dinov2_main` cache dir makes the reference code use the pinned version offline, without patching it.
- **Offline compute nodes.** Every network access happens in `download.py` on the login node. `audit.py` only reads `$IRWM_CKPT_ROOT`.
- **No scikit-learn downstream.** The released PCA is a pickled sklearn object. It is unpickled once (pinned `scikit-learn==1.5.2`, as in their `requirements.txt`) and exported to plain tensors. `irwm.features.pca.PCABasis` is pure torch and is tested against the reference.
- **Dependencies are isolated.** `pytorch_lightning`, `timm`, `einops` and `sklearn` are only needed to run *their* code. They go in an optional extra `irwm[reference]` used by `projects/dinof_audit`, and they are imported lazily inside `irwm.worldmodel.reference`, so `lib/irwm` keeps importing without them.
- **R5 probe.** No FLIR data exists yet, so the probe cuts FLIR's 5:4 aspect out of the in-domain Cityscapes demo clip. It isolates the geometry question (distortion vs lost field of view vs an untrained grid) from the domain shift. With a single clip the evidence is weak. The note says so, and F10 repeats the comparison on FLIR RGB.
- **The audit note is a product of the run.** `audit.md` is written after `audit.py` has produced `audit_results.json` on ctcomp3, citing numbers by key.

## Step 1 — Dependencies and project skeleton
- `lib/pyproject.toml`: add `[project.optional-dependencies] reference = ["timm>=0.9", "einops>=0.7", "scikit-learn==1.5.2"]`.
- `projects/dinof_audit/pyproject.toml`: `name = "irwm-dinof-audit"`, deps `["irwm[reference]", "huggingface_hub>=0.24", "matplotlib"]`, `package = false`.
- Root `pyproject.toml`: add `irwm-dinof-audit` to the dependencies and sources. Then `uv lock` and `uv sync`.

## Step 2 — `lib/irwm/worldmodel/assets.py`
Constants and path helpers as in FR1. The `pre378` sha is resolved once with the GitHub API (parent of `e1277af`) and hard-coded. `sha256_file` reads in 8 MiB chunks. `write_manifest` and `verify_manifest` follow the JSON schema in FR1.

## Step 3 — `lib/irwm/features/pca.py`
`PCABasis(nn.Module)` registers the buffers `mean, std, pca_mean, components`, stores `sha256`, and implements `transform` and `inverse` with shape validation (FR4).

## Step 4 — `lib/irwm/worldmodel/reference.py`
- `checkpoint_args`: `torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]["args"]`. A light path reads only the hparams. Note: loading a 3.4 GB checkpoint just for the args is acceptable once.
- `inference_args`: deep-copy, apply the overrides and return the diff (pure, testable).
- `load_reference`: insert the submodule root into `sys.path`, set the hub dir, import `src.dino_f.Dino_f` lazily, call `load_from_checkpoint(...)`, capture the keys via a custom `strict=False` load, then `eval()` and `requires_grad_(False)`.
- `load_demo_frames`: PIL → optional center crop → the reference transform.

## Step 5 — `projects/dinof_audit/download.py` (+ `export_pca` in `projects/dinof_audit/pca_export.py`)
For each source, `download → verify → manifest`. GitHub zips are extracted into `dinov2_code_dir(name)`, with `hubconf.py` at the root. Then `shutil.copytree(pinned, hub_dir/"facebookresearch_dinov2_main", dirs_exist_ok=True)`. Finally export the PCA (FR3).

## Step 6 — `projects/dinof_audit/audit.py`
A Hydra entry point with config keys `device`, `sections` (list, default all), `hub_versions` and `r5_policies`. Each FR7 section is a function `section_x(ctx) -> dict`. Results are merged into `audit_results.json` together with `git_info()` and the resolved config. Figures go to `<run_dir>/figures/`. The W&B logger is `make_wandb_logger`, and `wandb.log` is used for scalars and images. Implementation notes:
- Manual hooks for FR7.3: register a forward hook on `dino_v2.blocks[i]` for i in {2,5,8,11}, apply `dino_v2.norm`, drop `1 + num_register_tokens` tokens, and compare with `get_intermediate_layers`.
- Hub-version backbones: `torch.hub.load(str(dinov2_code_dir(name)), "dinov2_vitb14_reg", source="local", pretrained=False)` + `load_state_dict(official weights)`. The resulting features feed `model.pca_transform` and `model.forward`. The 1-step loss uses the reference masking path (`get_mask_tokens(mode="full_mask")`).
- R5 (c): a second `load_reference(img_size=(448,560))`. Its `inference_args` sets `high_res_adapt=True`, so their `on_load_checkpoint` interpolates `d_1`/`d_2`.
- Determinism: `torch.backends.cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True, warn_only=True)`.

## Step 7 — Tests (`tests/test_dinof_audit.py`; CPU, no weights)
Covered by the validation Groups 1–4. Real-weight checks are marked `@pytest.mark.data`.

## Step 8 — Run on ctcomp3 (user)
1. `uv run python projects/dinof_audit/download.py` on the login node.
2. `slurm/submit.sh ctcomp3 projects/dinof_audit/audit.py cluster.timeout_min=60 cluster.mem_gb=32`.

The user returns `audit_results.json`.

## Step 9 — Audit note and constitution updates
Write `audit.md` from the results (FR8). Then update the constitution:
- **TechStack:** §3.1 (hub pin, final norm, standardisation-before-PCA, resolution policy), §3.3 (stride range in training, rollout in PCA space, precision, positional-embedding interpolation), §4.2 (`encoder_tag` if R5 changes it), and the Component Registry (assets, `PCABasis`, reference loader).
- **Roadmap:** F1 → ✓; adjust F3 (stride), F4 (hub pin, `PCABasis`) and F10 (reference equivalence test) as the note dictates; close R1, decide R5/R7.
- **Memory:** record the decisions in [[world-model-pivot-dino-foresight]].

## Implementation Order
1. Deps and project skeleton
2. `assets.py`
3. `pca.py`
4. `reference.py`
5. `download.py` + `pca_export.py`
6. `audit.py`
7. Tests → `uv run pytest`
8. User runs download + audit on ctcomp3
9. `audit.md` + TechStack/Roadmap updates
