# Performance benchmarks: short runs (2026-10-25)

Brief for the session that analyses the short runs. The overnight runs are in `bench-overnight-runs.md`.

## Why these benchmarks exist
- **New thesis direction (user, 2026-10-25).** There is no time left to train a DINO detector on IR (F8 is not built). The thesis instead shows **measured performance optimisations** of the pipeline on the cluster's worst-case GPU, the NVIDIA L4 (ctcomp3 `hpc-gpu5`/`hpc-gpu6`, 2 × L4 each, 48 CPUs, 375 GB RAM).
- **The three optimisations measured:**
  1. **Data on node-local scratch vs. BeeGFS.** `IRWM_STAGE_DATA=1` makes `slurm/job.sbatch` source `slurm/stage_data.sh`. This builds on Roadmap R11.
  2. **Fused attention in the DINOv2 encoder.** `irwm.features.encoder.set_attention(backbone, "sdpa")` swaps the hub code's plain attention (explicit `q @ kᵀ → softmax → @ v`, which is what the encoder and DINO-Foresight's reference really run, since xFormers is not installed) for `F.scaled_dot_product_attention`, using torch 2.2's FlashAttention-2 / memory-efficient backends. Weights are unchanged. The output is not bit-exact; `fidelity` measures the difference.
  3. **detrex's compiled CUDA MSDA vs. its pure-PyTorch fallback.** `head_bench.py msda=pytorch` replaces `detrex.layers.multi_scale_deform_attn.MultiScaleDeformableAttnFunction` at runtime with a shim that calls `multi_scale_deformable_attn_pytorch`. detrex itself is not modified.
- **Resolution policy:** everything defaults to **native** (Roadmap R5, decided 2026-10-25): centre crop to 5:4, then resize to 448×560, giving a 32×40 token grid.

## Prerequisites that are already done (Step 9 of `plan.md`)
- **CUDA build on the ctcomp3 login node:** `nvcc` 12.4 + gcc 8.5.0. The phase-2 build took 801 s. `opencv-python` was swapped for `opencv-python-headless` (see `plan.md ## Amendments`).
- **F7 GPU tests on hpc-gpu5: 6/6 passed, none skipped.**
  - Overfit passed at 200 iterations: loss ratio 0.0859, min IoU 0.951, min score 0.704.
  - It took 141.8 s and peaked at 7.63 GiB, on 32×64, fp32, batch 4, so about 0.7 s per train step.

## The short runs
These are crash checks before the overnight jobs. With 1–2 batches, **their timings mean nothing**. What to check is that each run reaches `results: ...json` and that the fidelity values are sane.

| # | Where | Command | What it checks |
|---|---|---|---|
| S1 | login node | `uv run --frozen pytest tests/test_attention_sdpa.py tests/test_encoder.py -q` | SDPA swap = plain attention on a copy of DINOv2's `Attention` (CPU, atol 1e-5). Switching back to `plain` is bit-exact. Bad input is rejected. The encoder Hydra group now defaults to `native`. |
| S2 | `srun` on hpc-gpu5, 8 CPUs, `WANDB_MODE=disabled` | `projects/feature_extraction/throughput.py "batch_sizes=[1]" e2e_warmup=1 e2e_batches=2 loader_batches=2` | Every part of the encoder benchmark runs end to end on real FLIR (native): GPU-only (plain/SDPA × fp16/fp32), `fidelity`, `end_to_end` per attention, `loader_only`. |
| S3 | `srun` on hpc-gpu5, `WANDB_MODE=disabled` | `projects/detection/head_bench.py msda=pytorch "batch_sizes=[1]" warmup=1 timed=1` | The PyTorch-MSDA shim works on CUDA in `train` and `infer` modes (32×40, fp32 and fp16). |

There was also an earlier S2 run under the old defaults (448×896 and 448×560, `stretch` end to end). It only checked the code path.

## Outputs (to fill in)
- **S1:** pytest summary line. *(pending: user will paste)*
- **S2:** console output and the `throughput.json` path.
  - Key lines: `[throughput] fidelity {...}`, with `mean_token_cosine` expected ≥ 0.999 and `max_abs_diff` reported;
  - `[throughput] e2e {...}` × 4 and `[throughput] loader {...}` × 2. *(pending)*
- **S3:** console output and the `head_bench.json` path. *(pending)*

## What the analysis session should do
1. Confirm S1–S3 succeeded. If anything failed, find the cause before trusting the overnight runs, which use the same code.
2. Read the S2 `fidelity` rows. They are the only real measurement in the short runs, because fidelity does not depend on batch count. Compare the fp16 SDPA-vs-plain drift with the fp16-vs-fp32 drift of the plain encoder from F4: max |Δ| 4.05e-1, mean |Δ| 3.37e-3, token cosine 0.999995 (`spec/2026-10-16-dinov2-reg-encoder-wrapper/results.md §1`, measured at 448×896). If SDPA's drift is of the same order or smaller, the accuracy cost of the speed-up is no worse than the fp16 the project already accepts.
3. Note any warnings that matter, for example an SDPA backend fallback.

## Code involved (uncommitted on 2026-10-25 until the user's push; check the `git` field in the JSONs)
- `lib/irwm/features/encoder.py`: `set_attention`, `_sdpa_forward`.
- `projects/feature_extraction/throughput.py` and `conf/throughput.yaml`:
  - keys `attentions`, `run_gpu_only`, `loader_batches`; `sizes` defaults to `[[448, 560]]`;
  - each end-to-end run and the loader-only run read **disjoint slices** of the train split, so none is helped by files an earlier one cached.
- `projects/detection/head_bench.py` and `conf/head_bench.yaml`: keys `msda` and `modes: [train, infer]`; `grids` defaults to `[[32, 40]]`; the JSON is saved after every row.
- `slurm/job.sbatch`: the `IRWM_STAGE_DATA=1` hook.
- `lib/irwm/conf/encoder/default.yaml`: `policy: native`.
- `tests/test_attention_sdpa.py` (new); `tests/test_encoder.py` (Hydra default changed to `native`).
