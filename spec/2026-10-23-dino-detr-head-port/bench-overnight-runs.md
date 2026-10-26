# Performance benchmarks: overnight runs (2026-10-25)

Brief for the session that analyses the overnight runs. Context, the three optimisations, the code and the short crash checks are in `bench-short-runs.md`; read that first.

## Goal
Measure, on an NVIDIA L4 (the project's worst-case GPU) with the **native** policy (448×560 input, 32×40 tokens):
1. what **fused SDPA attention** buys the frozen DINOv2 encoder over the plain attention it runs today, in speed, memory and numerical drift;
2. what **staging FLIR on node-local scratch** buys over reading it from BeeGFS;
3. what **detrex's compiled CUDA MSDA** buys the DINO head over its pure-PyTorch fallback, for training and inference.

## The four jobs
Submitted from the ctcomp3 login node on 2026-10-25, each with `cluster.gres=gpu:L4:1` and `cluster.timeout_min=600`. Job ids follow submission order; confirm them in each log's first line.

| Job | SLURM id | Node | Command (after `bash slurm/submit.sh ctcomp3`) | Measures |
|---|---|---|---|---|
| **A** | 5525665 | hpc-gpu5 | `projects/feature_extraction/throughput.py cluster.cpus_per_task=8 cluster.mem_gb=32` | Encoder: GPU-only, fidelity, end-to-end and loader-only, **data on BeeGFS** |
| **B** | 5525666 | hpc-gpu5 | same + `run_gpu_only=false`, submitted with `IRWM_STAGE_DATA=1` | Encoder: fidelity, end-to-end and loader-only, **data on `$LOCAL_SCRATCH`** |
| **C** | 5525667 | hpc-gpu6 | `projects/detection/head_bench.py` | DINO head, **compiled CUDA MSDA**, train + infer |
| **D** | 5525668 | hpc-gpu6 | `projects/detection/head_bench.py msda=pytorch` | DINO head, **pure-PyTorch MSDA**, train + infer |

**What each job runs (defaults):**
- **A/B, GPU-only:** attentions plain and SDPA × precisions fp16 and fp32 × batch sizes 1, 16, 32, 64, on synthetic `[B, 3, 448, 560]` input. 5 warm-up + 20 timed batches; out-of-memory is recorded as `"oom"`.
- **A/B, fidelity:** SDPA vs. plain `encode_pca` on one random batch of 4, for each precision.
- **A/B, end-to-end:** FLIR train → `Preprocess("native")` in 8 DataLoader workers → fp16 `encode_pca`. Batch 32, 10 warm-up + 100 timed batches, once per attention, for RGB and then thermal. Each run reads its own slice of the split: plain frames 0–3519, SDPA frames 3520–7039.
- **A/B, loader-only:** the same DataLoader with no GPU work, on frames 7040–9279 (10 warm-up + 60 timed batches). This is the read + decode + resize rate of the data source on its own.
- **C/D:** modes train and infer × grid 32×40 × batch sizes 1, 2, 4, 8 × fp32 and fp16 autocast; 3 warm-up + 10 timed steps, median reported.
  - Train step: forward, backward, grad clip 0.1, AdamW.
  - Infer step: eval-mode forward under `no_grad`.
  - Synthetic features `[B, 1152, 32, 40]` with 5 random boxes per image.

## Outputs to collect
- `throughput.json` from A and B, and `head_bench.json` from C and D. They are in the Hydra run dirs under `$IRWM_OUTPUT_ROOT`; each log's last line prints the path.
- The SLURM logs in `$IRWM_OUTPUT_ROOT/slurm_logs/irwm-feature_extraction-<id>.out` and `irwm-detection-<id>.out`.
- The four W&B run ids.
- From B's log, the `[stage] copied N files in T s (X MB/s)` line: the cost of staging itself.

## Baselines already measured (for comparison)
- **F4 encoder, plain attention, L4, synthetic** (`spec/2026-10-16-dinov2-reg-encoder-wrapper/results.md §3`, 2 CPUs).
  - 448×560 fp16: 33.4 / 36.0 / 36.8 fps at B = 1 / 16 / 64 (2.6 GB at B = 16, 9.5 GB at B = 64).
  - 448×560 fp32: 17.8 / 18.9 / 18.9 fps (12.1 GB at B = 64).
  - At 448×896, fp16 B = 64 ran out of memory: the plain attention materialises the full score matrix.
  - Job A's plain rows should reproduce these numbers within a few percent. If they don't, something else changed.
- **F4 end-to-end** used `stretch` (448×896) with 2 workers: 17.1 fps, encode share 1.00, so the encoder was the bottleneck. There is **no earlier native end-to-end baseline**; job A's plain rows are the first.
- **R11 (Roadmap):**
  - `FlirSplit.load` (6 annotation files, about 260 MB) takes 176 s from BeeGFS vs. 5.7 s from scratch;
  - the full staging copy takes 276 s (12 GB, 44 MB/s);
  - a cold `import torch` from the BeeGFS `.venv` takes about 40 s.
- **F7 overfit test:** about 0.7 s per train step at 32×64, fp32, batch 4, compiled MSDA, peak 7.63 GiB.

## Questions the analysis must answer
1. **SDPA vs. plain (A, GPU-only):**
   - speed-up in fps for each precision and batch size;
   - peak memory reduction;
   - the largest batch that fits for each attention;
   - whether B = 64 fp32 fits now.
2. **Accuracy cost (A and B, fidelity):** SDPA's drift (max/mean |Δ|, token cosine) next to the plain encoder's own fp16-vs-fp32 drift from F4 (cosine 0.999995).
3. **Does SDPA move the bottleneck?** Compare `end_to_end.frames_per_s` with the GPU-only fps at B = 32 fp16, and look at `encode_share`.
   - An encode share near 1 means the GPU is the limit.
   - Well below 1 means data loading has become the limit, and that is where staging can help.
4. **BeeGFS vs. scratch (A vs. B):**
   - `loader_only.frames_per_s` per modality;
   - `end_to_end` with SDPA, A vs. B;
   - the staging copy time against the per-epoch time it saves. Epoch sizes: RGB 10,318 frames, thermal 10,742; `projected_epoch_s` is in the JSON.
5. **Compiled vs. PyTorch MSDA (C vs. D):** speed-up and memory ratio for train and infer at each batch size and precision. Note whether D runs out of memory where C doesn't.
6. **End-to-end inference cost of the whole detector:** time per frame = 1/encoder fps + 1/head infer fps (B = 8, fp16). Report the optimised configuration (SDPA + compiled MSDA + staged data) against the unoptimised one (plain + PyTorch MSDA + BeeGFS).
7. **A thesis-ready table and short text.** Wording rules:
   - Describe the baseline as "the pinned DINOv2 code's unfused attention fallback (no xFormers), as in our and, likely, DINO-Foresight's setup". Don't call SDPA a speed-up over DINOv2's intended xFormers setup.
   - Describe SDPA as "PyTorch fused SDPA (FlashAttention-2 / memory-efficient backends)".

## Caveats to keep in mind
- **Two jobs per node at the same time, on separate GPUs** (A + B on gpu5, C + D on gpu6). They use at most 16 of 48 CPUs and little RAM, so interference should be small, but say so.
- **B's staging copy (about 5 min, network) overlapped A's start.** A begins with the GPU-only part on synthetic data, so its BeeGFS reads probably started after the copy finished. Check: the time of B's `[stage] copied` line vs. A's first `[throughput] e2e` line in the logs.
- **B reads from page cache.** The node has 375 GB of RAM and the copy was just written, so B's reads probably come from memory, not the local disk. That is what a staged training job gets too, so the number is fair; call it "staged (page cache)".
- **A's BeeGFS reads may also be partly cached** by the BeeGFS client or by an earlier job on the node. The disjoint slices protect only against caching within the job.
- **D's fp16 isn't pure fp16 inside MSDA.** detrex casts `value` to fp32 before the op. Under autocast, the fallback's `grid_sample` also runs in fp32. Both paths compute the same thing.
- **All runs are single-GPU, single-process microbenchmarks on synthetic tensors**, except the end-to-end and loader-only parts, which read real FLIR. They do not measure training convergence or detection quality.

## If a job failed or timed out
- `head_bench.json` is saved after every row, and `throughput.json` after every section and every end-to-end run, so partial results are usable. W&B is logged only at the end.
- To rerun only the missing part: `run_gpu_only=false` skips the encoder's GPU-only section, and narrow `batch_sizes`, `modes`, `attentions` or `precisions` for the rest.
