# F7 · DINO-DETR head port: Results (2026-10-26)

Results note per FR11, written from the user's ctcomp3 runs of 2026-10-25 (plan.md Step 9) and the overnight benchmarks briefed in `bench-short-runs.md` / `bench-overnight-runs.md`. Raw outputs: `outputs/` (Hydra run dirs + `slurm_logs/`), analysed in `notebooks/bench_short_runs.ipynb`.

**Provenance.** Every GPU run used commit `9b5a7ff` with uncommitted changes (`git.dirty = true` in every JSON): the SDPA / MSDA-shim / staging code listed in `bench-short-runs.md` "Code involved", committed afterwards in `ca02d8e`. detrex @ `e244e6c`, detectron2 @ `70f4543`, torch `2.2.0+cu121`, GPU NVIDIA L4 (24 GB) on hpc-gpu5 / hpc-gpu6.

## 1. CUDA build on ctcomp3
- Toolchain: **`nvcc` 12.4 + gcc 8.5.0**, on the ctcomp3 login node with `FORCE_CUDA=1` (plan.md Amendments, 2026-10-25).
- Phase-2 build (detectron2 + detrex, CUDA ops): **801 s**.
- Fix needed on the way: `opencv-python` → `opencv-python-headless` (nodes lack `libGL.so.1`; plan.md Amendments).
- `cuda_ops_available()` is `true` in both `head_bench.json` files (C and D).

## 2. GPU test suite (`pytest tests/test_detection_head.py -m gpu`, hpc-gpu5)
- **6 passed, 0 skipped** (Groups 7–8: 5 CUDA/loss/fp16 tests + the overfit). Reported by the user; the full console output was not kept.

## 3. Overfit (validation Group 8)
32×64 grid, full default config, fp32, batch 4, 4 synthetic images / 11 boxes.

| Iterations to pass | Loss ratio (final / step 0) | Min IoU | Min score | Wall time | Peak memory |
|---|---|---|---|---|---|
| **200** (bar ≤ 1500) | **0.0859** (bar ≤ 0.10) | **0.951** (bar ≥ 0.8) | **0.704** (bar ≥ 0.5) | 141.8 s | 7.63 GiB |

About 0.7 s per train step, consistent with job C's 0.49 s at 32×40 fp32 B = 4 (1.6× fewer tokens).

## 4. Head benchmark (FR10, amended 2026-10-26)
Job C (SLURM 5525667, hpc-gpu6, W&B `4c11kfnm`): `outputs/detection/2026-10-25/13-30-17/head_bench.json`, compiled CUDA MSDA. Job D (SLURM 5525668, hpc-gpu6, W&B `64y1udph`): `.../13-30-40/head_bench.json`, `msda=pytorch` (detrex's pure-PyTorch fallback through a runtime shim; detrex unmodified). Grid 32×40 (native), synthetic features `[B, 1152, 32, 40]`, 5 random boxes per image, 3 warm-up + 10 timed steps, median. Train step = forward + backward + clip 0.1 + AdamW; infer = eval forward under `no_grad`. fp16 = CUDA autocast. 16/16 rows, **no OOM** in either job.

| mode | precision | batch | median step s | images/s | peak GiB | PyTorch MSDA images/s | PyTorch MSDA peak GiB | speed-up |
|---|---|---|---|---|---|---|---|---|
| train | fp32 | 1 | 0.137 | 7.3 | 1.60 | 4.9 | 2.31 | 1.49× |
| train | fp16 | 1 | 0.093 | 10.7 | 1.42 | 6.2 | 2.17 | 1.73× |
| train | fp32 | 2 | 0.249 | 8.0 | 2.93 | 5.3 | 4.37 | 1.51× |
| train | fp16 | 2 | 0.156 | 12.8 | 2.53 | 7.1 | 4.06 | 1.81× |
| train | fp32 | 4 | 0.487 | 8.2 | 5.52 | 5.3 | 8.45 | 1.54× |
| train | fp16 | 4 | 0.302 | 13.2 | 4.71 | 7.0 | 7.79 | 1.89× |
| train | fp32 | 8 | 0.977 | 8.2 | 10.78 | 5.2 | 16.57 | 1.57× |
| train | fp16 | 8 | 0.612 | 13.1 | 9.07 | 6.7 | 15.22 | 1.94× |
| infer | fp32 | 1 | 0.033 | 30.3 | 0.43 | 15.4 | 0.68 | 1.97× |
| infer | fp16 | 1 | 0.018 | 56.2 | 0.43 | 20.3 | 0.70 | 2.77× |
| infer | fp32 | 2 | 0.064 | 31.1 | 0.52 | 16.2 | 1.08 | 1.92× |
| infer | fp16 | 2 | 0.037 | 54.7 | 0.53 | 20.2 | 1.11 | 2.71× |
| infer | fp32 | 4 | 0.134 | 30.0 | 0.76 | 15.8 | 1.89 | 1.90× |
| infer | fp16 | 4 | 0.075 | 53.2 | 0.74 | 19.7 | 1.91 | 2.70× |
| infer | fp32 | 8 | 0.275 | 29.1 | 1.26 | 15.4 | 3.50 | 1.89× |
| infer | fp16 | 8 | 0.160 | 50.1 | 1.17 | 19.1 | 3.52 | 2.62× |

(`peak GiB` = `peak_mem_gb`, from `torch.cuda.max_memory_allocated`.)

- **Throughput is flat from batch 2.** Compiled MSDA trains at ≈ 8 img/s in fp32 and **≈ 13 img/s in fp16**. Batch 8 fp16 uses 9.1 GiB, so batch 16 likely fits in an L4.
- **Compiled vs PyTorch MSDA:** 1.5–1.6× faster training in fp32, 1.7–1.9× in fp16, with 0.60–0.65× the memory. Inference is 1.9× faster in fp32 and 2.6–2.8× in fp16. The fallback gains little from fp16 (detrex casts `value` to fp32, and `grid_sample` runs in fp32 under autocast). This confirms the spec's choice of the compiled op.
- **The head bounds F8's step, not the encoder.** At batch 8 fp16, the head trains at 13.1 img/s. The frozen encoder runs, end to end on real FLIR, at 36.7 fps with plain attention and 89 fps with SDPA (job A, §6). F4 measured 36 fps at 448×560. So the head is ≈ 2.8× slower than the plain encoder and ≈ 6.8× slower than SDPA: ≈ 76 ms of head per frame against 11 ms of encoder. A rough sequential estimate per epoch per modality, at native, fp16, compiled MSDA + SDPA, is RGB 10,318 frames × (1/13.1 + 1/89.3) s ≈ **15 min**, of which ≈ 87 % is the head. This assumes the bench's 5 boxes per image; real FLIR frames carry more, which makes matching slower.

## 5. Parameter count
`n_params` in both JSONs: **23,001,286** (default head, C = 1152), equal to the spec-time probe. The 3072-d variant (24,479,686) is checked on CPU (Group 4).

## 6. Related benchmarks run under this spec (not F7 FRs)
The same campaign measured two encoder optimisations for the thesis (`bench-*.md`). Full analysis in `notebooks/bench_short_runs.ipynb`; key numbers:
- **Fused SDPA in the DINOv2 encoder** (job A, SLURM 5525665, hpc-gpu5, W&B `6o47ocr3`; `outputs/feature_extraction/2026-10-25/13-28-51/throughput.json`):
  - fp16: 37.0 → 90.4 fps at B = 64 (**2.44×**), peak memory 9.49 → 3.34 GB;
  - fp32: 18.9 → 24.8 fps (1.32×), 12.09 → 3.82 GB.
  - Plain rows reproduce F4 within 1.5 % from B = 16 up.
  - Drift, SDPA vs plain, fp16, random batch of 4: mean token cosine 0.9999982, mean |Δ| 2.3e-3, max |Δ| 0.047. All are below the plain encoder's own fp16-vs-fp32 drift from F4 (0.999995 / 3.37e-3 / 0.405).
  - No SDPA backend-fallback warning in the SLURM logs.
- **Staging FLIR to node-local scratch** (job B, SLURM 5525666, hpc-gpu5, W&B `xlop65eg`; `.../13-34-14/throughput.json`):
  - Copy: 46,450 files, 12,159 MB in **283 s** (42 MB/s).
  - The loader alone (8 workers) gets faster: RGB 349 → 478 fps, thermal 952 → 1537 fps.
  - End to end it changes nothing: SDPA ≈ 89 fps on both sources, encode share ≥ 0.998. The GPU stays the limit, so staging does not pay off for the encoder on an L4.
- **Full detector inference, thermal (encoder end to end + head infer at B = 8, fp16):**
  - plain + PyTorch MSDA + BeeGFS: 79.6 ms per frame;
  - SDPA + compiled MSDA + staged: 31.2 ms per frame;
  - **2.55× faster**.
- **Caveats:** two jobs ran per node on separate GPUs. A's GPU-only section overlapped B's staging copy, which affects only the batch-1 rows. B read from page cache. All tensors are synthetic except the end-to-end and loader-only sections.

## 7. Deviation list
D1–D8 in plan.md remain the only differences from detrex's `dino_vitdet` in the model. Nothing new was found in Step 9. Changes made since, none of which touch the model:
- the bench's grid, modes and `msda` switch (amendment 2026-10-26);
- `opencv-python-headless` (amendment 2026-10-25);
- the GIoU test expectation (amendment 2026-10-23).

The `msda=pytorch` shim lives in `projects/detection/head_bench.py` and replaces detrex's autograd function at runtime, only when it is requested. detrex's source is unchanged.

## 8. Not run / caveats
- The full `pytest -m gpu` console output and the `setup_env.sh` lines were not kept. The numbers above are as reported by the user in `bench-short-runs.md` and plan.md Amendments.
- `git status --porcelain` in the two detrex trees after the **CUDA** build on ctcomp3 was not captured. It is accepted (user decision 2026-10-26) on three grounds: both trees are clean on ctdesk236 after the CPU build, detrex's `.gitignore` covers every build artefact (`*.so`, `build/`, `detrex/version.py`), and the submodule pin is `e244e6c` with no `+` prefix.
- The head was not benchmarked at 32×64 (amendment 2026-10-26). The overfit test (§3) is the only 32×64 GPU timing.
- The S1 short run (`test_attention_sdpa.py` + `test_encoder.py` on the login node) was reported as pending. Both files pass on ctdesk236 as part of the default suite (213 passed, 2026-10-26).
