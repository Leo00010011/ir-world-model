# F7 · DINO-DETR head port: Handover (2026-10-23)

Build session status. **Paused at plan.md Step 9 (user)**: the CUDA build and GPU runs on ctcomp3. Everything that doesn't need a GPU is built and tested on ctdesk236.

## What was built
- **FR1:** detrex is a submodule at `third_party/detrex` (`e244e6c`), with detectron2 inside it (`70f4543`). Both pins match the spec and both trees stay clean after the build.
- **FR2:** detectron2 and detrex are core dependencies, with the lock metadata declared statically, the no-isolation build and `setuptools<81` (`pyproject.toml`, `lib/pyproject.toml`). `slurm/setup_env.sh` now does the two-phase sync, builds the CUDA ops when `nvcc` is present, prints the compiler versions and build time, and checks the detrex import. `uv lock` takes under 1 s and builds nothing. The CPU build on ctdesk236 took 52 s.
- **FR3–FR7:** `lib/irwm/detection/detrex_adapter.py`, `stem.py`, `detector.py` and the re-exports in `__init__.py`. The parameter counts match the spec exactly: 23,001,286 for the default head and 24,479,686 with 3072-d input.
- **FR8:** the `detector` Hydra group (`lib/irwm/conf/detector/default.yaml`), added to `irwm_base.yaml`.
- **FR9:** `tests/test_detection_head.py` and the new `gpu` marker. The 31 CPU tests pass in 9.5 s. Groups 2–6 also pass with networking switched off (`unshare -rn`), not only reviewed by eye.
- **FR10:** `projects/detection/` with `head_bench.py`. Without CUDA ops it exits with status 1 and a clear message.

## Validation tally
**36 passed / 0 deferred / 14 pending.**
- **Waiting on Step 9:**
  - Group 7 (5 items) and Group 8 (1 item);
  - all 5 Data Validity (results) items;
  - "detrex unmodified" (the CUDA build still has to leave the tree clean);
  - "Provenance" (needs `head_bench.json`).
- **Blocked by an older problem:** the "default suite unaffected" item is still open. The full suite on ctdesk236 is 208 passed, 2 failed. One failure was the GIoU test, fixed by the amendment below (the F7 file now passes 31/31). The other, `test_source_files_tracked_by_git`, was already failing before this work: line 39 of `.gitignore` has `feature_extraction/` without a leading `/`, so it matches the tracked `projects/feature_extraction/` files. It was not fixed because it's outside the spec. Making it `/feature_extraction/`, or removing the line, would probably clear it.

## Amendment (user decision)
- **Group 6 GIoU check:** the test now expects `mean(1e-6/(A+1e-6))` instead of ≈ 0, because detrex's `box_iou` divides by `union + 1e-6` (`detrex/layers/box_ops.py:81`). This is recorded in `plan.md ## Amendments` and in the `validation.md` item.

## Next steps (user)
1. Commit and push. The cluster needs this code (suggested message below).
2. On the ctcomp3 login node, load a CUDA 12.x `nvcc` (for example `module load cuda/12.1`) and a `gcc` no newer than 12. Check them with `nvcc --version` and `gcc --version`.
3. Run `git pull && bash slurm/setup_env.sh`. The CUDA build may take tens of minutes.
4. Run the GPU tests on the L4 node:
   ```
   srun --gres=gpu:1 -w hpc-gpu5 --pty bash -c 'source slurm/env.sh && uv run pytest tests/test_detection_head.py -m gpu -v -s'
   ```
   - Add your site's partition flags if needed.
   - `-s` was added to the spec's command so the overfit summary line gets printed.
5. Run the benchmark:
   ```
   bash slurm/submit.sh ctcomp3 projects/detection/head_bench.py cluster.timeout_min=60
   ```

**Bring back:**
- the `setup_env.sh` lines with the `nvcc`/`gcc` versions and the phase-2 build seconds;
- the full pytest output (including the `[overfit] {...}` line);
- `head_bench.json` and the W&B run id.

With those, the next `/build_spec dino-detr-head-port` session writes `results.md` (Step 10), flips the pending items and proposes the Roadmap and TechStack updates.

## Suggested commit message
```
F7: DINO-DETR head port on detrex (submodule, adapter, stem, wrapper, tests, bench)

- third_party/detrex @ e244e6c (+ detectron2 @ 70f4543) as a pinned, unmodified submodule,
  installed editable without build isolation; static uv dependency-metadata; setuptools<81
- setup_env.sh: two-phase sync, CUDA ops when nvcc is present (FORCE_CUDA, arch list)
- irwm.detection: detrex adapter, FeatureStem, FeatureDINO, build_dino_config, DINODetector
- Hydra group `detector`; `gpu` pytest marker; tests/test_detection_head.py
- projects/detection/head_bench.py (GPU train-step cost for F8)
- spec amendment: analytic GIoU expectation in validation Group 6

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

## Other things noticed
- **Root dependency:** `irwm-detection` was also added to the root `dependencies`, the same way the other project folders are listed. The spec only mentioned the root `[tool.uv.sources]` entry.
- **Notebook group:** a plain `uv sync` removes the `notebook` dependency group, as TechStack already says.
- **Windows laptop:** detectron2 and detrex are now core dependencies, so they need MSVC Build Tools there. The spec already flags this.
