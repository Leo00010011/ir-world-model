# SLURM

All scripts work from any working directory inside the clone. Login nodes need internet; compute nodes do not.

| Script | Run it | What it does |
|---|---|---|
| `setup_env.sh` | `bash slurm/setup_env.sh [--test]` once per cluster, and again after `git pull` | Installs uv in user space if needed (`pip --user`, falling back to the standalone installer), inits submodules, runs `uv sync --frozen`, creates `.env` from the example, creates the `IRWM_*` dirs, prints torch/CUDA/driver versions, and verifies the W&B key. `--test` also runs pytest. |
| `submit.sh` | `slurm/submit.sh <site> <entry.py> [overrides]` | Single job. Derives the sbatch flags from `lib/irwm/conf/cluster/<site>.yaml` (the only place resources are defined) and writes logs to `$IRWM_OUTPUT_ROOT/slurm_logs/`. |
| `job.sbatch` | (called by `submit.sh`) | Sources `env.sh`, prints job/GPU info, then runs `srun uv run --frozen python <entry.py> ...`. |
| `env.sh` | `source slurm/env.sh` | Shared env: adds `~/.local/bin` to PATH, loads `.env`, and exports `WANDB_API_KEY` from `$WANDB_API_KEY_FILE` (default `~/.wandb_api_key`). Never overrides variables that are already set. |
| `stage_data.sh` | `source slurm/stage_data.sh` inside a job, after `env.sh` | Copies FLIR to node-local `$LOCAL_SCRATCH/irwm_data/` (12 GB in about 4.6 min on ctcomp3), checks the file count and points `IRWM_DATA_ROOT` at the copy. Scratch is wiped when the job ends: never write results there. `stage_data_test.sbatch` is the R11 timing experiment. `IRWM_STAGE_DATA=1 slurm/submit.sh ...` makes `job.sbatch` do this before the entry point runs. |

## Adding a cluster
1. `cp lib/irwm/conf/cluster/example_site.yaml lib/irwm/conf/cluster/<site>.yaml`, then set `name`, `partition`, `account`, `qos` and `gres` (and anything else from `cluster/slurm.yaml`). Commit it.
2. On the login node: `git clone --recurse-submodules ... && bash slurm/setup_env.sh`. Edit `.env` (point the paths at scratch) and re-run.
3. `slurm/submit.sh <site> projects/smoke/train.py`. Check the log and the W&B run.

## Notes
- **Sweeps:** `uv run python <entry.py> -m cluster=<site> a=1,2` goes through submitit with the same site config. Each job sources `env.sh` through `hydra.launcher.setup`.
- **Per-run resources:** `slurm/submit.sh <site> <entry.py> cluster.timeout_min=600 cluster.gres=gpu:2`. Raw extra flags go in `SBATCH_EXTRA="--exclude=node12"`.
- **Preemption:** `--requeue` and `--signal=SIGUSR1@90` let Lightning save `hpc_ckpt_*.ckpt` and requeue. The job keeps its `SLURM_JOB_ID`, so `run_name=slurm-<id>` points at the same checkpoint dir, and `resume_ckpt_path()` picks up the newest checkpoint. Submitit sweeps requeue through submitit itself and resume from `last.ckpt`.
- **Small home quota:** set `UV_CACHE_DIR` to scratch before running `setup_env.sh`. The cache is several GB, mostly torch.
- **Offline compute nodes:** set `WANDB_MODE=offline` (sync later with `uv run wandb sync <dir>`). `IRWM_NO_UV_RUN=1` makes jobs call `.venv/bin/python` directly.
