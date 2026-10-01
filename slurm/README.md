# SLURM

- **Single job:** `sbatch slurm/job.sbatch <entry_point.py> cluster=slurm [overrides...]`, submitted from the repo root. Logs go to `slurm_logs/`.
- **Sweep (submitit):** `uv run python <entry_point.py> -m cluster=slurm a=1,2`. The resources come from `lib/irwm/conf/cluster/slurm.yaml`.
- **Preemption:** `--requeue` together with `--signal=SIGUSR1@90` lets Lightning's `SLURMEnvironment(auto_requeue=True)` save a checkpoint and requeue the job. The requeued job keeps its `SLURM_JOB_ID`, so `run_name=slurm-<id>` resolves to the same checkpoint dir and the job resumes.
- **Before the first run:** fill in the `TODO` values here and in `cluster/slurm.yaml`. Build the env on the login node (`pip install --user uv && uv sync --frozen`). If compute nodes have no internet, set `WANDB_MODE=offline` and `IRWM_NO_UV_RUN=1`.
