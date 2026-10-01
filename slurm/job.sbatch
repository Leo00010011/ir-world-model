#!/usr/bin/env bash
# Generic single-job script. Prefer submitting through slurm/submit.sh, which derives the
# resource flags (partition, gres, time, ...) from lib/irwm/conf/cluster/<site>.yaml:
#   slurm/submit.sh <site> projects/smoke/train.py [hydra overrides...]
# Raw use (resources from the command line / cluster defaults):
#   sbatch --partition=gpu --gres=gpu:1 slurm/job.sbatch projects/smoke/train.py cluster=<site>
#
#SBATCH --job-name=irwm
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch [flags] slurm/job.sbatch <entry_point.py> [hydra overrides...]" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh

echo "[irwm] job ${SLURM_JOB_ID:-?} on $(hostname) | restart ${SLURM_RESTART_COUNT:-0} | $(date -Is)"
echo "[irwm] git $(git rev-parse --short HEAD 2>/dev/null || echo unknown) | cmd: $*"
echo "[irwm] W&B key: $([[ -n "${WANDB_API_KEY:-}" ]] && echo present || echo MISSING) | WANDB_MODE=${WANDB_MODE:-online}"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "[irwm] nvidia-smi not found"

# srun so Lightning sees one task per GPU. IRWM_NO_UV_RUN=1 skips `uv run` (no lockfile/network
# access at all) on nodes where the env was built beforehand with slurm/setup_env.sh.
if [[ "${IRWM_NO_UV_RUN:-0}" == "1" ]]; then
    srun .venv/bin/python "$@"
else
    srun uv run --frozen python "$@"
fi
