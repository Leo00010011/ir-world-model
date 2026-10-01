#!/usr/bin/env bash
# Generic single-job template. Submit from the repo root:
#   sbatch slurm/job.sbatch <entry_point.py> [hydra overrides...]
#   sbatch slurm/job.sbatch projects/smoke/train.py cluster=slurm
# Resources: edit the TODO lines (keep them in sync with lib/irwm/conf/cluster/slurm.yaml),
# or override per submission: sbatch --time=08:00:00 --gres=gpu:2 slurm/job.sbatch ...
#
#SBATCH --job-name=irwm
#SBATCH --partition=TODO_PARTITION
##SBATCH --account=TODO_ACCOUNT
##SBATCH --qos=TODO_QOS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch slurm/job.sbatch <entry_point.py> [hydra overrides...]" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

# Load IRWM_* / WANDB_* from .env if present. Variables already exported in the shell win.
if [[ -f .env ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        key="${key// /}"
        value="${value%%#*}"; value="${value%"${value##*[![:space:]]}"}"
        value="${value/#\~/$HOME}"
        [[ -z "${!key:-}" ]] && export "$key=$value"
    done < .env
fi

echo "[irwm] job ${SLURM_JOB_ID:-?} on $(hostname) | restart count ${SLURM_RESTART_COUNT:-0} | $(date -Is)"
echo "[irwm] git $(git rev-parse --short HEAD 2>/dev/null || echo unknown) | cmd: $*"

# srun so Lightning sees one task per GPU. Set IRWM_NO_UV_RUN=1 on nodes where `uv run` must not
# touch the network or the lockfile (the env was built on the login node with `uv sync --frozen`).
if [[ "${IRWM_NO_UV_RUN:-0}" == "1" ]]; then
    srun .venv/bin/python "$@"
else
    srun uv run --frozen python "$@"
fi
