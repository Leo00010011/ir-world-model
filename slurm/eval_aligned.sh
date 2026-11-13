#!/usr/bin/env bash
# F19: submit the aligned FLIR v1 evaluation (projects/detection/eval_aligned.py) as one SLURM job.
#   bash slurm/eval_aligned.sh <site> [hydra overrides...]
#   bash slurm/eval_aligned.sh ctcomp3                                       # all six aligned heads
#   bash slurm/eval_aligned.sh ctcomp3 'heads=[aligned_mae_rgb,aligned_mae_ir]'  # a subset
# Wait for the F18 trainings (the export must have happened; an exported head is what is scored):
#   SBATCH_EXTRA="--dependency=afterok:<job1>:<job2>:..." bash slurm/eval_aligned.sh ctcomp3
# Stages aligned/align to node-local scratch (R11). Other sbatch flags also go through SBATCH_EXTRA.
# Default resources (override with cluster.* as with slurm/submit.sh): 60 min, 8 CPUs/workers, 32 GB.
# Any GPU works; the job records which one it got in eval_aligned.json.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: slurm/eval_aligned.sh <site> [hydra overrides...]" >&2
    exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."

site="$1"; shift
export IRWM_STAGE_DATA=1
export IRWM_STAGE_DATASETS="aligned/align"
export SBATCH_EXTRA="${SBATCH_EXTRA:-}"

exec bash slurm/submit.sh "${site}" projects/detection/eval_aligned.py \
    cluster.timeout_min=60 cluster.cpus_per_task=8 cluster.num_workers=8 cluster.mem_gb=32 "$@"
