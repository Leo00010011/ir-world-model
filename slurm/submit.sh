#!/usr/bin/env bash
# Submit one job with resources taken from lib/irwm/conf/cluster/<site>.yaml.
#   slurm/submit.sh <site> <entry_point.py> [hydra overrides...]
#   slurm/submit.sh snellius projects/smoke/train.py trainer.max_epochs=5 cluster.timeout_min=30
# cluster.* overrides change both the sbatch flags and the job's config. Extra raw sbatch flags
# can be passed through SBATCH_EXTRA="--exclude=node12 --dependency=afterok:123".
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: slurm/submit.sh <site> <entry_point.py> [hydra overrides...]" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/env.sh

site="$1"; entry="$2"; shift 2
[[ -f "${entry}" ]] || { echo "entry point not found: ${entry}" >&2; exit 2; }

flags="$(uv run --frozen python -m irwm.utils.slurm "${site}" "$@")"
read -ra flag_arr <<< "${flags}"
read -ra extra_arr <<< "${SBATCH_EXTRA:-}"
# ${arr[@]+...} below: empty arrays are "unbound" under set -u on bash < 4.4.

project="$(basename "$(dirname "${entry}")")"
log_dir="${IRWM_OUTPUT_ROOT:?IRWM_OUTPUT_ROOT is not set (see .env.example)}/slurm_logs"
mkdir -p "${log_dir}"

set -x
sbatch --job-name="irwm-${project}" --output="${log_dir}/%x-%j.out" \
    ${flag_arr[@]+"${flag_arr[@]}"} ${extra_arr[@]+"${extra_arr[@]}"} \
    slurm/job.sbatch "${entry}" "cluster=${site}" "$@"
