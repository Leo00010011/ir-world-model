#!/usr/bin/env bash
# R11 experiment: does staging FLIR on node-local scratch fix the slow `data` tests?
# Times, on one compute node: nproc, `import torch` from the shared .venv, the FLIR `data` tests
# reading the shared filesystem, the copy to $LOCAL_SCRATCH, and the same tests reading the copy.
#   sbatch --partition=defaultPartition --account=citius slurm/stage_data_test.sbatch
# SKIP_SHARED=1 skips the shared-filesystem baseline (174 s in Roadmap R11).
#
#SBATCH --job-name=irwm-stage-test
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:45:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/env.sh

if [[ "${IRWM_NO_UV_RUN:-0}" == "1" ]]; then py=(.venv/bin/python); else py=(uv run --frozen python); fi
tests=(-m pytest -m data tests/test_flir_data.py -q -p no:cacheprovider --durations=5)

timed() {  # timed <label> <cmd...>: run and print the wall time
    local label="$1" t0=${SECONDS}; shift
    echo "=== ${label}"
    "$@"
    echo "=== ${label}: $(( SECONDS - t0 )) s"
}

echo "[irwm] job ${SLURM_JOB_ID:-?} on $(hostname) | $(date -Is) | git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[irwm] nproc=$(nproc) | LOCAL_SCRATCH=${LOCAL_SCRATCH:-unset} | IRWM_DATA_ROOT=${IRWM_DATA_ROOT:-unset}"
df -h "${LOCAL_SCRATCH:?LOCAL_SCRATCH is not set}" "${IRWM_DATA_ROOT:?}"

timed "import torch (shared .venv)" "${py[@]}" -c "import torch; print(torch.__version__)"

if [[ "${SKIP_SHARED:-0}" != "1" ]]; then
    timed "data tests, shared filesystem" "${py[@]}" "${tests[@]}"
fi

timed "stage to local scratch" source slurm/stage_data.sh

timed "data tests, local scratch" "${py[@]}" "${tests[@]}"
