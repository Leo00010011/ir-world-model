#!/usr/bin/env bash
# Submit the KAIST-Paired detector trainings (2026-11-27): kaist_dinopca_rgb (visible camera) and
# kaist_dinopca_ir (lwir camera). Each head is its OWN SLURM job, with no dependency between them.
#
#   slurm/train_kaist.sh <site> [--smoke] [--heads h1,h2] [--dry-run] [-- hydra overrides...]
#
#   --heads    comma list: kaist_dinopca_rgb, kaist_dinopca_ir (or rgb, ir). Default: both.
#   --smoke    pipeline check: 32 frames, 60 steps, val on the same frames (smoke.enabled=true), 45 min.
#   --dry-run  print the exact sbatch command of every head and submit nothing.
#   after --   extra Hydra overrides passed to every job (e.g. -- seed=1 data.batch_size=8).
#
# Every job stages only what it reads to node-local scratch (slurm/stage_data.sh): the train-all-04
# frames of its own camera (kaist-cvpr15) and their KAIST-Paired annotations (kaist-paired).
# Resources: gpu:1 [GRES], 270 min [TIME], 8 cpus / 8 workers, 32 GB [MEM].
# Time limit sized for the WORST GPU a job can land on, an L4 (estimate, not measured):
#   train: ~3.4k positive + ~0.85k empty frames (data.empty_ratio 0.25) / 16 = ~265 steps x 1.52 s (F8, L4)
#   = ~6.7 min per epoch, 20 epochs = 2.2 h; 10 validations x ~4.7k frames (2.7k val + 2.0k train subset)
#   at ~30 ms/frame = ~25 min; total ~2.7 h, x1.5 margin + staging/startup -> 270 min.
#   (data.empty_ratio=all: ~9.4k frames per epoch, ~5.5 h total: pass TIME=540.) Any GPU at least as fast as an L4 is fine: never pin a type.
#
# Prints one "<head> <job id>" line per job at the end.
set -euo pipefail

usage() { sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }

[[ $# -ge 1 && "$1" != -* ]] || usage
site="$1"; shift
smoke=0; dry=0; heads_arg="all"; extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) smoke=1 ;;
        --dry-run) dry=1 ;;
        --heads) [[ $# -ge 2 ]] || usage; heads_arg="$2"; shift ;;
        --heads=*) heads_arg="${1#--heads=}" ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
    shift
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ALL_HEADS=(kaist_dinopca_rgb kaist_dinopca_ir)
heads=()
IFS=',' read -ra req <<< "${heads_arg}"
for h in "${req[@]}"; do
    case "${h}" in
        all) heads+=("${ALL_HEADS[@]}") ;;
        rgb|ir) heads+=("kaist_dinopca_${h}") ;;
        *) heads+=("${h}") ;;
    esac
done
for h in "${heads[@]}"; do
    [[ " ${ALL_HEADS[*]} " == *" ${h} "* ]] || { echo "unknown head: ${h} (known: ${ALL_HEADS[*]})" >&2; exit 2; }
done
mapfile -t heads < <(printf '%s\n' "${heads[@]}" | awk '!seen[$0]++')

export IRWM_STAGE_DATA=1
export IRWM_STAGE_DATASETS="kaist-cvpr15 kaist-paired"
export IRWM_STAGE_KAIST_LIST="train-all-04.txt"
[[ "${dry}" == "1" ]] && export IRWM_SUBMIT_DRY_RUN=1

gres="${GRES:-gpu:1}"; mem="${MEM:-32}"
time_min="${TIME:-270}"; [[ "${smoke}" == "1" ]] && time_min="${SMOKE_TIME:-45}"

declare -a summary=()
for h in "${heads[@]}"; do
    if [[ "${h}" == *_rgb ]]; then export IRWM_STAGE_KAIST_CAMERAS=visible; else export IRWM_STAGE_KAIST_CAMERAS=lwir; fi
    overrides=("kaist=${h}" "cluster.gres='${gres}'" "cluster.timeout_min=${time_min}"
               "cluster.cpus_per_task=8" "cluster.num_workers=8" "cluster.mem_gb=${mem}")
    [[ "${smoke}" == "1" ]] && overrides+=("smoke.enabled=true")
    overrides+=(${extra[@]+"${extra[@]}"})
    tag="${h}$([[ "${smoke}" == "1" ]] && echo "-smoke" || true)"
    out="$(SBATCH_EXTRA="${SBATCH_EXTRA:-} --job-name=irwm-${tag}" \
        bash slurm/submit.sh "${site}" projects/detection/train_kaist.py "${overrides[@]}" 2>&1)" || {
        echo "${out}" >&2; echo "[train_kaist] submission of ${h} FAILED" >&2; exit 1; }
    if [[ "${dry}" == "1" ]]; then
        echo "${out}" | grep '^\[dry-run\]' | sed "s/^\[dry-run\]/[dry-run ${h} camera=${IRWM_STAGE_KAIST_CAMERAS}]/"
        summary+=("${h} (dry-run)")
    else
        job="$(echo "${out}" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1)"
        [[ -n "${job}" ]] || { echo "${out}" >&2; echo "[train_kaist] no job id for ${h}" >&2; exit 1; }
        summary+=("${h} ${job}")
    fi
done

echo "[train_kaist] $([[ "${dry}" == "1" ]] && echo "would submit" || echo "submitted") $([[ "${smoke}" == "1" ]] && echo "SMOKE " || true)jobs:"
printf '  %s\n' "${summary[@]}"
