#!/usr/bin/env bash
# Submit the aligned FLIR v1 detector trainings (F18): each head is its OWN SLURM job, no
# dependencies between them, so they schedule and run in parallel.
#
#   slurm/train_aligned.sh <site> [--smoke] [--heads h1,h2,...] [--dry-run] [-- hydra overrides...]
#
#   --heads    comma list; full names (aligned_dinopca_rgb) or short ones (dinopca_rgb), or a
#              backbone (dinopca | dino3072 | mae) for both modalities. Default: all six.
#   --smoke    pipeline check: 32 frames, 60 steps, val on the same frames (smoke.enabled=true), 45 min.
#   --dry-run  print the exact sbatch command of every head (resources, staging env) and submit nothing.
#   after --    extra Hydra overrides passed to every job (e.g. -- seed=1 data.batch_size=8).
#
# Every job stages aligned FLIR v1 to node-local scratch (IRWM_STAGE_DATA=1,
# IRWM_STAGE_DATASETS="aligned/align", slurm/stage_data.sh) and records the GPU it ran on.
# Per-head resources (env overrides in brackets):
#   dinopca / dino3072: gpu:1 [DINO_GRES], 210 min [DINO_TIME], 8 cpus / 8 workers, 32 GB
#   mae               : [MAE_GRES], 360 min [MAE_TIME], 8 cpus / 8 workers, 48 GB [MAE_MEM]
# Time limits are sized for the WORST GPU a job can land on, an L4 (2026-11-13, from the cluster smoke):
#   full run = 20 epochs x 231 steps = 4,620 steps (+ 20 validations of 413 frames, staging, startup).
#   DINO: L4 1.52 s/step (F8, measured) -> 1.95 h train, ~2.3 h total, x1.5 margin -> 210 min
#         (smoke: V100S 0.73 s/step, A100 0.49 s/step).
#   MAE : A100 0.86 s/step (smoke); L4 ~3.1x slower (A100->L4 ratio seen for DINO) -> ~2.7 s/step,
#         3.5 h train, ~3.8 h total, x1.5 margin -> 360 min. Peak 19.7 GiB (act checkpointing): fits 24 GB.
#   The USR1 -> requeue -> resume path is untested on the cluster (F8), so do not rely on it.
# Any GPU at least as fast as an L4 is fine for the DINO heads: never pin them to a GPU type.
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! MAE jobs must request the LARGEST GPU on the cluster (end-to-end ViT-B at 512x640, TechStack     !!
# !! §3.1b). The GPU type is NOT known yet, so MAE_GRES defaults to plain "gpu:1" (ANY GPU). Set it,  !!
# !! e.g.  MAE_GRES=gpu:a100:1 slurm/train_aligned.sh ctcomp3 --heads mae                           !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# Prints one "<head> <job id>" line per job at the end.
set -euo pipefail

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }

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

ALL_HEADS=(aligned_dinopca_rgb aligned_dinopca_ir aligned_dino3072_rgb aligned_dino3072_ir aligned_mae_rgb aligned_mae_ir)
heads=()
IFS=',' read -ra req <<< "${heads_arg}"
for h in "${req[@]}"; do
    case "${h}" in
        all) heads+=("${ALL_HEADS[@]}") ;;
        dinopca|dino3072|mae) heads+=("aligned_${h}_rgb" "aligned_${h}_ir") ;;
        aligned_*) heads+=("${h}") ;;
        *) heads+=("aligned_${h}") ;;
    esac
done
for h in "${heads[@]}"; do
    [[ " ${ALL_HEADS[*]} " == *" ${h} "* ]] || { echo "unknown head: ${h} (known: ${ALL_HEADS[*]})" >&2; exit 2; }
done
# de-duplicate, keep order
mapfile -t heads < <(printf '%s\n' "${heads[@]}" | awk '!seen[$0]++')

export IRWM_STAGE_DATA=1
export IRWM_STAGE_DATASETS="aligned/align"
[[ "${dry}" == "1" ]] && export IRWM_SUBMIT_DRY_RUN=1

if [[ " ${heads[*]} " == *"_mae_"* && -z "${MAE_GRES:-}" ]]; then
    echo "[train_aligned] WARNING: MAE_GRES is not set; mae jobs request plain gpu:1 (ANY GPU)." >&2
    echo "[train_aligned]          Set MAE_GRES=gpu:<largest type>:1 once the GPU type is known." >&2
fi

declare -a summary=() job_ids=()
for h in "${heads[@]}"; do
    if [[ "${h}" == aligned_mae_* ]]; then
        gres="${MAE_GRES:-gpu:1}"; time_min="${MAE_TIME:-360}"; mem="${MAE_MEM:-48}"
    else
        gres="${DINO_GRES:-gpu:1}"; time_min="${DINO_TIME:-210}"; mem="${DINO_MEM:-32}"
    fi
    [[ "${smoke}" == "1" ]] && time_min="${SMOKE_TIME:-45}"
    overrides=("aligned=${h}" "cluster.gres='${gres}'" "cluster.timeout_min=${time_min}"
               "cluster.cpus_per_task=8" "cluster.num_workers=8" "cluster.mem_gb=${mem}")
    [[ "${smoke}" == "1" ]] && overrides+=("smoke.enabled=true")
    overrides+=(${extra[@]+"${extra[@]}"})
    tag="${h}$([[ "${smoke}" == "1" ]] && echo "-smoke" || true)"
    out="$(SBATCH_EXTRA="${SBATCH_EXTRA:-} --job-name=irwm-${tag}" \
        bash slurm/submit.sh "${site}" projects/detection/train_aligned.py "${overrides[@]}" 2>&1)" || {
        echo "${out}" >&2; echo "[train_aligned] submission of ${h} FAILED" >&2; exit 1; }
    if [[ "${dry}" == "1" ]]; then
        echo "${out}" | grep '^\[dry-run\]' | sed "s/^\[dry-run\]/[dry-run ${h}]/"
        summary+=("${h} (dry-run)")
    else
        job="$(echo "${out}" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1)"
        [[ -n "${job}" ]] || { echo "${out}" >&2; echo "[train_aligned] no job id for ${h}" >&2; exit 1; }
        summary+=("${h} ${job}")
        job_ids+=("${job}")
    fi
done

echo "[train_aligned] $([[ "${dry}" == "1" ]] && echo "would submit" || echo "submitted") $([[ "${smoke}" == "1" ]] && echo "SMOKE " || true)jobs:"
printf '  %s\n' "${summary[@]}"

# Ready-to-paste F19 evaluation, waiting for these trainings (each exports its frozen head at the end).
if [[ "${smoke}" == "1" ]]; then
    echo "[train_aligned] smoke runs export no frozen head: no eval command."
else
    if [[ "${dry}" == "1" ]]; then
        dep="afterok:<job ids>"
    else
        dep="afterok:$(IFS=:; echo "${job_ids[*]}")"
    fi
    heads_override=""
    if (( ${#heads[@]} < ${#ALL_HEADS[@]} )); then
        heads_override=" 'heads=[$(IFS=,; echo "${heads[*]}")]'"
    fi
    echo "[train_aligned] evaluation (F19), after all of them:"
    echo "  SBATCH_EXTRA=\"--dependency=${dep}\" bash slurm/eval_aligned.sh ${site}${heads_override}"
fi
