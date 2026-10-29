#!/usr/bin/env bash
# Detect objects (person, bike, car, motor, light, sign) with a DINO-DETR detector on frozen
# DINOv2-Reg features. Works on a GPU or on CPU (slow), with or without detrex's compiled ops.
#
#   bash projects/detector_kit/inference.sh --head det_ir --images /path/to/frames [options]
#   bash projects/detector_kit/inference.sh --head det_rgb --split test [options]
#
# Model (one of):
#   --head det_rgb|det_ir    a frozen detector from $IRWM_CKPT_ROOT/detection/frozen/<head>/
#   --ckpt PATH              a checkpoint written by train.sh (best-eNNN.ckpt or last.ckpt)
# Input (one of):
#   --images DIR             every .jpg/.jpeg/.png/.bmp/.tif/.tiff under DIR
#   --split val|test|train   a FLIR split; also computes COCO mAP against its labels
# Options:
#   --modality rgb|thermal   how to read the images (default: what the model was trained on)
#   --device auto|cuda|cpu   default auto (GPU if there is one)
#   --batch-size N           default 8
#   --workers N              data-loading processes for --split (default: min(8, CPU cores))
#   --score-thr X            keep/draw boxes with score >= X (default 0.3)
#   --max-draw N             draw at most N images (default 100); --no-draw draws none
#   --msda auto|cuda|pytorch deformable attention implementation (default auto)
#   --out DIR                output folder (default $IRWM_OUTPUT_ROOT/detector_kit/<date>/<time>)
#   -h, --help               this help
# Anything after `--` goes to projects/detector_kit/inference.py as Hydra overrides (see conf/inference.yaml).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || kit_die "$1 needs a value (see --help)"; }

head_name="" ; ckpt="" ; images="" ; split="" ; workers="" ; overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --head)       need_value "$@"; head_name="$2"; shift 2 ;;
        --ckpt)       need_value "$@"; ckpt="$(kit_abspath "$2")"; shift 2 ;;
        --images)     need_value "$@"; images="$(kit_abspath "$2")"; shift 2 ;;
        --split)      need_value "$@"; split="$2"; shift 2 ;;
        --modality)   need_value "$@"; overrides+=("modality=$2"); shift 2 ;;
        --device)     need_value "$@"; overrides+=("device=$2"); shift 2 ;;
        --batch-size) need_value "$@"; overrides+=("batch_size=$2"); shift 2 ;;
        --workers)    need_value "$@"; workers="$2"; shift 2 ;;
        --score-thr)  need_value "$@"; overrides+=("score_thr=$2"); shift 2 ;;
        --max-draw)   need_value "$@"; overrides+=("max_draw=$2"); shift 2 ;;
        --no-draw)    overrides+=("draw=false"); shift ;;
        --msda)       need_value "$@"; overrides+=("msda=$2"); shift 2 ;;
        --out)        need_value "$@"; overrides+=("hydra.run.dir='$(kit_abspath "$2")'"); shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; overrides+=("$@"); break ;;
        *)            kit_die "unknown option '$1' (see --help)" ;;
    esac
done
[[ -n "${head_name}" || -n "${ckpt}" ]] || kit_die "choose a model: --head det_rgb|det_ir or --ckpt PATH (see --help)"
[[ -z "${head_name}" || -z "${ckpt}" ]] || kit_die "give --head or --ckpt, not both"
[[ -n "${images}" || -n "${split}" ]] || kit_die "choose an input: --images DIR or --split val|test|train (see --help)"
[[ -z "${images}" || -z "${split}" ]] || kit_die "give --images or --split, not both"
[[ -z "${images}" || -d "${images}" ]] || kit_die "--images: ${images} is not a folder"
[[ -z "${ckpt}" || -f "${ckpt}" ]] || kit_die "--ckpt: ${ckpt} does not exist"
if [[ -z "${workers}" ]]; then
    cores="$(nproc 2>/dev/null || echo 2)"
    workers=$(( cores < 8 ? cores : 8 ))
fi

kit_setup
if [[ -n "${head_name}" ]]; then
    [[ "${head_name}" == det_rgb || "${head_name}" == det_ir ]] || kit_die "--head must be det_rgb or det_ir"
    frozen="${IRWM_CKPT_ROOT}/detection/frozen/${head_name}"
    [[ -f "${frozen}/head.pt" && -f "${frozen}/meta.json" ]] \
        || kit_die "frozen detector not found: ${frozen}/{head.pt,meta.json}. Copy the '${head_name}' folder you were given there (see README)"
    overrides=("head=${head_name}" ${overrides[@]+"${overrides[@]}"})
else
    overrides=("ckpt='${ckpt}'" ${overrides[@]+"${overrides[@]}"})
fi
if [[ -n "${images}" ]]; then
    overrides+=("images='${images}'")
else
    overrides+=("split=${split}")
fi

kit_python projects/detector_kit/inference.py "cluster.num_workers=${workers}" "${overrides[@]}"
