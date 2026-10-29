#!/usr/bin/env bash
# Train a DINO-DETR detector on FLIR (frozen DINOv2-Reg features), on this machine's GPU.
#
#   bash projects/detector_kit/train.sh --modality thermal [options] [-- hydra overrides]
#
# Options:
#   --modality rgb|thermal   which FLIR images to train on (required)
#   --epochs N               training epochs (default 50; lr drops x0.1 at 80 % of them)
#   --batch-size N           images per step and GPU (default 16, needs ~24 GB; use 4-8 on smaller GPUs)
#   --workers N              data-loading processes (default: min(8, CPU cores))
#   --gpus N                 GPUs on this node (default 1; >1 uses DDP)
#   --lr X                   learning rate (default 1e-4)
#   --val-every N            validate (COCO mAP on FLIR val) every N epochs (default 5)
#   --name NAME              run name = checkpoint folder; re-run with the same name to RESUME
#   --msda auto|cuda|pytorch deformable attention: compiled detrex op or PyTorch fallback (default auto)
#   --wandb                  also log to Weights & Biases (needs `wandb login`)
#   --quick                  setup check: 1 epoch on 20 batches, validation on 10 batches
#   --out DIR                folder for logs/metrics (default $IRWM_OUTPUT_ROOT/detector_kit/<date>/<time>)
#   -h, --help               this help
# Anything after `--` goes to projects/detector_kit/train.py as Hydra overrides (see conf/train.yaml).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || kit_die "$1 needs a value (see --help)"; }

modality="" ; workers="" ; overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --modality)   need_value "$@"; modality="$2"; shift 2 ;;
        --epochs)     need_value "$@"; overrides+=("trainer.max_epochs=$2"); shift 2 ;;
        --batch-size) need_value "$@"; overrides+=("data.batch_size=$2"); shift 2 ;;
        --workers)    need_value "$@"; workers="$2"; shift 2 ;;
        --gpus)       need_value "$@"; overrides+=("cluster.devices=$2"); shift 2 ;;
        --lr)         need_value "$@"; overrides+=("optim.lr=$2"); shift 2 ;;
        --val-every)  need_value "$@"; overrides+=("trainer.check_val_every_n_epoch=$2"); shift 2 ;;
        --name)       need_value "$@"; overrides+=("run_name=$2"); shift 2 ;;
        --msda)       need_value "$@"; overrides+=("msda=$2"); shift 2 ;;
        --wandb)      overrides+=("wandb.mode=online"); shift ;;
        --quick)      overrides+=("trainer.max_epochs=1" "trainer.limit_train_batches=20"
                                  "trainer.limit_val_batches=10" "trainer.check_val_every_n_epoch=1"
                                  "trainer.log_every_n_steps=5"); shift ;;
        --out)        need_value "$@"; overrides+=("hydra.run.dir='$(kit_abspath "$2")'"); shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; overrides+=("$@"); break ;;
        *)            kit_die "unknown option '$1' (see --help)" ;;
    esac
done
[[ "${modality}" == rgb || "${modality}" == thermal ]] || kit_die "--modality rgb|thermal is required (see --help)"
if [[ -z "${workers}" ]]; then
    cores="$(nproc 2>/dev/null || echo 2)"
    workers=$(( cores < 8 ? cores : 8 ))
fi

kit_setup
kit_say "training on FLIR ${modality} | ${workers} loader workers"
kit_python projects/detector_kit/train.py "modality=${modality}" "cluster.num_workers=${workers}" \
    ${overrides[@]+"${overrides[@]}"}
