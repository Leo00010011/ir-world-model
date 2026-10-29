# shellcheck shell=bash
# Shared by train.sh and inference.sh. SOURCE it (do not execute it). It:
#   1. moves to the repo root and loads .env + uv's PATH (slurm/env.sh)
#   2. checks the IRWM_* path variables
#   3. builds the uv env on first use (slurm/setup_env.sh), detrex/detectron2 included
#   4. downloads the frozen encoder's weights (DINOv2-Reg + PCA, ~350 MB) on first use

KIT_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

kit_say()  { printf '\033[1m[detector_kit] %s\033[0m\n' "$*"; }
kit_die()  { printf '\033[31m[detector_kit] ERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# Absolute path of $1, resolved against the caller's directory (call it BEFORE kit_setup changes dir).
kit_abspath() { [[ "$1" == /* ]] && printf '%s' "$1" || printf '%s/%s' "$(pwd)" "$1"; }

# Python of the project env: `uv run` by default; IRWM_NO_UV_RUN=1 uses .venv directly (offline nodes).
kit_python() {
    if [[ "${IRWM_NO_UV_RUN:-0}" == "1" ]]; then
        "${KIT_REPO}/.venv/bin/python" "$@"
    else
        uv run --frozen python "$@"
    fi
}

kit_setup() {
    cd "${KIT_REPO}"
    source slurm/env.sh

    # Path variables (TechStack §6): every run reads them, see projects/detector_kit/README.md.
    local var missing=()
    for var in IRWM_DATA_ROOT IRWM_CACHE_ROOT IRWM_CKPT_ROOT IRWM_OUTPUT_ROOT; do
        [[ -n "${!var:-}" ]] || missing+=("${var}")
    done
    if (( ${#missing[@]} )); then
        [[ -f .env ]] || cp .env.example .env
        kit_die "set ${missing[*]} in ${KIT_REPO}/.env (created from .env.example if it was missing), then re-run"
    fi

    if [[ ! -f third_party/detrex/projects/dino/modeling/__init__.py ]]; then
        kit_say "fetching the detrex / DINO-Foresight submodules"
        git submodule update --init --recursive
    fi

    if ! kit_python -c "import irwm.detection" >/dev/null 2>&1; then
        command -v uv >/dev/null 2>&1 || kit_say "uv not found; slurm/setup_env.sh installs it in ~/.local/bin"
        kit_say "building the Python env (first run only; detectron2 + detrex compile, ~5-15 min)"
        bash slurm/setup_env.sh || kit_die "env setup failed, see the messages above"
        kit_python -c "import irwm.detection" || kit_die "the env still does not import irwm.detection"
    fi

    local need
    need="$(kit_python - <<'PY'
from irwm.utils.paths import load_env
load_env()
from irwm.worldmodel import assets as A
files = [A.dinov2_code_dir() / "hubconf.py", A.dinov2_weights_path(), A.dinof_dir() / A.PCA_EXPORT]
print(" ".join(str(f) for f in files if not f.is_file()))
PY
)"
    if [[ -n "${need}" ]]; then
        kit_say "downloading the frozen encoder (DINOv2-Reg ViT-B/14 + DINO-Foresight PCA) into ${IRWM_CKPT_ROOT}"
        kit_python projects/dinof_audit/download.py "hf_files=['dinov2_pca_448_l[2_5_8_11]_1152.pth']" \
            || kit_die "encoder download failed (needs internet: run it once on a login node)"
    fi
}
