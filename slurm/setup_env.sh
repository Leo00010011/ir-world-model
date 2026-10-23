#!/usr/bin/env bash
# One-shot, idempotent environment setup for any cluster. Run it on a LOGIN node (needs internet),
# from anywhere inside the clone:
#   bash slurm/setup_env.sh            # install uv if needed, sync env, verify
#   bash slurm/setup_env.sh --test     # ... and run the CPU test suite
# Safe to re-run after `git pull` (re-syncs from uv.lock).
#
# Small home quota? Point uv's cache (several GB, mostly torch) elsewhere first, e.g. in ~/.bashrc:
#   export UV_CACHE_DIR=/scratch/$USER/uv-cache
# Python 3.11 is downloaded by uv into ~/.local/share/uv/python (override: UV_PYTHON_INSTALL_DIR).
set -euo pipefail

run_tests=0
for arg in "$@"; do
    case "${arg}" in
        --test) run_tests=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$(pwd)"
step() { printf '\n\033[1m[setup] %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[setup] WARNING: %s\033[0m\n' "$*" >&2; }

step "repo: ${repo} @ $(git rev-parse --short HEAD 2>/dev/null || echo '?') on $(hostname)"

# 1. .env -------------------------------------------------------------------------------------------
if [[ ! -f .env ]]; then
    cp .env.example .env
    warn "created .env from .env.example. Edit the IRWM_* paths for this cluster (scratch, not home), then re-run."
fi
source slurm/env.sh

# 2. Submodules -------------------------------------------------------------------------------------
step "submodules"
git submodule update --init --recursive
git submodule status

# 3. uv ---------------------------------------------------------------------------------------------
step "uv"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing in user space"
    if command -v python3 >/dev/null 2>&1 && python3 -m pip install --user --quiet uv; then
        echo "installed with pip --user"
    else
        # PEP 668 ("externally managed") systems or no pip: the official standalone installer,
        # which also needs no admin rights and installs into ~/.local/bin.
        warn "pip --user failed; using the standalone installer"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    fi
    hash -r
fi
command -v uv >/dev/null 2>&1 || { echo "uv still not on PATH (expected in ~/.local/bin)" >&2; exit 1; }
uv --version
echo "uv cache: $(uv cache dir)"
if ! grep -qs '.local/bin' "${HOME}/.bashrc" "${HOME}/.bash_profile" "${HOME}/.profile"; then
    warn "~/.local/bin is not added to PATH in your shell rc. Repo scripts handle this, but for an interactive uv add:  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# 4. Environment ------------------------------------------------------------------------------------
step "uv sync --frozen (Python 3.11 + torch cu121 from uv.lock)"
# Two phases: detectron2/detrex build without isolation against the installed torch (F7).
uv sync --frozen --no-install-package detectron2 --no-install-package detrex
reinstall=()
if command -v nvcc >/dev/null 2>&1; then
    # Login nodes have no GPU, so force the CUDA ops (detrex setup.py:106-109); archs: V100S, L4, others.
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"
    reinstall=(--reinstall-package detectron2 --reinstall-package detrex)
    echo "  building detectron2/detrex CUDA ops: $(nvcc --version | tail -1) | $(gcc --version | head -1)"
else
    warn "nvcc not found: detectron2/detrex get CPU-only ops, so GPU training will fail (load a CUDA 12.x module and re-run)"
fi
build_start=${SECONDS}
uv sync --frozen ${reinstall[@]+"${reinstall[@]}"}  # empty-array safe under set -u
echo "  phase-2 sync (detectron2/detrex build): $((SECONDS - build_start)) s"

# 5. Paths ------------------------------------------------------------------------------------------
step "IRWM_* paths"
missing=0
for var in IRWM_DATA_ROOT IRWM_CACHE_ROOT IRWM_CKPT_ROOT IRWM_OUTPUT_ROOT; do
    val="${!var:-}"
    if [[ -z "${val}" ]]; then
        warn "${var} is not set (add it to .env)"; missing=1; continue
    fi
    if [[ "${var}" != IRWM_DATA_ROOT ]]; then mkdir -p "${val}"; fi
    printf '  %-17s %s %s\n' "${var}" "${val}" "$([[ -d "${val}" ]] && echo '' || echo '(does not exist yet)')"
done

# 6. Verification -----------------------------------------------------------------------------------
step "verify"
uv run --frozen python - <<'PY'
import sys, torch, numpy, lightning, hydra, irwm
print(f"  python {sys.version.split()[0]} | torch {torch.__version__} (CUDA build {torch.version.cuda}) "
      f"| numpy {numpy.__version__} | lightning {lightning.__version__} | hydra {hydra.__version__}")
import detectron2, detrex
from detrex import _C
print(f"  detectron2 {detectron2.__version__} | detrex _C ops: {_C.__file__}")
if torch.cuda.is_available():
    print(f"  CUDA available: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
else:
    print("  CUDA not available here (normal on a login node; the smoke job checks the GPU nodes)")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    [[ -n "${driver}" ]] && echo "  NVIDIA driver ${driver} (cu121 wheels need >= 525.60)"
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    if uv run --frozen python -c "import os, wandb; wandb.login(key=os.environ['WANDB_API_KEY'], verify=True, relogin=True)" >/dev/null 2>&1; then
        echo "  W&B: key verified"
    else
        warn "W&B: key present but verification failed (no internet here, or invalid key)"
    fi
else
    warn "W&B: no WANDB_API_KEY and no key file at ${WANDB_API_KEY_FILE:-~/.wandb_api_key}"
fi

# 7. Tests ------------------------------------------------------------------------------------------
if [[ "${run_tests}" == 1 ]]; then
    step "pytest (CPU)"
    uv run --frozen pytest -q
fi

# 8. Cluster config ---------------------------------------------------------------------------------
step "done"
sites="$(cd lib/irwm/conf/cluster && ls *.yaml | sed 's/\.yaml$//' | grep -vE '^(local|slurm|example_site)$' | tr '\n' ' ' || true)"
echo "  cluster site configs: ${sites:-none yet (copy lib/irwm/conf/cluster/example_site.yaml)}"
echo "  next: slurm/submit.sh <site> projects/smoke/train.py"
[[ "${missing}" == 0 ]] || exit 1
