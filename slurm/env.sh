# shellcheck shell=bash
# Environment for every cluster-side command. SOURCE it (do not execute it):
#   source slurm/env.sh
# Used by setup_env.sh, submit.sh, job.sbatch and submitit jobs (hydra.launcher.setup).
# It does three things and never overrides variables that are already set:
#   1. puts ~/.local/bin (where `pip install --user uv` / the uv installer put uv) on PATH
#   2. loads <repo>/.env (KEY=VALUE lines; `export`, quotes, comments, CRLF and ~ are handled)
#   3. exports WANDB_API_KEY from $WANDB_API_KEY_FILE (default ~/.wandb_api_key) if it is unset

_irwm_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

if [[ -f "${_irwm_root}/.env" ]]; then
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        _line="${_line%$'\r'}"
        _line="${_line#"${_line%%[![:space:]]*}"}"          # ltrim
        [[ -z "${_line}" || "${_line}" == \#* || "${_line}" != *=* ]] && continue
        _line="${_line#export }"
        _key="${_line%%=*}"; _key="${_key//[[:space:]]/}"
        _val="${_line#*=}"
        if [[ "${_val}" == \"*\" || "${_val}" == \'*\' ]]; then
            _val="${_val:1:${#_val}-2}"
        else
            _val="${_val%%#*}"                                  # inline comment
            _val="${_val%"${_val##*[![:space:]]}"}"             # rtrim
        fi
        [[ "${_val}" == "~" || "${_val}" == "~/"* ]] && _val="${HOME}${_val:1}"
        [[ "${_key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -z "${!_key:-}" ]] && export "${_key}=${_val}"
    done < "${_irwm_root}/.env"
fi

_key_file="${WANDB_API_KEY_FILE:-${HOME}/.wandb_api_key}"
if [[ -z "${WANDB_API_KEY:-}" && -r "${_key_file}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${_key_file}")"
    export WANDB_API_KEY
fi

unset _irwm_root _line _key _val _key_file
