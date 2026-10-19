# shellcheck shell=bash
# Copy the FLIR dataset to node-local scratch and point IRWM_DATA_ROOT at the copy (R11).
# SOURCE it inside a job, after slurm/env.sh:
#   source slurm/env.sh && source slurm/stage_data.sh
# $LOCAL_SCRATCH is wiped when the job ends: the copy is read-only input, never write results there.
# The six split dirs are copied in parallel (the shared filesystem is latency-bound on small files).

_irwm_stage() {
    local src_root="${IRWM_DATA_ROOT:?IRWM_DATA_ROOT is not set (source slurm/env.sh first)}"
    local scratch="${LOCAL_SCRATCH:?LOCAL_SCRATCH is not set (not on a compute node?)}"
    local name="teledyne_flir_adas_v2"
    local src="${src_root}/${name}" dst="${scratch}/irwm_data/${name}"
    [[ -d "${src}" ]] || { echo "[stage] ${src} does not exist" >&2; return 1; }

    local need_kb free_kb
    need_kb="$(du -sk "${src}" | cut -f1)"
    free_kb="$(df -Pk "${scratch}" | awk 'NR==2 {print $4}')"
    if (( free_kb < need_kb * 11 / 10 )); then
        echo "[stage] not enough space in ${scratch}: need ${need_kb} KB, free ${free_kb} KB" >&2
        return 1
    fi

    echo "[stage] ${src} -> ${dst} ($((need_kb / 1024)) MB)"
    local t0=${SECONDS}
    mkdir -p "${dst}/FLIR_ADAS_v2"
    # Top-level files (README, rgb_to_thermal_vid_map.json, ...), then the split dirs in parallel.
    find "${src}" -maxdepth 1 -type f -exec cp -p {} "${dst}/" \;
    local d pids=() rc=0
    for d in "${src}"/FLIR_ADAS_v2/*/; do
        d="$(basename "${d}")"
        ( tar -C "${src}/FLIR_ADAS_v2" -cf - "${d}" | tar -C "${dst}/FLIR_ADAS_v2" -xf - ) &
        pids+=($!)
    done
    for d in "${pids[@]}"; do wait "${d}" || rc=1; done
    (( rc == 0 )) || { echo "[stage] copy failed" >&2; return 1; }

    local n_src n_dst
    n_src="$(find "${src}" -type f | wc -l)"
    n_dst="$(find "${dst}" -type f | wc -l)"
    if [[ "${n_src}" != "${n_dst}" ]]; then
        echo "[stage] file count mismatch: ${n_src} in source, ${n_dst} copied" >&2
        return 1
    fi
    local dt=$(( SECONDS - t0 ))
    echo "[stage] copied ${n_dst} files in ${dt} s ($(( need_kb / 1024 / (dt > 0 ? dt : 1) )) MB/s)"

    export IRWM_DATA_ROOT_SHARED="${src_root}"
    export IRWM_DATA_ROOT="${scratch}/irwm_data"
    echo "[stage] IRWM_DATA_ROOT=${IRWM_DATA_ROOT}"
}

_irwm_stage
