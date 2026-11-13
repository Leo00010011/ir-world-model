# shellcheck shell=bash
# Copy datasets to node-local scratch and point IRWM_DATA_ROOT at the copy (R11).
# SOURCE it inside a job, after slurm/env.sh:
#   source slurm/env.sh && source slurm/stage_data.sh
# IRWM_STAGE_DATASETS picks what is copied: a space-separated list of dirs under $IRWM_DATA_ROOT.
# The default is FLIR v2 alone (unchanged behaviour). Known entries:
#   teledyne_flir_adas_v2   FLIR ADAS v2 (all splits + top-level files)
#   aligned/align           aligned FLIR v1 (F16): split lists, JPEGImages/, Annotations/*.xml;
#                           AnnotatedImages/ and Annotations/*_mask.jpg are not read, so not copied
# e.g. IRWM_STAGE_DATASETS="aligned/align" source slurm/stage_data.sh
# Afterwards IRWM_DATA_ROOT holds ONLY the staged datasets: stage everything the job reads.
# $LOCAL_SCRATCH is wiped when the job ends: the copy is read-only input, never write results there.
# Subdirectories are copied in parallel (the shared filesystem is latency-bound on small files).

# _irwm_copy_parallel <src> <dst> <tar args...> -- <subdir>...: one tar pipe per subdir, in parallel.
_irwm_copy_parallel() {
    local src="$1" dst="$2"; shift 2
    local targs=() d pids=() rc=0
    while [[ "$1" != "--" ]]; do targs+=("$1"); shift; done
    shift
    for d in "$@"; do
        ( tar -C "${src}" "${targs[@]}" -cf - "${d}" | tar -C "${dst}" -xf - ) &
        pids+=($!)
    done
    for d in "${pids[@]}"; do wait "${d}" || rc=1; done
    return "${rc}"
}

# _irwm_stage_one <name> <src> <dst>: copy one dataset; prints the expected file count on stdout.
_irwm_stage_one() {
    local name="$1" src="$2" dst="$3" d subdirs=()
    case "${name}" in
        teledyne_flir_adas_v2)
            mkdir -p "${dst}/FLIR_ADAS_v2"
            # Top-level files (README, rgb_to_thermal_vid_map.json, ...), then the split dirs.
            find "${src}" -maxdepth 1 -type f -exec cp -p {} "${dst}/" \; >&2
            for d in "${src}"/FLIR_ADAS_v2/*/; do subdirs+=("$(basename "${d}")"); done
            _irwm_copy_parallel "${src}/FLIR_ADAS_v2" "${dst}/FLIR_ADAS_v2" -- "${subdirs[@]}" >&2 || return 1
            find "${src}" -type f | wc -l
            ;;
        aligned/align)
            mkdir -p "${dst}"
            find "${src}" -maxdepth 1 -type f -name '*.txt' -exec cp -p {} "${dst}/" \; >&2
            _irwm_copy_parallel "${src}" "${dst}" --exclude='*_mask.jpg' -- JPEGImages Annotations >&2 || return 1
            echo $(( $(find "${src}" -maxdepth 1 -type f -name '*.txt' | wc -l)
                     + $(find "${src}/JPEGImages" -type f | wc -l)
                     + $(find "${src}/Annotations" -type f ! -name '*_mask.jpg' | wc -l) ))
            ;;
        *)
            echo "[stage] unknown dataset '${name}'; known: teledyne_flir_adas_v2 aligned/align" >&2
            return 1
            ;;
    esac
}

_irwm_need_kb() {  # _irwm_need_kb <name> <src>: KB that will be copied
    case "$1" in
        aligned/align) du -sck "$2"/*.txt "$2/JPEGImages" "$2/Annotations" | tail -1 | cut -f1 ;;
        *) du -sk "$2" | cut -f1 ;;
    esac
}

_irwm_stage() {
    local src_root="${IRWM_DATA_ROOT:?IRWM_DATA_ROOT is not set (source slurm/env.sh first)}"
    local scratch="${LOCAL_SCRATCH:?LOCAL_SCRATCH is not set (not on a compute node?)}"
    local datasets="${IRWM_STAGE_DATASETS:-teledyne_flir_adas_v2}"
    local name src dst need_kb free_kb t0 dt n_exp n_dst

    for name in ${datasets}; do
        src="${src_root}/${name}" dst="${scratch}/irwm_data/${name}"
        [[ -d "${src}" ]] || { echo "[stage] ${src} does not exist" >&2; return 1; }

        need_kb="$(_irwm_need_kb "${name}" "${src}")"
        free_kb="$(df -Pk "${scratch}" | awk 'NR==2 {print $4}')"
        if (( free_kb < need_kb * 11 / 10 )); then
            echo "[stage] not enough space in ${scratch}: need ${need_kb} KB, free ${free_kb} KB" >&2
            return 1
        fi

        echo "[stage] ${src} -> ${dst} ($((need_kb / 1024)) MB)"
        t0=${SECONDS}
        n_exp="$(_irwm_stage_one "${name}" "${src}" "${dst}")" || { echo "[stage] copy failed" >&2; return 1; }

        n_dst="$(find "${dst}" -type f | wc -l)"
        if [[ "${n_exp}" != "${n_dst}" ]]; then
            echo "[stage] file count mismatch for ${name}: ${n_exp} expected, ${n_dst} copied" >&2
            return 1
        fi
        dt=$(( SECONDS - t0 ))
        echo "[stage] copied ${n_dst} files in ${dt} s ($(( need_kb / 1024 / (dt > 0 ? dt : 1) )) MB/s)"
    done

    export IRWM_DATA_ROOT_SHARED="${src_root}"
    export IRWM_DATA_ROOT="${scratch}/irwm_data"
    echo "[stage] IRWM_DATA_ROOT=${IRWM_DATA_ROOT} (staged: ${datasets})"
}

_irwm_stage
