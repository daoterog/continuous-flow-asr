# stage_features CONFIG [train.py args...]: copy every feature directory that run reads onto
# this node's local NVMe, and set STAGED_ROOT to the copy's data root. Source it.
#
# train.960's 1.1 TB does not fit in RAM, and GPFS gives one node 0.2-0.8 GB/s of random
# reads whatever the loader does (mmfsd at 718% CPU with 256 reads in flight, the GPU 0-10%
# busy). A bulk sequential copy is what GPFS streams best, and after it every read is local.
# The volume is /scratch-node/$USER.$SLURM_JOB_ID (1.8 TB for a 1-GPU H100 job, 2026-10-20),
# deleted when the job ends; it exists only under --constraint=scratch-node.
#
# Used by jobs/batch/train.sh (STAGE_DATA=1) and jobs/batch/bench_train.sh (SUITE=now), so a
# benchmark reads the data exactly as a training run does.

stage_features() {
    local config="$1"
    shift
    STAGE_DIR="${STAGE_DIR:-/scratch-node/${USER}.${SLURM_JOB_ID}}"
    if [[ ! -d "${STAGE_DIR}" ]]; then
        echo "ERROR: STAGE_DATA needs node-local NVMe at ${STAGE_DIR}; submit with" >&2
        echo "       --constraint=scratch-node (SCRATCH_NODE=1 ./jobs/submit/train.sh ...)" >&2
        exit 1
    fi
    mapfile -t LISTING < <(python train.py --config "${config}" "$@" --print-data-dirs)
    SRC_ROOT="$(printf '%s\n' "${LISTING[@]}" | awk '$1 == "root" {print $2}')"
    mapfile -t SRC_DIRS < <(printf '%s\n' "${LISTING[@]}" | awk '$1 == "dir" {print $2}')
    if [[ -z "${SRC_ROOT}" || ${#SRC_DIRS[@]} -eq 0 ]]; then
        echo "ERROR: could not list the run's feature directories" >&2
        exit 1
    fi
    DST_ROOT="${STAGE_DIR}/data"

    # Summed file sizes, not du: du also counts directory entries, which differ in number.
    need=$(find "${SRC_DIRS[@]}" -maxdepth 1 -type f -printf "%s\n" | awk '{s += $1} END {print s}')
    have=$(df -B1 --output=avail "${STAGE_DIR}" | tail -1)
    echo "stage     : ${#SRC_DIRS[@]} directories, $((need / 2**30)) GiB -> ${DST_ROOT}" \
        "($((have / 2**30)) GiB free)"
    if ((need > have)); then
        echo "ERROR: the features do not fit on ${STAGE_DIR}" >&2
        exit 1
    fi

    t0=$(date +%s)
    for d in "${SRC_DIRS[@]}"; do
        mkdir -p "${DST_ROOT}/${d#"${SRC_ROOT}"/}"
    done
    # One cp per file, 16 at a time: shards are ~10 GB each, and parallel sequential
    # streams are what GPFS delivers fastest.
    for d in "${SRC_DIRS[@]}"; do
        find "${d}" -maxdepth 1 -type f -printf "%p\0${DST_ROOT}/${d#"${SRC_ROOT}"/}/\0"
    done | xargs -0 -n 2 -P 16 cp
    t1=$(date +%s)

    got=$(find "${DST_ROOT}" -type f -printf "%s\n" | awk '{s += $1} END {print s}')
    if ((got != need)); then
        echo "ERROR: staged ${got} bytes, expected ${need}" >&2
        exit 1
    fi
    secs=$((t1 - t0 > 0 ? t1 - t0 : 1))
    echo "stage     : copied in ${secs}s, $(awk -v b="${need}" -v s="${secs}" \
        'BEGIN {printf "%.2f", b / s / 1e9}') GB/s"

    STAGED_ROOT="${DST_ROOT}"
}
