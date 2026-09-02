#!/bin/bash

# Submit the eval-dataset download array job.
#
# Usage:
#   ./jobs/submit_eval_sets.sh                                  # all sets, 4 tasks
#   DATASETS="voxpopuli_cleaned_aa" ./jobs/submit_eval_sets.sh  # one set first
#   N_JOBS=8 ./jobs/submit_eval_sets.sh
#
# ~18 GB total across 405 files. LibriSpeech is not included: it is already downloaded
# from openslr, and its normalized references were verified identical (2620/2620) to this
# repo's copy.
#
# Afterwards the configs are fully cached, which is what makes data_utils.load_data a pure
# cache hit inside the eval — it resolves a whole config, so a partial cache would trigger
# a re-download from every array task at once.

set -eo pipefail

DATASETS="${DATASETS:-ami_cleaned gigaspeech_cleaned spgispeech voxpopuli_cleaned_aa earnings22_cleaned_aa_chunked}"
N_JOBS="${N_JOBS:-4}"

# Wall-time model, matching submit_librispeech.sh: 10 MB/s is a pessimistic floor measured
# on a home connection, whereas a Snellius staging node is far faster. Over-booking is
# free; under-booking loses the transfer, so the estimate is doubled on top.
THROUGHPUT_MB_S="${THROUGHPUT_MB_S:-10}"
SAFETY="${SAFETY:-2}"
OVERHEAD_S="${OVERHEAD_S:-600}"
MIN_S=1800                        # 30 min
MAX_S=28800                       # 8 h, the staging partition limit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

if [[ -z "${WALLTIME}" ]]; then
    # Shard 0 takes ceil(n/N_JOBS) files, so it is the worst-case task.
    # `|| true` is load-bearing: under `set -eo pipefail` a failing python would abort the
    # script instead of reaching the fallback.
    BYTES="$(cd "${REPO_DIR}" && python -m scripts.download_eval_sets \
        --dry-run --env snellius --datasets ${DATASETS} \
        --num-shards "${N_JOBS}" --shard-id 0 2>/dev/null \
        | awk '/^bytes/ {print $2}' || true)"

    if [[ -z "${BYTES}" ]]; then
        WALLTIME="$(fmt_hms ${MAX_S})"
        echo "WARNING: could not size the download (is the env active?)," \
             "falling back to ${WALLTIME}" >&2
    else
        EST_S=$((BYTES / (THROUGHPUT_MB_S * 1000000)))
        WALL_S=$((OVERHEAD_S + EST_S * SAFETY))
        ((WALL_S < MIN_S)) && WALL_S=${MIN_S}
        ((WALL_S > MAX_S)) && WALL_S=${MAX_S}
        WALLTIME="$(fmt_hms ${WALL_S})"

        printf 'per-task  : %.2f GB (worst case)\n' "$(echo "${BYTES}" | awk '{print $1/1e9}')"
        echo "estimate  : ${EST_S}s at ${THROUGHPUT_MB_S} MB/s, x${SAFETY} + ${OVERHEAD_S}s overhead"
    fi
fi

echo "datasets  : ${DATASETS}"
echo "tasks     : ${N_JOBS}"
echo "walltime  : ${WALLTIME}"

# --time here overrides the #SBATCH directive in download_eval_sets.sh.
sbatch \
    --array="0-$((N_JOBS - 1))" \
    --time="${WALLTIME}" \
    --export=ALL,DATASETS="${DATASETS}" \
    "${SCRIPT_DIR}/download_eval_sets.sh"
