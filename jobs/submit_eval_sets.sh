#!/bin/bash

# Submit the eval-dataset download array job.
#
# Usage:
#   ./jobs/submit_eval_sets.sh                                  # all sets, 4 tasks
#   DATASETS="voxpopuli_cleaned_aa" ./jobs/submit_eval_sets.sh  # one set first
#   N_JOBS=8 ./jobs/submit_eval_sets.sh
#
# ~18.6 GB across 407 files. LibriSpeech IS included even though openslr has a copy: the
# audio path reads it from hf-audio, and leaving it out left the eval tasks racing to
# build its Arrow cache themselves.
#
# Two stages: the download array, then a single-task Arrow cache build chained with
# --dependency=afterok. The second stage is not optional — downloading fills the *hub*
# cache, but load_dataset then builds a separate Arrow cache, and eight eval tasks doing
# that conversion concurrently corrupt each other on GPFS.

set -eo pipefail

DATASETS="${DATASETS:-ami_cleaned gigaspeech_cleaned spgispeech voxpopuli_cleaned_aa librispeech_test.clean librispeech_test.other earnings22_cleaned_aa_chunked}"
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
ARRAY_JOB="$(sbatch --parsable \
    --array="0-$((N_JOBS - 1))" \
    --time="${WALLTIME}" \
    --export=ALL,DATASETS="${DATASETS}" \
    "${SCRIPT_DIR}/download_eval_sets.sh")"

# Single task, and only if every download succeeded: a partial download would build a
# cache over missing shards.
CACHE_JOB="$(sbatch --parsable \
    --dependency=afterok:"${ARRAY_JOB}" \
    --export=ALL,REPO_HOME="${REPO_DIR}",DATASETS="${DATASETS}" \
    "${SCRIPT_DIR}/build_eval_cache.sh")"

echo
echo "download array : ${ARRAY_JOB}"
echo "cache build    : ${CACHE_JOB}  (afterok:${ARRAY_JOB})"
echo
echo "Wait for BOTH before running ./jobs/submit_eval.sh — the eval assumes a warm"
echo "Arrow cache and will race if the build has not finished."
