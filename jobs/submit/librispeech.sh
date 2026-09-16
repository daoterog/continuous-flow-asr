#!/bin/bash

# Submit the LibriSpeech download array job.
#
# Usage:
#   ./jobs/submit/librispeech.sh                            # everything, 8 tasks
#   N_JOBS=4 ./jobs/submit/librispeech.sh                   # everything, 4 tasks
#   SPLITS="validation test" ./jobs/submit/librispeech.sh   # small run first
#   CONFIGS="clean" SPLITS="train.100" ./jobs/submit/librispeech.sh
#   WALLTIME=02:00:00 ./jobs/submit/librispeech.sh          # skip the estimate
#
# The full corpus (clean + other) is ~62 GB across 130 shards.  Downloading the
# "all" config is deliberately not an option: it is the same audio under merged
# split names, so it would fetch everything twice.

set -eo pipefail

CONFIGS="${CONFIGS:-clean other}"
SPLITS="${SPLITS:-train.100 train.360 train.500 validation test}"
N_JOBS="${N_JOBS:-8}"

# Wall-time model.  Throughput is per array task, and 10 MB/s is a deliberately
# pessimistic floor — it is what a home connection measured, whereas a Snellius
# staging node is far faster.  Over-booking is free (the job just ends early);
# under-booking loses the whole transfer, so the estimate is doubled on top.
THROUGHPUT_MB_S="${THROUGHPUT_MB_S:-10}"
SAFETY="${SAFETY:-2}"
OVERHEAD_S="${OVERHEAD_S:-600}"   # module load, conda activate, HF metadata, retries
MIN_S=1800                        # 30 min
MAX_S=28800                       # 8 h, the staging partition limit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
BATCH_DIR="${REPO_DIR}/jobs/batch"

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

if [[ -z "${WALLTIME}" ]]; then
    # Shard 0 takes ceil(n/N_JOBS) files, so it is the worst-case task.
    # `|| true` is load-bearing: under `set -eo pipefail` a failing python (env not
    # active, no network) would abort the script instead of reaching the fallback.
    SHARD_BYTES="$(cd "${REPO_DIR}" && python -m scripts.download_librispeech \
        --dry-run --configs ${CONFIGS} --splits ${SPLITS} \
        --num-shards "${N_JOBS}" --shard-id 0 2>/dev/null \
        | awk '/^shard bytes/ {print $3}' || true)"

    if [[ -z "${SHARD_BYTES}" ]]; then
        WALLTIME="$(fmt_hms ${MAX_S})"
        echo "WARNING: could not size the download (is the env active?), " \
             "falling back to ${WALLTIME}" >&2
    else
        EST_S=$((SHARD_BYTES / (THROUGHPUT_MB_S * 1000000)))
        WALL_S=$((OVERHEAD_S + EST_S * SAFETY))
        ((WALL_S < MIN_S)) && WALL_S=${MIN_S}
        ((WALL_S > MAX_S)) && WALL_S=${MAX_S}
        WALLTIME="$(fmt_hms ${WALL_S})"

        printf 'per-task  : %.2f GB (worst case)\n' "$(echo "${SHARD_BYTES}" | awk '{print $1/1e9}')"
        echo "estimate  : ${EST_S}s at ${THROUGHPUT_MB_S} MB/s, x${SAFETY} + ${OVERHEAD_S}s overhead"
    fi
fi

echo "configs   : ${CONFIGS}"
echo "splits    : ${SPLITS}"
echo "tasks     : ${N_JOBS}"
echo "walltime  : ${WALLTIME}"

# --time here overrides the #SBATCH directive in download_librispeech.sh.
sbatch \
    --array="0-$((N_JOBS - 1))" \
    --time="${WALLTIME}" \
    --export=ALL,CONFIGS="${CONFIGS}",SPLITS="${SPLITS}" \
    "${BATCH_DIR}/download_librispeech.sh"
