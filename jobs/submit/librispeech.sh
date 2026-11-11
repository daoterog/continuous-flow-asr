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

SIZING_ERR="$(mktemp)"
trap 'rm -f "${SIZING_ERR}"' EXIT

if [[ -z "${WALLTIME}" ]]; then
    # Shard 0 takes ceil(n/N_JOBS) files, so it is the worst-case task.
    # `|| true` is load-bearing: under `set -eo pipefail` a failing python (env not
    # active, no network) would abort the script instead of reaching the fallback.
    SHARD_BYTES="$(cd "${REPO_DIR}" && python -m scripts.data.download_librispeech \
        --dry-run --configs ${CONFIGS} --splits ${SPLITS} \
        --num-shards "${N_JOBS}" --shard-id 0 2>"${SIZING_ERR}" \
        | awk '/^shard bytes/ {print $3}' || true)"

    if [[ -z "${SHARD_BYTES}" ]]; then
        # Do NOT fall back to the partition maximum. A maximal request is not the safe
        # choice: it queues far longer than a right-sized one, and on the first run
        # sizing legitimately cannot work (whisper.sh counts utterances from data that
        # is not downloaded yet). Fall back to a modest default and say what broke, so
        # the fix is to pass WALLTIME= rather than to wait behind a 24h reservation.
        echo "ERROR: could not size the job. Reason:" >&2
        sed 's/^/  | /' "${SIZING_ERR}" >&2
        echo "  Pass an explicit walltime instead, e.g. WALLTIME=01:00:00 $0" >&2
        WALLTIME="$(fmt_hms 3600)"
        echo "  Falling back to ${WALLTIME} (NOT the partition maximum)." >&2
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
