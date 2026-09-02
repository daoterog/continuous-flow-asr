#!/bin/bash

# Submit the Whisper encoder feature precompute array job.
#
# Usage:
#   ./jobs/submit/whisper.sh                                # train.100, 7 tasks
#   N_JOBS=14 ./jobs/submit/whisper.sh                      # one task per shard
#   SPLITS="validation" ./jobs/submit/whisper.sh            # small run first
#   MODEL=openai/whisper-small ./jobs/submit/whisper.sh
#   WALLTIME=02:00:00 ./jobs/submit/whisper.sh              # skip the estimate
#
# train.clean.100 is 28,539 utterances across 14 shards, ~46 GB of trimmed float16
# features. The shard is the unit of parallelism, so N_JOBS above 14 gains nothing.

set -eo pipefail

CONFIGS="${CONFIGS:-clean}"
SPLITS="${SPLITS:-train.100}"
MODEL="${MODEL:-openai/whisper-large-v3}"
N_JOBS="${N_JOBS:-7}"

# Batch size. Measured: 1.19 GiB of fp16 encoder weights + ~45 MiB of activations per
# utterance, so batch 128 is ~6.8 GiB — far under a 40 GB A100, which could fit ~790.
# The real ceiling is diminishing returns: an RTX 3060 plateaued at batch 16, and an
# A100 is roughly 8-15x faster, putting saturation near 128-256.
BATCH_SIZE="${BATCH_SIZE:-128}"
# Set NO_TRIM=1 for splits that will be decoded back through Whisper (the eval sets).
NO_TRIM="${NO_TRIM:-}"

# Wall-time model. The encode is GPU-bound (an isolated benchmark matched the
# end-to-end rate), and an RTX 3060 managed 8.1 utts/s. 30 utts/s assumes an A100 is
# only ~4x that — deliberately pessimistic against the 8-15x the specs suggest.
# Over-booking is free; under-booking loses the work, so the estimate is doubled again.
UTTS_PER_SEC="${UTTS_PER_SEC:-30}"
SAFETY="${SAFETY:-2}"
OVERHEAD_S="${OVERHEAD_S:-900}"   # module load, conda activate, ~3 GB of Whisper weights
MIN_S=1800                        # 30 min
MAX_S=86400                       # 24 h, the gpu partition limit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
BATCH_DIR="${REPO_DIR}/jobs/batch"

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

if [[ -z "${WALLTIME}" ]]; then
    # Shard 0 takes ceil(n/N_JOBS) shards, so it is the worst-case task. --dry-run
    # reports the utterance count without loading the model.
    # `|| true` is load-bearing: under `set -eo pipefail` a failing python would abort
    # the script instead of reaching the fallback.
    UTTS="$(cd "${REPO_DIR}" && python -m scripts.precompute_whisper \
        --dry-run --env snellius --model "${MODEL}" \
        --configs ${CONFIGS} --splits ${SPLITS} \
        --num-shards "${N_JOBS}" --shard-id 0 2>/dev/null \
        | awk '/^utterances/ {print $2}' || true)"

    if [[ -z "${UTTS}" ]]; then
        WALLTIME="$(fmt_hms ${MAX_S})"
        echo "WARNING: could not size the job (is the env active? is the data downloaded?)," \
             "falling back to ${WALLTIME}" >&2
    else
        EST_S=$(awk -v u="${UTTS}" -v r="${UTTS_PER_SEC}" 'BEGIN{printf "%d", u/r}')
        WALL_S=$((OVERHEAD_S + EST_S * SAFETY))
        ((WALL_S < MIN_S)) && WALL_S=${MIN_S}
        ((WALL_S > MAX_S)) && WALL_S=${MAX_S}
        WALLTIME="$(fmt_hms ${WALL_S})"

        echo "per-task  : ${UTTS} utterances (worst case)"
        echo "estimate  : ${EST_S}s at ${UTTS_PER_SEC} utts/s, x${SAFETY} + ${OVERHEAD_S}s overhead"
    fi
fi

echo "model     : ${MODEL}"
echo "configs   : ${CONFIGS}"
echo "splits    : ${SPLITS}"
echo "tasks     : ${N_JOBS}"
echo "trim      : $([[ -n "${NO_TRIM}" ]] && echo "off (full 1500 frames)" || echo "on")"
echo "walltime  : ${WALLTIME}"

# --time here overrides the #SBATCH directive in precompute_whisper.sh.
sbatch \
    --array="0-$((N_JOBS - 1))" \
    --time="${WALLTIME}" \
    --export=ALL,CONFIGS="${CONFIGS}",SPLITS="${SPLITS}",MODEL="${MODEL}",BATCH_SIZE="${BATCH_SIZE}",NO_TRIM="${NO_TRIM}" \
    "${BATCH_DIR}/precompute_whisper.sh"
