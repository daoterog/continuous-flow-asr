#!/bin/bash

# Submit one training run, with a walltime derived from the config's own step count.
#
# Usage:
#   ./jobs/submit/train.sh configs/linear.yaml
#   ./jobs/submit/train.sh configs/audio_cond.yaml train.num_epochs=20
#   STEPS_PER_SEC=12 ./jobs/submit/train.sh configs/linear.yaml     # after measuring
#   WALLTIME=02:00:00 ./jobs/submit/train.sh configs/linear.yaml    # skip the estimate
#
# Everything after the config is passed through to train.py as OmegaConf overrides.

set -eo pipefail

CONFIG="${1:?usage: $0 <config.yaml> [overrides...]}"
shift || true
OVERRIDES="$*"

# Throughput, measured on an A100 node (scripts/bench_train.py, batch 256 + 16 workers +
# compile). Two regimes, 15x apart, and which one applies is decided by whether the data
# fits the job's 64G of RAM:
#   ~3350 utts/s  reading a subset the page cache holds
#   ~220 utts/s   reading more than it holds — ~800 MiB/s off GPFS, the real ceiling
# train.100 is 102 GiB untrimmed, so a full-data run is the second regime. Size off it.
#
# This is a utts/s ceiling, so steps/s falls as the batch grows: STEPS_PER_SEC ~ 220/batch.
# The default suits the default batch of 32; at batch 256 pass STEPS_PER_SEC=0.9.
STEPS_PER_SEC="${STEPS_PER_SEC:-6}"
SAFETY="${SAFETY:-1.3}"
OVERHEAD_S="${OVERHEAD_S:-600}"   # module load, conda, tokenizer, tau sweep, first epoch
MIN_S=900                         # 15 min
MAX_S=432000                      # 5 days, the gpu_a100 limit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
BATCH_DIR="${REPO_DIR}/jobs/batch"

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

SIZING_ERR="$(mktemp)"
trap 'rm -f "${SIZING_ERR}"' EXIT

if [[ -z "${WALLTIME}" ]]; then
    # --print-steps builds the data and counts batches; no model, no GPU, so it runs
    # here on the login node. It needs the features to exist, which they must anyway.
    TOTAL_STEPS="$(cd "${REPO_DIR}" && python train.py --config "${CONFIG}" \
        data.env=snellius ${OVERRIDES} --print-steps 2>"${SIZING_ERR}" \
        | awk '/^total_steps/ {print $2}' || true)"

    if [[ -z "${TOTAL_STEPS}" ]]; then
        # Loud, and not the partition maximum: a 5-day request would sit in the queue
        # far longer than the run needs, and being wrong in that direction is penalised.
        echo "ERROR: could not size the run. Reason:" >&2
        sed 's/^/  | /' "${SIZING_ERR}" >&2
        echo "  Pass an explicit walltime, e.g. WALLTIME=02:00:00 $0 ${CONFIG}" >&2
        exit 1
    fi

    EST_S=$(awk -v s="${TOTAL_STEPS}" -v r="${STEPS_PER_SEC}" 'BEGIN{printf "%d", s/r}')
    WALL_S=$(awk -v e="${EST_S}" -v f="${SAFETY}" -v o="${OVERHEAD_S}" \
        'BEGIN{printf "%d", o + e*f}')
    ((WALL_S < MIN_S)) && WALL_S=${MIN_S}
    ((WALL_S > MAX_S)) && WALL_S=${MAX_S}
    WALLTIME="$(fmt_hms ${WALL_S})"

    echo "steps     : ${TOTAL_STEPS}"
    echo "estimate  : ${EST_S}s at ${STEPS_PER_SEC} steps/s, x${SAFETY} + ${OVERHEAD_S}s overhead"
fi

echo "config    : ${CONFIG}"
echo "overrides : ${OVERRIDES:-(none)}"
echo "walltime  : ${WALLTIME}"

sbatch \
    --time="${WALLTIME}" \
    --export=ALL,TRAIN_CONFIG="${CONFIG}",TRAIN_OVERRIDES="${OVERRIDES}" \
    "${BATCH_DIR}/train.sh"
