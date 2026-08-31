#!/bin/bash

# Submit the ASR eval array job: decode cached Whisper latents into transcripts.
#
# Usage:
#   ./jobs/submit_eval.sh                              # clean+other test, 8 tasks
#   N_JOBS=16 ./jobs/submit_eval.sh                    # more parallelism
#   CONFIGS=clean ./jobs/submit_eval.sh                # test.clean only
#   WALLTIME=01:00:00 ./jobs/submit_eval.sh            # skip the estimate
#
# Afterwards, on the login node (CPU only, seconds):
#   python -m scripts.score_eval
# which merges the per-task .partN files and prints WER against the leaderboard targets.
#
# Prerequisite: the latents must exist and be UNTRIMMED:
#   CONFIGS="clean other" SPLITS=test NO_TRIM=1 ./jobs/submit_whisper.sh

set -eo pipefail

CONFIGS="${CONFIGS:-clean other}"
SPLITS="${SPLITS:-test}"
MODEL="${MODEL:-openai/whisper-large-v3}"
N_JOBS="${N_JOBS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# LIMIT caps utterances per split before sharding — used by rehearse_eval.sh to run
# the identical array shape on a handful of utterances.
LIMIT="${LIMIT:-}"

# Wall-time model. Decoding is autoregressive, so it is much slower per utterance than
# the encode: an RTX 3060 managed ~0.7 utts/s at batch 16. 5 utts/s assumes an A100 is
# ~7x that, which is deliberately pessimistic. Over-booking is free; under-booking loses
# the work, so the estimate is doubled again.
UTTS_PER_SEC="${UTTS_PER_SEC:-5}"
SAFETY="${SAFETY:-2}"
OVERHEAD_S="${OVERHEAD_S:-900}"   # module load, conda activate, ~3 GB of Whisper weights
MIN_S=1800                        # 30 min
MAX_S=86400                       # 24 h

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results}"

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

if [[ -n "${LIMIT}" && -z "${WALLTIME}" ]]; then
    WALLTIME="00:30:00"
fi
if [[ -z "${WALLTIME}" ]]; then
    # Reuse the precompute's dry-run to count utterances without loading a model.
    # `|| true` is load-bearing: under `set -eo pipefail` a failing python would abort
    # the script instead of reaching the fallback.
    UTTS="$(cd "${REPO_DIR}" && python -m scripts.precompute_whisper \
        --dry-run --env snellius --model "${MODEL}" \
        --configs ${CONFIGS} --splits ${SPLITS} \
        --num-shards 1 --shard-id 0 2>/dev/null \
        | awk '/^utterances/ {print $2}' || true)"

    if [[ -z "${UTTS}" ]]; then
        WALLTIME="$(fmt_hms ${MAX_S})"
        echo "WARNING: could not size the job (is the env active? are the latents there?)," \
             "falling back to ${WALLTIME}" >&2
    else
        PER_TASK=$(( (UTTS + N_JOBS - 1) / N_JOBS ))
        EST_S=$(awk -v u="${PER_TASK}" -v r="${UTTS_PER_SEC}" 'BEGIN{printf "%d", u/r}')
        WALL_S=$((OVERHEAD_S + EST_S * SAFETY))
        ((WALL_S < MIN_S)) && WALL_S=${MIN_S}
        ((WALL_S > MAX_S)) && WALL_S=${MAX_S}
        WALLTIME="$(fmt_hms ${WALL_S})"

        echo "utterances: ${UTTS} total, ${PER_TASK} per task"
        echo "estimate  : ${EST_S}s at ${UTTS_PER_SEC} utts/s, x${SAFETY} + ${OVERHEAD_S}s overhead"
    fi
fi

echo "model     : ${MODEL}"
echo "configs   : ${CONFIGS}"
echo "splits    : ${SPLITS}"
echo "tasks     : ${N_JOBS}"
echo "results   : ${RESULTS_DIR}"
[[ -n "${LIMIT}" ]] && echo "limit     : ${LIMIT} utt(s) per split (REHEARSAL, not a full run)"
echo "walltime  : ${WALLTIME}"

# --time here overrides the #SBATCH directive in run_eval.sh.
sbatch \
    --array="0-$((N_JOBS - 1))" \
    --time="${WALLTIME}" \
    --export=ALL,CONFIGS="${CONFIGS}",SPLITS="${SPLITS}",MODEL="${MODEL}",BATCH_SIZE="${BATCH_SIZE}",RESULTS_DIR="${RESULTS_DIR}",LIMIT="${LIMIT}" \
    "${SCRIPT_DIR}/run_eval.sh"
