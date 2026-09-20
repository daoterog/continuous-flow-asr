#!/bin/bash

# Submit the ASR eval array job.
#
# Usage:
#   ./jobs/submit/eval.sh                                   # all 7 sets from audio, 8 tasks
#   DATASETS="librispeech_test.clean" ./jobs/submit/eval.sh # one set
#   N_JOBS=16 ./jobs/submit/eval.sh                         # more parallelism
#   SOURCE=latents CONFIGS="clean other" SPLITS=test ./jobs/submit/eval.sh
#   WALLTIME=01:00:00 ./jobs/submit/eval.sh                 # skip the estimate
#
# Afterwards, to merge the per-task .partN files and print WER against the leaderboard
# targets. The exact command is echoed at the end of this script with this run's MODEL,
# RESULTS_DIR and N_JOBS already filled in — the bare `python -m scripts.score_eval` only
# scores the right thing when all three were left at their defaults.
# Cheap enough for the login node (CPU only, seconds); jobs/batch/score_eval.sh is the
# same step as a job, for chaining with --dependency=afterok.
#
# Prerequisites:
#   audio   — ./jobs/submit/eval_sets.sh   (the configs must be fully cached, because
#             data_utils.load_data resolves a whole config)
#   latents — CONFIGS="clean other" SPLITS=test NO_TRIM=1 ./jobs/submit/whisper.sh

set -eo pipefail

SOURCE="${SOURCE:-audio}"
# audio source: leaderboard eval keys. librispeech appears here because the audio path
# reads it from hf-audio, unlike the latent path which reads the openslr feature cache.
DATASETS="${DATASETS:-ami_cleaned gigaspeech_cleaned spgispeech voxpopuli_cleaned_aa librispeech_test.clean librispeech_test.other earnings22_cleaned_aa_chunked}"
# latents source only (the LibriSpeech regression control); empty on the audio path.
CONFIGS="${CONFIGS:-}"
SPLITS="${SPLITS:-}"
N_JOBS="${N_JOBS:-8}"

# 64 matches the leaderboard's run_whisper.sh. RTFx varies with batch size, so holding it
# fixed is part of the measurement rather than a free knob.
BATCH_SIZE="${BATCH_SIZE:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
# LIMIT caps utterances per split before sharding — used by submit/rehearse.sh to run the
# identical array shape on a handful of utterances.
LIMIT="${LIMIT:-}"

# Wall-time model. The audio path runs the encoder AND an autoregressive decode, so it is
# slower per utterance than either alone: an RTX 3060 managed ~8 utts/s encoding and
# ~0.7 utts/s decoding. 4 utts/s assumes an A100 is several times the combined 3060 rate,
# which is deliberately pessimistic. Over-booking is free; under-booking loses the work,
# so the estimate is doubled again.
UTTS_PER_SEC="${UTTS_PER_SEC:-4}"
SAFETY="${SAFETY:-2}"
OVERHEAD_S="${OVERHEAD_S:-900}"   # module load, conda activate, ~3 GB of Whisper weights
MIN_S=1800                        # 30 min
MAX_S=86400                       # 24 h

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
BATCH_DIR="${REPO_DIR}/jobs/batch"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results}"

# Resolves MODEL and checks it here, on the login node, rather than letting the array
# queue for an hour and then fail — or worse, succeed against the wrong checkpoint.
# shellcheck disable=SC1091
source "${REPO_DIR}/jobs/pinned_model.sh"
require_pinned_model

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

if [[ -n "${LIMIT}" && -z "${WALLTIME}" ]]; then
    WALLTIME="00:30:00"
fi
if [[ -z "${WALLTIME}" ]]; then
    # run_eval --dry-run reports the utterance count from the dataset card, so this needs
    # neither the audio downloaded nor a model loaded.
    # `|| true` is load-bearing: under `set -eo pipefail` a failing python would abort the
    # script instead of reaching the fallback.
    UTTS="$(cd "${REPO_DIR}" && python -m scripts.run_eval \
        --dry-run --env snellius --source "${SOURCE}" --model "${MODEL}" \
        ${DATASETS:+--datasets ${DATASETS}} \
        ${CONFIGS:+--configs ${CONFIGS}} \
        ${SPLITS:+--splits ${SPLITS}} 2>/dev/null \
        | awk '/^utterances/ {print $2}' || true)"

    if [[ -z "${UTTS}" ]]; then
        WALLTIME="$(fmt_hms ${MAX_S})"
        echo "WARNING: could not size the job (is the env active? are the data cached?)," \
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
echo "source    : ${SOURCE}"
[[ "${SOURCE}" == "audio" ]] && echo "datasets  : ${DATASETS}"
[[ "${SOURCE}" == "latents" ]] && echo "configs   : ${CONFIGS} / splits: ${SPLITS}"
echo "tasks     : ${N_JOBS}"
echo "batch     : ${BATCH_SIZE} (warmup ${WARMUP_STEPS})"
echo "results   : ${RESULTS_DIR}"
[[ -n "${LIMIT}" ]] && echo "limit     : ${LIMIT} utt(s) per split (REHEARSAL, not a full run)"
echo "walltime  : ${WALLTIME}"

# --time here overrides the #SBATCH directive in run_eval.sh.
sbatch \
    --array="0-$((N_JOBS - 1))" \
    --time="${WALLTIME}" \
    --export=ALL,SOURCE="${SOURCE}",DATASETS="${DATASETS}",CONFIGS="${CONFIGS}",SPLITS="${SPLITS}",MODEL="${MODEL}",BATCH_SIZE="${BATCH_SIZE}",WARMUP_STEPS="${WARMUP_STEPS}",RESULTS_DIR="${RESULTS_DIR}",LIMIT="${LIMIT}" \
    "${BATCH_DIR}/run_eval.sh"

# The scorer filters manifests by model and refuses to merge an incomplete array, so it
# needs the same three values this run used. Printed rather than chained: scoring a
# rehearsal or a partial run is usually a deliberate choice.
echo
echo "When the array finishes, score it with:"
echo "  python -m scripts.score_eval \\"
echo "      --results-dir ${RESULTS_DIR} \\"
echo "      --model ${MODEL} \\"
echo "      --num-shards ${N_JOBS}"
