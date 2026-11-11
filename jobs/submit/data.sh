#!/bin/bash

# Get the training data onto scratch: download LibriSpeech if any of it is missing, then
# encode it through Whisper once the download has finished. Two jobs through the usual
# submitters, the second held with --dependency=afterok on the first, so it starts only
# if every download task succeeded. Training is submitted separately, as always.
#
# Usage:
#   ./jobs/submit/data.sh                              # everything
#   SPLITS="validation test" ./jobs/submit/data.sh     # a subset
#
# The download is skipped when `download_librispeech --verify` finds every file already
# there. The encode always goes in: it skips the shards it has already finished, so on
# complete data it ends in minutes.
#
# NO_TRIM=1 always: the loader refuses trimmed latents.
#
# Env vars:
#   CONFIGS       (default "clean other")
#   SPLITS        (default "train.100 train.360 train.500 validation test")
#   N_DL          download tasks (default 8)
#   N_ENC         encode tasks, one A100 each (default 16)
#   UTTS_PER_SEC  per-task encode rate the walltime assumes when the data is not there yet
#                 to size it from (default 30, as in submit/whisper.sh)

set -eo pipefail

CONFIGS="${CONFIGS:-clean other}"
SPLITS="${SPLITS:-train.100 train.360 train.500 validation test}"
N_DL="${N_DL:-8}"
N_ENC="${N_ENC:-16}"
UTTS_PER_SEC="${UTTS_PER_SEC:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }
# The submitters print sbatch's "Submitted batch job <id>"; take the id off it.
job_id() { awk '/Submitted batch job/ {print $4}' | tail -1; }

DEPENDENCY=""
if (cd "${REPO_DIR}" && python -m scripts.data.download_librispeech --env snellius \
        --configs ${CONFIGS} --splits ${SPLITS} --verify); then
    echo "download  : not needed"
else
    dl="$(CONFIGS="${CONFIGS}" SPLITS="${SPLITS}" N_JOBS="${N_DL}" \
        "${REPO_DIR}/jobs/submit/librispeech.sh" | tee /dev/stderr | job_id)"
    [[ -n "${dl}" ]] || { echo "ERROR: the download was not submitted" >&2; exit 1; }
    echo "download  : job ${dl}"
    DEPENDENCY="afterok:${dl}"

    # submit/whisper.sh sizes its walltime off the downloaded data, which does not exist
    # yet. Size it here instead, from LibriSpeech's own utterance counts: the worst task's
    # share padded 30% for uneven shards, doubled, plus model load, as whisper.sh does.
    declare -A UTTS=(
        [clean/train.100]=28539 [clean/train.360]=104014 [other/train.500]=148688
        [clean/validation]=2703 [other/validation]=2864 [clean/test]=2620 [other/test]=2939
    )
    total=0
    for c in ${CONFIGS}; do
        for s in ${SPLITS}; do
            total=$((total + ${UTTS[${c}/${s}]:-0}))
        done
    done
    per_task=$(((total + N_ENC - 1) / N_ENC * 13 / 10))
    wall_s=$((900 + per_task / UTTS_PER_SEC * 2))
    ((wall_s < 1800)) && wall_s=1800
    ((wall_s > 86400)) && wall_s=86400
    export WALLTIME="$(fmt_hms "${wall_s}")"
    echo "encode    : ~${per_task} utterances per task -> walltime ${WALLTIME}"
fi

enc="$(CONFIGS="${CONFIGS}" SPLITS="${SPLITS}" N_JOBS="${N_ENC}" NO_TRIM=1 \
    DEPENDENCY="${DEPENDENCY}" "${REPO_DIR}/jobs/submit/whisper.sh" | tee /dev/stderr | job_id)"
[[ -n "${enc}" ]] || { echo "ERROR: the encode was not submitted" >&2; exit 1; }
echo "encode    : job ${enc}${DEPENDENCY:+, waiting on ${DEPENDENCY#afterok:}}"
