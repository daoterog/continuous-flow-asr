#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=asr_eval
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Worst case: one task over all 5559 eval utterances. submit_eval.sh overrides this with
# a --time derived from the actual selection and task count.
#SBATCH --output=slurm_eval_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task decodes a disjoint slice of the cached encoder latents.
# Submit via submit_eval.sh — do not run this script directly.
#
# Required env vars (set by submit_eval.sh via --export):
#   CONFIGS — space-separated subset of "clean other"
#   SPLITS  — space-separated split names as stored in the feature cache
#   MODEL   — Whisper checkpoint
#
# Each task writes MODEL_..._<dataset>_<split>.part<TASK_ID>.jsonl; score_eval merges the
# parts back into one manifest per split before scoring, because score_results derives the
# dataset name from the filename and would otherwise average the parts as separate sets.
#
# This decodes cached latents rather than audio: the encoder does not run here. The
# latents must have been precomputed with NO_TRIM=1 — Whisper's decoder cross-attends over
# all 1500 frames and produces repeated <|en|> tokens from a trimmed cache. run_eval.py
# refuses a trimmed cache rather than scoring an approximation.
#
# Decoding is autoregressive and therefore far slower per utterance than encoding was;
# more array tasks help here much more than a bigger batch.

# No -u: conda's activate script reads unset variables and would abort under it.
set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/continuous-flow-asr"

# ── environment ──────────────────────────────────────────────────────────────
source activate "${CONDA_ENV:-flow_asr}"

# Conda relies on RPATH and does not set LD_LIBRARY_PATH, but pip wheels such as
# torchcodec carry no RPATH into the env, so its libtorchcodec resolves libstdc++
# against Snellius's /lib64 — too old for the GLIBCXX conda's own libicuuc needs.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "WARNING: no .env found; Hub requests will be unauthenticated" >&2
fi

# The eval imports the leaderboard normalizer from the pinned submodule.
if [[ ! -f open_asr_leaderboard/normalizer/__init__.py ]]; then
    echo "ERROR: submodule empty; run 'git submodule update --init --recursive'" >&2
    exit 1
fi

# ── run ──────────────────────────────────────────────────────────────────────
python -m scripts.run_eval \
    --env snellius \
    --model "${MODEL}" \
    --configs ${CONFIGS} \
    --splits ${SPLITS} \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --results-dir "${RESULTS_DIR:-$HOME/continuous-flow-asr/results}"
