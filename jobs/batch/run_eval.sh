#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=asr_eval
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Worst case: one task over all 5559 eval utterances. submit/eval.sh overrides this with
# a --time derived from the actual selection and task count.
#SBATCH --output=slurm_eval_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task decodes a disjoint slice of the cached encoder latents.
# Submit via submit/eval.sh — do not run this script directly.
#
# Required env vars (set by submit/eval.sh via --export):
#   SOURCE   — "audio" (default, all datasets) or "latents" (LibriSpeech control)
#   DATASETS — audio source: space-separated leaderboard eval keys
#   CONFIGS  — latents source: space-separated subset of "clean other"
#   SPLITS   — latents source: split names as stored in the feature cache
#   MODEL    — Whisper checkpoint
#   LIMIT    — optional; utterances per split BEFORE sharding, for rehearsals
#
# Each task writes MODEL_..._<dataset>_<split>.part<TASK_ID>.jsonl; score_eval merges the
# parts back into one manifest per split before scoring, because score_results derives the
# dataset name from the filename and would otherwise average the parts as separate sets.
#
# SOURCE=audio runs the full encoder-decoder and is the path for every dataset; it is the
# only one that yields a comparable RTFx. SOURCE=latents decodes the cached encoder
# output instead, skipping the encoder — the LibriSpeech control that reproduced the
# published numbers. Latents must have been precomputed with NO_TRIM=1: Whisper's decoder
# cross-attends over all 1500 frames and emits repeated <|en|> from a trimmed cache, and
# run_eval.py
# refuses a trimmed cache rather than scoring an approximation.
#
# Decoding is autoregressive and therefore far slower per utterance than encoding was;
# more array tasks help here much more than a bigger batch.

# No -u: conda's activate script reads unset variables and would abort under it.
set -eo pipefail

source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

# ── run ──────────────────────────────────────────────────────────────────────
python -m scripts.run_eval \
    --env snellius \
    --model "${MODEL}" \
    --source "${SOURCE:-audio}" \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --batch-size "${BATCH_SIZE:-64}" \
    --warmup-steps "${WARMUP_STEPS:-10}" \
    --results-dir "${RESULTS_DIR:-$HOME/continuous-flow-asr/results}" \
    ${DATASETS:+--datasets ${DATASETS}} \
    ${CONFIGS:+--configs ${CONFIGS}} \
    ${SPLITS:+--splits ${SPLITS}} \
    ${LIMIT:+--limit "${LIMIT}"}
