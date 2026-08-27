#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=whisper_feats
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Worst case: one task over all 14 shards of train.clean.100 at the locally measured
# 8.9 utts/s. An A100 with 16 decode workers is much faster; submit_whisper.sh
# overrides this with a --time derived from the actual selection and task count.
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task encodes a disjoint slice of the input parquet shards.
# Submit via submit_whisper.sh — do not run this script directly.
#
# Required env vars (set by submit_whisper.sh via --export):
#   CONFIGS — space-separated subset of "clean other"
#   SPLITS  — space-separated subset of the LibriSpeech split names
#   MODEL   — Whisper checkpoint
#
# Shard assignment: shards[TASK_ID::TASK_COUNT], one output .npy/.json pair per input
# shard, so tasks never write the same file. Max useful task count is the number of
# shards selected (14 for train.100). Completed shards are skipped on resume.
#
# Throughput is GPU-bound: an isolated encoder benchmark on an RTX 3060 hit 8.1 utts/s
# at batch 16, matching the end-to-end rate once model load is subtracted. The decode
# workers keep up, so --batch-size matters more than --cpus-per-task here.
#
# Memory is not the constraint. Measured: 1.19 GiB of fp16 encoder weights plus ~45 MiB
# of activations per utterance (flat across batch sizes, since no_grad retains no graph).
# Batch 128 is ~6.8 GiB on a 40 GB card — the ceiling is diminishing returns, not OOM.

# No -u: conda's activate script reads unset variables and would abort under it.
set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/continuous-flow-asr"

# ── environment ──────────────────────────────────────────────────────────────
source activate "${CONDA_ENV:-flow_asr}"

# HF_TOKEN, for authenticated Hub requests (higher rate limits when pulling the
# Whisper weights). set -a exports everything the file defines.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "WARNING: no .env found; Hub requests will be unauthenticated" >&2
fi

# ── run ──────────────────────────────────────────────────────────────────────
python -m scripts.precompute_whisper \
    --env snellius \
    --model "${MODEL}" \
    --configs ${CONFIGS} \
    --splits ${SPLITS} \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --batch-size "${BATCH_SIZE:-128}"
