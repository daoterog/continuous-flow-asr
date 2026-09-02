#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=dl_librispeech
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# Worst case: one task, cold cache, the whole 61.8 GB corpus. At a pessimistic
# 10 MB/s that is ~1h43m; doubled plus overhead gives ~3h36m, rounded up here.
# submit/librispeech.sh overrides this with a --time derived from the actual
# selection and task count (8 tasks over the full corpus needs ~36 min).
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task downloads a disjoint slice of LibriSpeech parquet shards.
# Submit via submit/librispeech.sh — do not run this script directly.
#
# Required env vars (set by submit/librispeech.sh via --export):
#   CONFIGS — space-separated subset of "clean other"
#   SPLITS  — space-separated subset of "train.100 train.360 train.500 validation test"
#
# Shard assignment: the script lists every parquet file for the selected
# (config, split) pairs, sorts it, and takes files[TASK_ID::TASK_COUNT].  Shards are
# ~500 MB each, so interleaving keeps every task's slice the same size even though the
# splits themselves range from 0.35 GB to 31.8 GB.  Slices are disjoint, so all tasks
# can write into the one shared cache safely.  Failures go to a per-task log.
#
# NOTE: staging is the partition meant for data transfer — it is cheap, allows long
# wall times, and keeps a pure-I/O job off the GPU allocation.

# No -u: conda's activate script reads unset variables and would abort under it.
set -eo pipefail

source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

# ── run ──────────────────────────────────────────────────────────────────────
python -m scripts.download_librispeech \
    --env snellius \
    --configs ${CONFIGS} \
    --splits ${SPLITS} \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}"
