#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=build_eval_cache
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_buildcache_%j.out

# Converts the downloaded parquet into the Arrow cache that load_dataset uses, ONCE.
# Submitted by submit_eval_sets.sh with --dependency=afterok, after the download array.
#
# Single task by design. Eight eval tasks calling load_dataset against a cold Arrow cache
# race on the same <hash>.incomplete/ directory and corrupt each other — the failures look
# like "Tried reading schema message, was null or length 0", "Cannot find data file", or a
# truncated message body. datasets guards this with file locks, which are not reliable on
# GPFS. Building it here makes every array task a pure cache hit.
#
# Required env vars (set via --export):
#   DATASETS — space-separated dataset keys

set -eo pipefail

source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/_env.sh"

python -m scripts.download_eval_sets \
    --env snellius \
    --build-cache \
    --num-shards 1 --shard-id 0 \
    --datasets ${DATASETS}
