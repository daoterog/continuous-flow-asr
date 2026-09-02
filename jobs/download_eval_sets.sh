#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=dl_eval_sets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# Worst case: one task pulling all ~18 GB. submit_eval_sets.sh overrides this with a
# --time derived from the actual selection and task count.
#SBATCH --output=slurm_dl_eval_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Array job: each task downloads a disjoint slice of the eval datasets.
# Submit via submit_eval_sets.sh — do not run this script directly.
#
# Required env vars (set by submit_eval_sets.sh via --export):
#   DATASETS — space-separated subset of the leaderboard eval configs
#
# File assignment: files[TASK_ID::TASK_COUNT] over every selected repo, so tasks never
# write the same blob. hf_hub locks per file, making concurrent writes into one cache safe.
#
# staging is the partition meant for data transfer — cheap, long wall times, and keeps a
# pure-I/O job off the GPU allocation.

# No -u: conda's activate script reads unset variables and would abort under it.
set -eo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/continuous-flow-asr"

# ── environment ──────────────────────────────────────────────────────────────
source activate "${CONDA_ENV:-flow_asr}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "WARNING: no .env found; Hub requests will be unauthenticated" >&2
fi

# ── run ──────────────────────────────────────────────────────────────────────
python -m scripts.download_eval_sets \
    --env snellius \
    --datasets ${DATASETS} \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --workers "${SLURM_CPUS_PER_TASK}"
