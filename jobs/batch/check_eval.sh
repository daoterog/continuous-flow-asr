#!/bin/bash

#SBATCH --job-name=eval_check
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

# Verifies that a sharded eval array produced complete, non-overlapping results.
# Submitted by submit/rehearse.sh with --dependency=afterok, so it runs only if every array
# task succeeded. CPU only; no GPU, no partition pinned here (the submitter chooses).
#
# Required env vars (set via --export):
#   RESULTS_DIR  — directory holding the .partN.jsonl files
#   NUM_SHARDS   — array task count the run used
#   EXPECT_TOTAL — utterances expected per split (the LIMIT used, or the split size)
#   DATASETS     — space-separated dataset keys, for the manifest-count check

set -eo pipefail

source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

python -m scripts.evaluation.check_eval \
    --results-dir "${RESULTS_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --expect-total "${EXPECT_TOTAL}" \
    --datasets ${DATASETS}
