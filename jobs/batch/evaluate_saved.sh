#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=evaluate_saved
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
# 120G with one GPU and 18 cores keeps the job billed as one GPU (jobs/common/budget.sh).
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_evaluate_saved_%j.out

# train.py's end-of-training evaluation, on a run that has already finished
# (scripts/evaluation/evaluate_saved.py): `_best` and `_final` sampled on val and test. A few
# minutes of sampling, at most ~64 SBU at the 30 min limit.
#
#   ./jobs/sbatch.sh --export=ALL,COMPONENTS=runs/<id>/components jobs/batch/evaluate_saved.sh
#
# The reloaded `_final` on val must equal the sampled WER the run logged at its last
# sampling epoch; that is the check this exists for.
#
# Optional env vars:
#   EVAL_ROLES      — splits to score               (default "val test")
#   EVAL_OVERRIDES  — dotlist overrides on top of the run's own config

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

if [[ -z "${COMPONENTS}" ]]; then
    echo "ERROR: set COMPONENTS to a runs/<id>/components directory" >&2
    exit 1
fi

echo "components: ${COMPONENTS}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# data.env=snellius because the features are on scratch, not in the repo's data/.
srun python -m scripts.evaluation.evaluate_saved --components "${COMPONENTS}" \
    --roles ${EVAL_ROLES:-val test} data.env=snellius ${EVAL_OVERRIDES:-}
