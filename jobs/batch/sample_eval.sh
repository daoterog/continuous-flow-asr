#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=step_sweep
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_output_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Score a saved run by sampling, over a range of integration step counts. No training, so
# this is minutes rather than hours — the point is that `sampling.num_steps` was inherited
# from the toy and has never been swept on real data.
#
#   sbatch --export=ALL,COMPONENTS=runs/<id>/components jobs/batch/sample_eval.sh
#
# Optional env vars:
#   EVAL_CONFIG   — YAML for the data pipeline (default configs/audio_cond.yaml). Only the
#                   data side is used: geometry and the flow come from the run's manifest.
#   EVAL_ROLE     — train | val | test        (default val)
#   EVAL_STEPS    — step counts to try        (default "1 2 4 8 16 32 64 128")
#   EVAL_VARIANT  — best | final              (default best)
#   EVAL_BATCH    — utterances per chunk      (default 128)

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

: "${COMPONENTS:?set COMPONENTS to a runs/<id>/components directory}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/audio_cond.yaml}"
EVAL_ROLE="${EVAL_ROLE:-val}"
EVAL_STEPS="${EVAL_STEPS:-1 2 4 8 16 32 64 128}"
EVAL_VARIANT="${EVAL_VARIANT:-best}"
EVAL_BATCH="${EVAL_BATCH:-128}"

echo "components: ${COMPONENTS}"
echo "config    : ${EVAL_CONFIG}   role ${EVAL_ROLE}   variant ${EVAL_VARIANT}"
echo "steps     : ${EVAL_STEPS}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# data.env=snellius because the features are on scratch, not in the repo's data/.
srun python -m scripts.sample_eval \
    --components "${COMPONENTS}" \
    --config "${EVAL_CONFIG}" \
    --role "${EVAL_ROLE}" \
    --variant "${EVAL_VARIANT}" \
    --batch-size "${EVAL_BATCH}" \
    --steps ${EVAL_STEPS} \
    data.env=snellius
