#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=vocab_waste
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# Measure how much of a full-vocabulary model's output lands on tokens the corpus can
# never produce. Forward passes only — no training, no sampling — so it is minutes.
#
#   sbatch --export=ALL,COMPONENTS=runs/<id>/components jobs/batch/vocab_waste.sh
#   sbatch --export=ALL,CKPT=runs/.../last.ckpt        jobs/batch/vocab_waste.sh
#
# Optional env vars:
#   EVAL_CONFIG   — YAML for the data pipeline (default configs/audio_cond.yaml)
#   EVAL_ROLE     — train | val | test                  (default val)
#   EVAL_TIMES    — flow times to probe                 (default "0.3 0.5 0.7 0.9")
#   EVAL_BATCHES  — batches to average over             (default 8)
#   EVAL_VARIANT  — best | final                        (default best)
#   EVAL_OVERRIDES— dotlist overrides. data.compact_vocab comes from the run's manifest,
#                   so it does not need setting here and should not be: forcing it would
#                   fight the only record of what the weights were trained over.

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

if [[ -z "${COMPONENTS}" && -z "${CKPT}" ]]; then
    echo "ERROR: set COMPONENTS to a runs/<id>/components directory, or CKPT to a .ckpt" >&2
    exit 1
fi
if [[ -n "${COMPONENTS}" && -n "${CKPT}" ]]; then
    echo "ERROR: set COMPONENTS or CKPT, not both" >&2
    exit 1
fi

EVAL_CONFIG="${EVAL_CONFIG:-configs/audio_cond.yaml}"
EVAL_ROLE="${EVAL_ROLE:-val}"
EVAL_TIMES="${EVAL_TIMES:-0.3 0.5 0.7 0.9}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
EVAL_VARIANT="${EVAL_VARIANT:-best}"
EVAL_OVERRIDES="${EVAL_OVERRIDES:-}"

if [[ -n "${COMPONENTS}" ]]; then
    SOURCE_ARGS=(--components "${COMPONENTS}" --config "${EVAL_CONFIG}"
                 --variant "${EVAL_VARIANT}")
    echo "components: ${COMPONENTS}   config ${EVAL_CONFIG}   variant ${EVAL_VARIANT}"
else
    SOURCE_ARGS=(--ckpt "${CKPT}")
    echo "checkpoint: ${CKPT}"
fi
echo "role      : ${EVAL_ROLE}   times ${EVAL_TIMES}   batches ${EVAL_BATCHES}"
echo "overrides : ${EVAL_OVERRIDES}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# data.env=snellius because the features are on scratch, not in the repo's data/.
srun python -m scripts.diagnostics.vocab_waste \
    "${SOURCE_ARGS[@]}" \
    --role "${EVAL_ROLE}" \
    --batches "${EVAL_BATCHES}" \
    --times ${EVAL_TIMES} \
    data.env=snellius ${EVAL_OVERRIDES}
