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
#   sbatch --export=ALL,CKPT=runs/.../last.ckpt        jobs/batch/sample_eval.sh
#
# Runs up to `skilled-lake-22` wrote a Lightning .ckpt and no components; later ones the
# reverse. Both load: the checkpoint carries its own config, so an older run is rescored
# under the schedule it was actually trained with.
#
# Optional env vars:
#   EVAL_CONFIG   — YAML for the data pipeline (default configs/audio_cond.yaml). Only the
#                   data side is used: geometry and the flow come from the run's manifest.
#   EVAL_ROLE     — train | val | test        (default val)
#   EVAL_STEPS    — step counts to try        (default "1 2 4 8 16 32 64 128")
#   EVAL_VARIANT  — best | final              (default best)
#   EVAL_BATCH    — utterances per chunk. Unset uses the run's own training batch, which
#                   sample_eval then caps to fit the integrator: a full-vocabulary field is
#                   43x a compact one, so 128 is routine at 1,193 tokens and OOMs at 51,866.
#   EVAL_HARD     — set to 1 to feed back a one-hot argmax between steps instead of the
#                   softmax. That is what training showed the model: the interpolant is
#                   built from F.one_hot(tokens), a vertex, never an interior point.
#   EVAL_OVERRIDES— extra dotlist overrides. The default forces the whole split: a run's
#                   own config may carry data.subset_n.val from when it trained, and
#                   scoring two runs on different amounts is what makes them incomparable.

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
EVAL_STEPS="${EVAL_STEPS:-1 2 4 8 16 32 64 128}"
EVAL_VARIANT="${EVAL_VARIANT:-best}"
EVAL_BATCH="${EVAL_BATCH:-}"
EVAL_OVERRIDES="${EVAL_OVERRIDES:-data.subset_n.val=null}"
# Tested by value, not by emptiness: ${EVAL_HARD:+--hard} expands for ANY non-empty
# string, so EVAL_HARD=0 would still pass the flag while the log said otherwise.
case "${EVAL_HARD:-0}" in
    1|true|TRUE|yes) HARD_ARGS=(--hard); EVAL_HARD=1 ;;
    *)               HARD_ARGS=();       EVAL_HARD=0 ;;
esac

if [[ -n "${COMPONENTS}" ]]; then
    # --config supplies the data pipeline only; geometry comes from the manifest.
    SOURCE_ARGS=(--components "${COMPONENTS}" --config "${EVAL_CONFIG}"
                 --variant "${EVAL_VARIANT}")
    echo "components: ${COMPONENTS}   config ${EVAL_CONFIG}   variant ${EVAL_VARIANT}"
else
    # A checkpoint carries its own config, so --config would only override it.
    SOURCE_ARGS=(--ckpt "${CKPT}")
    echo "checkpoint: ${CKPT}"
fi
echo "role      : ${EVAL_ROLE}   overrides ${EVAL_OVERRIDES}"
echo "steps     : ${EVAL_STEPS}   hard ${EVAL_HARD:-0}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# data.env=snellius because the features are on scratch, not in the repo's data/.
srun python -m scripts.sample_eval \
    "${SOURCE_ARGS[@]}" \
    --role "${EVAL_ROLE}" \
    ${EVAL_BATCH:+--batch-size "${EVAL_BATCH}"} \
    "${HARD_ARGS[@]}" \
    --steps ${EVAL_STEPS} \
    data.env=snellius ${EVAL_OVERRIDES}
