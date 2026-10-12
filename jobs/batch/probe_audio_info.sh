#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=probe_audio_info
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
# 120G with one GPU and 18 cores keeps the job billed as one GPU (jobs/common/budget.sh).
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_probe_audio_info_%j.out

# How much of the transcript the audio alone carries, against mu and the field, at fixed
# flow times (scripts/diagnostics/probe_audio_info.py). Forward passes only, ~10-15 min,
# so at most ~64 SBU at the 30 min limit.
#
#   sbatch --export=ALL,COMPONENTS=runs/5v9lar7i/components jobs/batch/probe_audio_info.sh
#
# Optional env vars:
#   PROBE_N_UTTS     — utterances per role               (default 4096)
#   PROBE_ROLES      — roles to probe                    (default "train val")
#   PROBE_VARIANT    — best | final                      (default best)
#   PROBE_BATCH      — utterances per forward pass       (default 64; 16 on a gpu_mig
#                      slice, whose ~20 GB does not hold 64 of full latents in fp32)
#   PROBE_OUT        — output dir   (default figures/audio_info/<run id>)
#   PROBE_OVERRIDES  — dotlist overrides on top of the run's own config

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

if [[ -z "${COMPONENTS}" ]]; then
    echo "ERROR: set COMPONENTS to a runs/<id>/components directory" >&2
    exit 1
fi

RUN_ID="$(basename "$(dirname "${COMPONENTS}")")"
PROBE_N_UTTS="${PROBE_N_UTTS:-4096}"
PROBE_ROLES="${PROBE_ROLES:-train val}"
PROBE_VARIANT="${PROBE_VARIANT:-best}"
PROBE_BATCH="${PROBE_BATCH:-64}"
PROBE_OUT="${PROBE_OUT:-figures/audio_info/${RUN_ID}}"
PROBE_OVERRIDES="${PROBE_OVERRIDES:-}"

echo "components: ${COMPONENTS}   variant ${PROBE_VARIANT}"
echo "roles     : ${PROBE_ROLES}   ${PROBE_N_UTTS} utterances each, batch ${PROBE_BATCH}"
echo "out       : ${PROBE_OUT}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# data.env=snellius because the features are on scratch, not in the repo's data/.
srun python -m scripts.diagnostics.probe_audio_info \
    --components "${COMPONENTS}" \
    --variant "${PROBE_VARIANT}" \
    --roles ${PROBE_ROLES} \
    --n-utts "${PROBE_N_UTTS}" \
    --batch-size "${PROBE_BATCH}" \
    --out "${PROBE_OUT}" \
    data.env=snellius ${PROBE_OVERRIDES}
