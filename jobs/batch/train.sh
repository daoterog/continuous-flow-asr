#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=flow_train
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
# Sized so the features fit the page cache, not so the process fits. Full clean/train.100
# is 102 GiB of fp16 latents; at 64G it did not cache and every epoch re-read it off GPFS
# at ~219 utts/s against 657 utts/s warm — a 3x walltime penalty for a request that was
# already below a 1-GPU fair share of the node (4 GPUs, ~500G).
#SBATCH --mem=120G
#SBATCH --time=01:00:00
# A placeholder only. submit/train.sh overrides --time with a value derived from the
# config's own step count and a measured steps-per-second, so this is what you get if
# you sbatch this file directly — which you should not.
#SBATCH --output=slurm_output_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# One training run. Submit via submit/train.sh — do not run this script directly.
#
# Required env vars (set by submit/train.sh via --export):
#   TRAIN_CONFIG    — configs/<name>.yaml
#   TRAIN_OVERRIDES — space-separated OmegaConf dotlist overrides, may be empty

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

# wandb runs on the compute node, which has no terminal to log in from. It reads
# WANDB_API_KEY from the environment (_env.sh sources .env) or ~/.netrc, which is shared
# with the login node. Without either it would block on a prompt and burn the whole
# reservation, so fail here instead.
if [[ -z "${WANDB_API_KEY}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: no wandb credentials. Put WANDB_API_KEY in .env, or run 'wandb login'" >&2
    echo "       on the login node so ~/.netrc is written." >&2
    exit 1
fi

# Features live on scratch, not in the repo's data/ — the DataModule resolves this from
# data.env, so make the choice explicit rather than relying on the config's default.
OVERRIDES="data.env=snellius ${TRAIN_OVERRIDES}"

echo "config    : ${TRAIN_CONFIG}"
echo "overrides : ${OVERRIDES}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python train.py --config "${TRAIN_CONFIG}" ${OVERRIDES}
