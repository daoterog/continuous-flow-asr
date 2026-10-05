#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=bench_probe
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
# No dataset: the probe is timed against synthetic tensors of the right shape, so none of
# the 160G the training job needs for the feature cache applies here.
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=slurm_output_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# How much does the measured-tau probe cost on an A100, and does the chunk matter once the
# probe is large?
#
#   sbatch jobs/batch/bench_probe.sh
#   PROBE_N="2048 32768" CHUNKS="32 1024" sbatch jobs/batch/bench_probe.sh
#
# jobs/sweeps/probe_chunk.sh already asked this through five full training runs and could
# not answer it: a 2048-utterance probe is ~2 s against an 81 s epoch, and the spread
# between nodes was 2.3%. This times the probe alone, so the answer is not buried in
# training. It needs no features and no wandb — read the stdout.

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

PROBE_N="${PROBE_N:-2048 4096 8192 16384 32768}"
# 2048 is past anything training would use; included to find where it stops paying, or
# where it runs out of card.
CHUNKS="${CHUNKS:-32 128 256 512 1024 2048}"
# One epoch of full train.100 at batch 128, measured across the probe_chunk runs.
EPOCH_SECONDS="${EPOCH_SECONDS:-81}"

echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python -m scripts.diagnostics.bench_probe \
    --probe-n ${PROBE_N} \
    --chunks ${CHUNKS} \
    --epoch-seconds "${EPOCH_SECONDS}"
