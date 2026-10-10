#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=bench_train
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
# Same shape as a training job (jobs/batch/train.sh), so what this measures is what a run
# gets: the feature cache, 16 workers, and two runs side by side in the packing step.
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_bench_train_%j.out

# What is the A100 actually limited by on the full-data recipe? Submit through
# jobs/submit/bench_train.sh, which prices and gates it.
#
# scripts/diagnostics/bench_train.py --suite prod: the runs as they were (train WER on),
# then the snellius profile, + compile, + bf16, + fused. Read ms/step and GPU %.
#
# Two runs sharing one GPU is not measured here: a job is billed for 2 GPUs anyway (see
# jobs/common/budget.sh), so runs are paired one per GPU instead, which shares nothing but
# the page cache.

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

STEPS="${STEPS:-200}"
N_UTTS="${N_UTTS:-8192}"

echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -m scripts.diagnostics.bench_train --suite prod --env snellius \
    --steps "${STEPS}" --n-utts "${N_UTTS}"
