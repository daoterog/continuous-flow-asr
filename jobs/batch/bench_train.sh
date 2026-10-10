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

# What is the A100 actually limited by on the full-data recipe, and can two runs share it?
# Submit through jobs/submit/bench_train.sh, which prices and gates it (~130 SBU).
#
# Part 1: scripts/diagnostics/bench_train.py --suite prod — the runs as they were (train
# WER on), then the snellius profile, + compile, + bf16, + fused. Read ms/step and GPU %.
#
# Part 2: PACK_TRIAL twice at once on the same GPU, as jobs/batch/train.sh packs a job.
# Each copy's utts/s against part 1's single-run number is the packing slowdown:
#   PACK_SLOWDOWN = single utts/s / packed utts/s per run  -> jobs/common/budget.sh
# Packing pays while that stays below 2.

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

STEPS="${STEPS:-200}"
N_UTTS="${N_UTTS:-8192}"
PACK_TRIAL="${PACK_TRIAL:-prod + compile}"

echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== part 1: one run on the GPU"
python -m scripts.diagnostics.bench_train --suite prod --env snellius \
    --steps "${STEPS}" --n-utts "${N_UTTS}"

echo
echo "=== part 2: '${PACK_TRIAL}' twice at once, 8 workers each"
for i in 0 1; do
    python -m scripts.diagnostics.bench_train --suite prod --env snellius --no-warm \
        --steps "${STEPS}" --n-utts "${N_UTTS}" --only "${PACK_TRIAL}" \
        --set data.num_workers=8 > "bench_pack_${SLURM_JOB_ID:-local}_${i}.out" 2>&1 &
done
wait
for i in 0 1; do
    echo "-- copy ${i}"
    grep -E "^(trial|${PACK_TRIAL})" "bench_pack_${SLURM_JOB_ID:-local}_${i}.out" || \
        tail -5 "bench_pack_${SLURM_JOB_ID:-local}_${i}.out"
done
