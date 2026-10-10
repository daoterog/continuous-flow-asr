#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=bench_train
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
# Same shape as a training job (jobs/batch/train.sh), so what this measures is what a run
# gets: the feature cache, 16 workers, and two runs side by side in the packing step.
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_bench_train_%j.out

# What fills the A100, and what does an evaluation cost? Submit through
# jobs/submit/bench_train.sh, which prices and gates it and asks for 2 GPUs — what a job
# needing 160G is billed for anyway — so the two halves below run side by side:
#
#   GPU 0  bench_train --suite gpu: the batch ladder 128 / 256 / 512 / 1024 under the
#          snellius profile (compile + bf16). Read utts/s and GPU %: the knee is the
#          smallest batch within ~5% of the best.
#   GPU 1  the compute knobs at AT_BATCH (matmul high, max-autotune), then --eval-timing:
#          one full validation pass and one sampling probe, in seconds, which is what the
#          eval cadence in configs/profiles/run/ is chosen from.
#
# SUITE=prod runs the older single-GPU comparison instead (as-run vs profile vs compile).

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

STEPS="${STEPS:-200}"
N_UTTS="${N_UTTS:-8192}"
AT_BATCH="${AT_BATCH:-512}"

echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [[ "${SUITE:-gpu}" == "prod" ]]; then
    python -m scripts.diagnostics.bench_train --suite prod --env snellius \
        --steps "${STEPS}" --n-utts "${N_UTTS}"
    exit $?
fi

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
OUT0="bench_gpu_${SLURM_JOB_ID:-local}_0.out"
OUT1="bench_gpu_${SLURM_JOB_ID:-local}_1.out"

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -m scripts.diagnostics.bench_train \
    --suite gpu --env snellius --steps "${STEPS}" --n-utts "${N_UTTS}" \
    --only "batch 128" "batch 256" "batch 512" "batch 1024" > "${OUT0}" 2>&1 &
PID0=$!
(
    CUDA_VISIBLE_DEVICES="${GPU_IDS[1]:-${GPU_IDS[0]}}" python -m scripts.diagnostics.bench_train \
        --suite gpu --env snellius --steps "${STEPS}" --n-utts "${N_UTTS}" \
        --at-batch "${AT_BATCH}" \
        --only "batch ${AT_BATCH} + matmul high" "batch ${AT_BATCH} + max-autotune"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[1]:-${GPU_IDS[0]}}" python -m scripts.diagnostics.bench_train \
        --suite gpu --env snellius --eval-timing --at-batch "${AT_BATCH}"
) > "${OUT1}" 2>&1 &
PID1=$!

failed=0
wait "${PID0}" || failed=1
wait "${PID1}" || failed=1

# The tables only; the full logs stay in the two files.
for f in "${OUT0}" "${OUT1}"; do
    echo
    echo "=== ${f}"
    grep -E '^(device|trial|batch [0-9]|fastest|eval at|  validation|  sampling)' "${f}" \
        || tail -20 "${f}"
done
exit "${failed}"
