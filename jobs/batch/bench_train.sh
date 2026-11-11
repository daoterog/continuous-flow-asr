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
#   GPU 0  bench_train --suite gpu: the batch ladder (BATCHES, default 32..512) for the
#          fulldata recipe (small + micro) under the snellius profile — full latents,
#          compile + bf16. Read utts/s and GPU %: the knee is the smallest batch within ~5%
#          of the best; an OOM row marks the ceiling.
#   GPU 1  `replay` at AT_BATCH/2 and AT_BATCH (one batch kept on the GPU, so no loader,
#          pinning or copy) against the ladder's real rows, then --eval-timing at AT_BATCH:
#          one full validation pass and one sampling probe, in seconds.
#
# SUITE=prod runs the older single-GPU comparison instead (as-run vs profile vs compile).

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

STEPS="${STEPS:-200}"
N_UTTS="${N_UTTS:-8192}"
AT_BATCH="${AT_BATCH:-128}"
BATCHES="${BATCHES:-32 64 128 256 512}"
read -ra LADDER <<< "${BATCHES}"
LADDER_TRIALS=()
for b in "${LADDER[@]}"; do LADDER_TRIALS+=("batch ${b}"); done

echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# SUITE=prefetch: does copying the next batch while the current one trains
# (data.gpu_prefetch) close the real-vs-replay gap? GPU 0 times batch 128 with and without
# it, GPU 1 the on-GPU ceiling at 128 and prefetch at 256 (whether the best batch moves).
# SUITE=tune: the last knobs, each against the production setting (batch 128 + prefetch).
if [[ "${SUITE:-gpu}" == "tune" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    OUT0="bench_gpu_${SLURM_JOB_ID:-local}_0.out"
    OUT1="bench_gpu_${SLURM_JOB_ID:-local}_1.out"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -m scripts.diagnostics.bench_train \
        --suite tune --env snellius --steps "${STEPS}" --n-utts "${N_UTTS}" \
        --only "tune: as is" "tune: matmul high" > "${OUT0}" 2>&1 &
    PID0=$!
    CUDA_VISIBLE_DEVICES="${GPU_IDS[1]:-${GPU_IDS[0]}}" python -m scripts.diagnostics.bench_train \
        --suite tune --env snellius --steps "${STEPS}" --n-utts "${N_UTTS}" \
        --only "tune: prefetch_factor 4" "tune: workers 17" > "${OUT1}" 2>&1 &
    PID1=$!
    failed=0
    wait "${PID0}" || failed=1
    wait "${PID1}" || failed=1
    for f in "${OUT0}" "${OUT1}"; do
        echo; echo "=== ${f}"
        grep -E '^(device|trial|tune:)' "${f}" || tail -20 "${f}"
    done
    exit "${failed}"
fi

if [[ "${SUITE:-gpu}" == "prefetch" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    OUT0="bench_gpu_${SLURM_JOB_ID:-local}_0.out"
    OUT1="bench_gpu_${SLURM_JOB_ID:-local}_1.out"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -m scripts.diagnostics.bench_train \
        --suite gpu --env snellius --steps "${STEPS}" --n-utts "${N_UTTS}" \
        --batches 128 256 --only "batch 128" "prefetch 128" > "${OUT0}" 2>&1 &
    PID0=$!
    CUDA_VISIBLE_DEVICES="${GPU_IDS[1]:-${GPU_IDS[0]}}" python -m scripts.diagnostics.bench_train \
        --suite gpu --env snellius --steps "${STEPS}" --n-utts "${N_UTTS}" --at-batch 128 \
        --batches 128 256 --only "replay 128" "prefetch 256" > "${OUT1}" 2>&1 &
    PID1=$!
    failed=0
    wait "${PID0}" || failed=1
    wait "${PID1}" || failed=1
    for f in "${OUT0}" "${OUT1}"; do
        echo; echo "=== ${f}"
        grep -E '^(device|trial|batch [0-9]|replay|prefetch|fastest)' "${f}" || tail -20 "${f}"
    done
    exit "${failed}"
fi

if [[ "${SUITE:-gpu}" == "prod" ]]; then
    python -m scripts.diagnostics.bench_train --suite prod --env snellius \
        --steps "${STEPS}" --n-utts "${N_UTTS}"
    exit $?
fi

IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
OUT0="bench_gpu_${SLURM_JOB_ID:-local}_0.out"
OUT1="bench_gpu_${SLURM_JOB_ID:-local}_1.out"

# BENCH_CONFIG benchmarks another recipe on the same ladder, e.g. configs/recipes/mu_only.yaml.
CONFIG_ARGS=()
[[ -n "${BENCH_CONFIG}" ]] && CONFIG_ARGS=(--config "${BENCH_CONFIG}")
echo "config    : ${BENCH_CONFIG:-the gpu suite default}"

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -m scripts.diagnostics.bench_train \
    --suite gpu --env snellius "${CONFIG_ARGS[@]}" --steps "${STEPS}" --n-utts "${N_UTTS}" \
    --batches "${LADDER[@]}" --only "${LADDER_TRIALS[@]}" > "${OUT0}" 2>&1 &
PID0=$!
(
    CUDA_VISIBLE_DEVICES="${GPU_IDS[1]:-${GPU_IDS[0]}}" python -m scripts.diagnostics.bench_train \
        --suite gpu --env snellius "${CONFIG_ARGS[@]}" --steps "${STEPS}" --n-utts "${N_UTTS}" \
        --at-batch "${AT_BATCH}" \
        --only "replay $((AT_BATCH / 2))" "replay ${AT_BATCH}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[1]:-${GPU_IDS[0]}}" python -m scripts.diagnostics.bench_train \
        --suite gpu --env snellius "${CONFIG_ARGS[@]}" --eval-timing --at-batch "${AT_BATCH}"
) > "${OUT1}" 2>&1 &
PID1=$!

failed=0
wait "${PID0}" || failed=1
wait "${PID1}" || failed=1

# The tables only; the full logs stay in the two files.
for f in "${OUT0}" "${OUT1}"; do
    echo
    echo "=== ${f}"
    grep -E '^(device|trial|batch [0-9]|replay|fastest|eval at|  validation|  sampling)' "${f}" \
        || tail -20 "${f}"
done
exit "${failed}"
