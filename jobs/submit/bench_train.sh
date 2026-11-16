#!/bin/bash

# Price, gate and submit the A100 throughput benchmark (jobs/batch/bench_train.sh).
#
#   ./jobs/submit/bench_train.sh                          # submit, if the budget allows
#   SUITE=prefetch ./jobs/submit/bench_train.sh           # data.gpu_prefetch vs plain, ~15 min
#   SUITE=tune ./jobs/submit/bench_train.sh               # matmul high, prefetch_factor, workers
#   SUITE=now ./jobs/submit/bench_train.sh                # today's recipe on one staged slice
#
# Expected ~30 min on 2 GPUs: the batch ladder on one, the compute knobs and the eval
# timing on the other. 2 GPUs is what any job needing 160G is billed for anyway.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# shellcheck source=../common/slurm_guard.sh
source "${REPO_DIR}/jobs/common/slurm_guard.sh"
# shellcheck source=../common/budget.sh
source "${REPO_DIR}/jobs/common/budget.sh"

# SUITE=prefetch is two trials per GPU, ~15 min; the default gpu suite ~30.
if [[ "${SUITE:-gpu}" == "prefetch" || "${SUITE:-gpu}" == "tune" ]]; then
    EXPECTED_S="${EXPECTED_S:-1200}"
elif [[ "${SUITE:-gpu}" == "now" ]]; then
    EXPECTED_S="${EXPECTED_S:-2700}"
else
    EXPECTED_S="${EXPECTED_S:-1800}"
fi
# SUITE=now runs on one slice like a training job: 1 GPU, 18 cores, 120G, the features
# staged on node-local NVMe (~10 min for train.100's 121 GiB) before ~8 trials and an eval.
CONSTRAINT=()
if [[ "${SUITE:-gpu}" == "now" ]]; then
    read -r GPUS CPUS MEM_G <<< "$(STAGED=1 job_resources 1)"
    EST="$(STAGED=1 job_sbu "${EXPECTED_S}" 1)"
    CONSTRAINT=(--constraint=scratch-node)
else
    read -r GPUS CPUS MEM_G <<< "$(job_resources 2)"
    EST="$(job_sbu "${EXPECTED_S}" 2)"
fi
echo "expected  : ${EXPECTED_S}s on ${GPUS} GPUs (${CPUS} CPUs, ${MEM_G}G) = ${EST} SBU"
budget_gate "${EST}"

JOBID="$(cd "${REPO_DIR}" && guarded_sbatch --parsable --gpus="${GPUS}" --cpus-per-task="${CPUS}" \
    --mem="${MEM_G}G" --time="$(awk -v s="${EXPECTED_S}" 'BEGIN { w = 2 * s; printf "%02d:%02d:00", w / 3600, (w % 3600) / 60 }')" \
    "${CONSTRAINT[@]}" \
    --export=ALL "${REPO_DIR}/jobs/batch/bench_train.sh")"
JOBID="${JOBID%%;*}"
echo "submitted : job ${JOBID} (output in slurm_bench_train_${JOBID}.out)"
ledger_append "${JOBID}" "bench" "bench" "-" "1" "${EST}" "bench_train"
