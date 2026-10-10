#!/bin/bash

# Price, gate and submit the A100 throughput benchmark (jobs/batch/bench_train.sh).
#
#   ./jobs/submit/bench_train.sh                          # submit, if the budget allows
#
# Expected ~30 min on 2 GPUs: the batch ladder on one, the compute knobs and the eval
# timing on the other. 2 GPUs is what any job needing 160G is billed for anyway.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# shellcheck source=../common/budget.sh
source "${REPO_DIR}/jobs/common/budget.sh"

EXPECTED_S="${EXPECTED_S:-1800}"
read -r GPUS CPUS MEM_G <<< "$(job_resources 2)"
EST="$(job_sbu "${EXPECTED_S}" 2)"
echo "expected  : ${EXPECTED_S}s on ${GPUS} GPUs (${CPUS} CPUs, ${MEM_G}G) = ${EST} SBU"
budget_gate "${EST}"

JOBID="$(cd "${REPO_DIR}" && sbatch --parsable --gpus="${GPUS}" --cpus-per-task="${CPUS}" --mem="${MEM_G}G" \
    --export=ALL "${REPO_DIR}/jobs/batch/bench_train.sh")"
JOBID="${JOBID%%;*}"
echo "submitted : job ${JOBID} (output in slurm_bench_train_${JOBID}.out)"
ledger_append "${JOBID}" "bench" "bench" "-" "1" "${EST}" "bench_train"
