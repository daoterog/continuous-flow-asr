#!/bin/bash

# Price, gate and submit the A100 throughput benchmark (jobs/batch/bench_train.sh).
#
#   ./jobs/submit/bench_train.sh                          # submit, if the budget allows
#
# Expected ~30 min: five single-run trials of a few minutes each (build, compile warmup,
# 200 timed steps). Billed as 2 GPUs, like any job that needs 160G.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# shellcheck source=../common/budget.sh
source "${REPO_DIR}/jobs/common/budget.sh"

EXPECTED_S="${EXPECTED_S:-1800}"
EST="$(job_sbu "${EXPECTED_S}" 1)"
echo "expected  : ${EXPECTED_S}s, charged as $(charged_gpus 1) GPUs = ${EST} SBU"
budget_gate "${EST}"

JOBID="$(cd "${REPO_DIR}" && sbatch --parsable --export=ALL "${REPO_DIR}/jobs/batch/bench_train.sh")"
JOBID="${JOBID%%;*}"
echo "submitted : job ${JOBID} (output in slurm_bench_train_${JOBID}.out)"
ledger_append "${JOBID}" "bench" "bench" "-" "1" "${EST}" "bench_train"
