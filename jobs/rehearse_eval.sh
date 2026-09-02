#!/bin/bash

# Dress rehearsal for the eval array: same job script, same task count, same paths,
# a handful of utterances each. Catches the failures that only appear at scale — missing
# deps, quota, sharding overlap, merge — for a couple of GPU-minutes instead of a full run.
#
#   ./jobs/rehearse_eval.sh              # 8 tasks x 4 utts per dataset
#   N_JOBS=16 LIMIT=32 ./jobs/rehearse_eval.sh
#   DATASETS="librispeech_test.clean" ./jobs/rehearse_eval.sh
#
# Submits the array, then chains jobs/check_eval.sh with --dependency=afterok so the
# verification runs only if every task succeeded. Watch both with `squeue -u $USER`.
#
# Results go to a scratch rehearsal directory, never the real results/ — a rehearsal
# manifest holds a few utterances and would otherwise look like a finished run.

set -eo pipefail

N_JOBS="${N_JOBS:-8}"
# Per dataset, before sharding, so each task gets LIMIT/N_JOBS utterances. Keep it a
# multiple of N_JOBS so every task has work; a task with zero utterances still exercises
# the write path but proves less.
LIMIT="${LIMIT:-$((N_JOBS * 4))}"
DATASETS="${DATASETS:-ami_cleaned gigaspeech_cleaned spgispeech voxpopuli_cleaned_aa librispeech_test.clean librispeech_test.other earnings22_cleaned_aa_chunked}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
REHEARSAL_DIR="${REHEARSAL_DIR:-/scratch-shared/cont-flow-asr/rehearsal}"
CHECK_PARTITION="${CHECK_PARTITION:-rome}"

rm -rf "${REHEARSAL_DIR}"
mkdir -p "${REHEARSAL_DIR}"

echo "=== submitting rehearsal array ==="
SUBMIT_OUT="$(
    N_JOBS="${N_JOBS}" LIMIT="${LIMIT}" DATASETS="${DATASETS}" \
    RESULTS_DIR="${REHEARSAL_DIR}" \
    "${SCRIPT_DIR}/submit_eval.sh"
)"
echo "${SUBMIT_OUT}"
ARRAY_JOB="$(echo "${SUBMIT_OUT}" | awk '/Submitted batch job/ {print $NF}')"
if [[ -z "${ARRAY_JOB}" ]]; then
    echo "ERROR: could not parse the array job id from sbatch output" >&2
    exit 1
fi

echo
echo "=== chaining verification (runs only if all ${N_JOBS} tasks succeed) ==="
VERIFY_JOB="$(sbatch --parsable \
    --dependency=afterok:"${ARRAY_JOB}" \
    --partition="${CHECK_PARTITION}" \
    --output="${REHEARSAL_DIR}/verify_%j.out" \
    --export=ALL,REPO_HOME="${REPO_DIR}",RESULTS_DIR="${REHEARSAL_DIR}",NUM_SHARDS="${N_JOBS}",EXPECT_TOTAL="${LIMIT}",DATASETS="${DATASETS}" \
    "${SCRIPT_DIR}/check_eval.sh")"

echo "array job    : ${ARRAY_JOB}"
echo "verify job   : ${VERIFY_JOB}  (afterok:${ARRAY_JOB})"
echo "results dir  : ${REHEARSAL_DIR}"
echo
echo "Watch:   squeue -u \$USER"
echo "Verdict: cat ${REHEARSAL_DIR}/verify_${VERIFY_JOB}.out"
echo
echo "If the verify job never starts, the array had a failing task —"
echo "check its slurm_eval_${ARRAY_JOB}_*.out files."
