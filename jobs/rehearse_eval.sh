#!/bin/bash

# Dress rehearsal for the eval array: same job script, same task count, same paths,
# a handful of utterances each. Catches the failures that only appear at scale —
# missing deps, quota, sharding overlap, merge — for a couple of GPU-minutes instead
# of a full run.
#
#   ./jobs/rehearse_eval.sh              # 8 tasks x 4 utts per split
#   N_JOBS=16 LIMIT=32 ./jobs/rehearse_eval.sh
#
# Submits two jobs: the array, then a verification job chained with
# --dependency=afterok so it only runs if every task succeeded. Watch both with
# `squeue -u $USER`; the verification job's output holds the PASS/FAIL.
#
# Results go to a scratch rehearsal directory, never the real results/ — a rehearsal
# manifest holds a few utterances and would otherwise be mistaken for a finished run.

set -eo pipefail

N_JOBS="${N_JOBS:-8}"
# Per split, before sharding, so each task gets LIMIT/N_JOBS utterances. Keep it a
# multiple of N_JOBS so every task has work; a task with zero utterances still
# exercises the write path but proves less.
LIMIT="${LIMIT:-$((N_JOBS * 4))}"
CONFIGS="${CONFIGS:-clean other}"
SPLITS="${SPLITS:-test}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
REHEARSAL_DIR="${REHEARSAL_DIR:-/scratch-shared/cont-flow-asr/rehearsal}"

rm -rf "${REHEARSAL_DIR}"
mkdir -p "${REHEARSAL_DIR}"

echo "=== submitting rehearsal array ==="
SUBMIT_OUT="$(
    N_JOBS="${N_JOBS}" LIMIT="${LIMIT}" CONFIGS="${CONFIGS}" SPLITS="${SPLITS}" \
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
    --job-name=eval_rehearse_check \
    --partition=rome \
    --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:15:00 \
    --output="${REHEARSAL_DIR}/verify_%j.out" \
    --wrap="cd ${REPO_DIR} && \
        module purge && module load 2025 && module load Anaconda3/2025.06-1 && \
        source activate ${CONDA_ENV:-flow_asr} && \
        export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} && \
        python -m scripts.check_eval --results-dir ${REHEARSAL_DIR} \
            --expect-total ${LIMIT} --num-shards ${N_JOBS} \
            --configs '${CONFIGS}' --splits '${SPLITS}'")"

echo "array job    : ${ARRAY_JOB}"
echo "verify job   : ${VERIFY_JOB}  (afterok:${ARRAY_JOB})"
echo "results dir  : ${REHEARSAL_DIR}"
echo
echo "Watch:   squeue -u \$USER"
echo "Verdict: cat ${REHEARSAL_DIR}/verify_${VERIFY_JOB}.out"
echo
echo "If the verify job never starts, the array had a failing task —"
echo "check its slurm_eval_${ARRAY_JOB}_*.out files."
