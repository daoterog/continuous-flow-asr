#!/bin/bash

# Submit one GPU job running one or more training runs, with a walltime derived from the
# config's own step count, and an SBU estimate checked against the budget before sbatch.
#
# Usage:
#   ./jobs/submit/train.sh configs/linear.yaml
#   ./jobs/submit/train.sh configs/audio_cond.yaml train.num_epochs=20
#   ./jobs/submit/train.sh configs/audio_cond.yaml seed=0 :: seed=1     # two runs, one GPU
#   STEPS_PER_SEC=12 ./jobs/submit/train.sh configs/linear.yaml     # after measuring
#   WALLTIME=02:00:00 ./jobs/submit/train.sh configs/linear.yaml    # skip the estimate
#
# Everything after the config is passed through to train.py as OmegaConf overrides. A
# literal `::` starts the next run's overrides. Each run gets its own GPU and 18 cores in
# one job (see jobs/batch/train.sh), which for two runs costs the same as one: a job is
# billed for 2 GPUs either way (jobs/common/budget.sh). The walltime is sized off the first
# run.
#
# Budget (jobs/common/budget.sh): the expected cost is EXPECTED_S seconds when the caller
# knows it (jobs/common/sweep.sh passes it), else the step-count estimate, else the whole
# walltime. A direct submission is gated against the cap here; a sweep gates its total
# once and passes BUDGET_GATED=1. Every job is appended to jobs/ledger.tsv.

set -eo pipefail

CONFIG="${1:?usage: $0 <config.yaml> [overrides...] [:: overrides...]}"
shift || true
RUNS=()
current=""
for arg in "$@"; do
    if [[ "${arg}" == "::" ]]; then
        RUNS+=("${current# }")
        current=""
    else
        current="${current} ${arg}"
    fi
done
RUNS+=("${current# }")
OVERRIDES="${RUNS[0]}"

# Throughput, measured on an A100 node (scripts/diagnostics/bench_train.py, batch 256 + 16 workers +
# compile). Two regimes, 15x apart, and which one applies is decided by whether the data
# fits the job's 64G of RAM:
#   ~3350 utts/s  reading a subset the page cache holds
#   ~220 utts/s   reading more than it holds — ~800 MiB/s off GPFS, the real ceiling
# train.100 is 102 GiB untrimmed, so a full-data run is the second regime. Size off it.
#
# This is a utts/s ceiling, so steps/s falls as the batch grows: STEPS_PER_SEC ~ 220/batch.
# The default suits the default batch of 32; at batch 256 pass STEPS_PER_SEC=0.9.
STEPS_PER_SEC="${STEPS_PER_SEC:-6}"
SAFETY="${SAFETY:-1.3}"
OVERHEAD_S="${OVERHEAD_S:-600}"   # module load, conda, tokenizer, tau sweep, first epoch
MIN_S=900                         # 15 min
MAX_S=432000                      # 5 days, the gpu_a100 limit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
BATCH_DIR="${REPO_DIR}/jobs/batch"

# shellcheck source=../common/budget.sh
source "${REPO_DIR}/jobs/common/budget.sh"

fmt_hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $((($1 % 3600) / 60)) $(($1 % 60)); }

SIZING_ERR="$(mktemp)"
trap 'rm -f "${SIZING_ERR}"' EXIT

WALLTIME_GIVEN="${WALLTIME}"
if [[ -z "${WALLTIME}" ]]; then
    # --print-steps builds the data and counts batches; no model, no GPU, so it runs
    # here on the login node. It needs the features to exist, which they must anyway.
    TOTAL_STEPS="$(cd "${REPO_DIR}" && python train.py --config "${CONFIG}" \
        --where snellius --run "${TRAIN_RUN:-full}" ${OVERRIDES} --print-steps \
        2>"${SIZING_ERR}" \
        | awk '/^total_steps/ {print $2}' || true)"

    if [[ -z "${TOTAL_STEPS}" ]]; then
        # Loud, and not the partition maximum: a 5-day request would sit in the queue
        # far longer than the run needs, and being wrong in that direction is penalised.
        echo "ERROR: could not size the run. Reason:" >&2
        sed 's/^/  | /' "${SIZING_ERR}" >&2
        echo "  Pass an explicit walltime, e.g. WALLTIME=02:00:00 $0 ${CONFIG}" >&2
        exit 1
    fi

    EST_S=$(awk -v s="${TOTAL_STEPS}" -v r="${STEPS_PER_SEC}" 'BEGIN{printf "%d", s/r}')
    WALL_S=$(awk -v e="${EST_S}" -v f="${SAFETY}" -v o="${OVERHEAD_S}" \
        'BEGIN{printf "%d", o + e*f}')
    ((WALL_S < MIN_S)) && WALL_S=${MIN_S}
    ((WALL_S > MAX_S)) && WALL_S=${MAX_S}
    WALLTIME="$(fmt_hms ${WALL_S})"

    echo "steps     : ${TOTAL_STEPS}"
    echo "estimate  : ${EST_S}s at ${STEPS_PER_SEC} steps/s, x${SAFETY} + ${OVERHEAD_S}s overhead"
fi

echo "config    : ${CONFIG}"
for i in "${!RUNS[@]}"; do
    echo "run ${i}     : ${RUNS[$i]:-(no overrides)}"
done
echo "profile   : --where snellius --run ${TRAIN_RUN:-full}"
echo "partition : ${PARTITION}"
echo "walltime  : ${WALLTIME}"

# What this job is expected to cost: the caller's figure, else the step-count estimate,
# else the whole walltime as the worst case — times the GPUs it is charged for.
if [[ -n "${EXPECTED_S}" ]]; then
    JOB_S="${EXPECTED_S}"
elif [[ -n "${EST_S}" ]]; then
    JOB_S=$((EST_S + OVERHEAD_S))
else
    JOB_S="$(_slurm_seconds "${WALLTIME}")"
fi
# A caller's WALLTIME is checked against how long the runs actually are — a derived one
# already came from the step count, max_steps included. Their epoch count
# is an explicit train.num_epochs override, else the --run profile's. A hand-written
# submission once forgot TRAIN_RUN=sweep, got the 300-epoch `full` profile under a 3 h
# walltime, and was killed at epoch 201 — paid for, and with no usable result.
if [[ -n "${WALLTIME_GIVEN}" && -z "${WALLTIME_OK}" ]]; then
    epochs_run="$(echo "${RUNS[0]}" | grep -o 'train.num_epochs=[0-9]*' | cut -d= -f2 || true)"
    if [[ -z "${epochs_run}" ]]; then
        epochs_run="$(awk '$1 == "num_epochs:" {print $2}' \
            "${REPO_DIR}/configs/profiles/run/${TRAIN_RUN:-full}.yaml" || true)"
    fi
    if [[ -n "${epochs_run}" ]]; then
        need_s="$(est_job_seconds "${epochs_run}")"
        wall_s="$(_slurm_seconds "${WALLTIME}")"
        echo "epochs    : ${epochs_run} (--run ${TRAIN_RUN:-full}), expected $(fmt_hms "${need_s}")"
        if ((wall_s < need_s)); then
            echo "ERROR: WALLTIME ${WALLTIME} is shorter than the expected $(fmt_hms "${need_s}")" >&2
            echo "       for ${epochs_run} epochs under --run ${TRAIN_RUN:-full}; the run would be cut" >&2
            echo "       off before it finishes. Raise WALLTIME, set TRAIN_RUN=sweep for a 100-epoch" >&2
            echo "       screen, or pass WALLTIME_OK=1 if the cut is deliberate." >&2
            exit 1
        fi
    fi
fi

read -r GPUS CPUS MEM_G <<< "$(job_resources "${#RUNS[@]}")"
JOB_SBU="$(job_sbu "${JOB_S}" "${#RUNS[@]}")"
echo "resources : ${GPUS} GPU, ${CPUS} CPUs, ${MEM_G}G -> charged as $(charged_gpus "${#RUNS[@]}") GPUs"
echo "expected  : $(fmt_hms "${JOB_S}") = ${JOB_SBU} SBU for ${#RUNS[@]} run(s)"
if [[ -z "${BUDGET_GATED}" ]]; then
    budget_gate "${JOB_SBU}"
fi

# Through the environment rather than as `--export=ALL,KEY=value`: sbatch splits that list
# on commas, so an override such as `sampling.log_methods=[interpolant,euler]` would reach
# the job cut off at `[interpolant`.
export TRAIN_CONFIG="${CONFIG}"
export TRAIN_RUN="${TRAIN_RUN:-full}"
export TRAIN_N="${#RUNS[@]}"
for i in "${!RUNS[@]}"; do
    export "TRAIN_OVERRIDES_${i}=${RUNS[$i]}"
done
JOBID="$(sbatch --parsable \
    --partition="${PARTITION}" --time="${WALLTIME}" \
    --gpus="${GPUS}" --cpus-per-task="${CPUS}" --mem="${MEM_G}G" \
    --export=ALL \
    "${BATCH_DIR}/train.sh")"
JOBID="${JOBID%%;*}"
echo "submitted : job ${JOBID}"

# `|| true`: a run without one of these keys is normal, and under `set -eo pipefail` a grep
# that finds nothing would abort here — after sbatch, so the job runs but is never ledgered.
names="$(for r in "${RUNS[@]}"; do echo "${r}" | grep -o 'logging.name=[^ ]*' | cut -d= -f2 || true; done | paste -sd, -)"
project="$(echo "${RUNS[0]}" | grep -o 'logging.project=[^ ]*' | cut -d= -f2 || true)"
epochs="$(echo "${RUNS[0]}" | grep -o 'train.num_epochs=[^ ]*' | cut -d= -f2 || true)"
epochs="${epochs:-${EPOCHS}}"
ledger_append "${JOBID}" "${project:-?}" "${TIER:-direct}" "${epochs:-?}" "${#RUNS[@]}" \
    "${JOB_SBU}" "${names:-?}"
