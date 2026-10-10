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
# literal `::` starts the next run's overrides; the runs share the GPU (see
# jobs/batch/train.sh). The walltime is sized off the first run.
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
echo "walltime  : ${WALLTIME}"

# What this job is expected to cost: the caller's figure, else the step-count estimate
# (scaled for packing), else the whole walltime as the worst case.
if [[ -n "${EXPECTED_S}" ]]; then
    JOB_S="${EXPECTED_S}"
elif [[ -n "${EST_S}" ]]; then
    JOB_S="$(awk -v e="${EST_S}" -v n="${#RUNS[@]}" -v f="${PACK_SLOWDOWN}" -v o="${OVERHEAD_S}" \
        'BEGIN { printf "%d", e * (n > 1 ? f : 1) + o }')"
else
    JOB_S="$(_slurm_seconds "${WALLTIME}")"
fi
JOB_SBU="$(seconds_to_sbu "${JOB_S}")"
echo "expected  : $(fmt_hms "${JOB_S}") on 1 GPU = ${JOB_SBU} SBU for ${#RUNS[@]} run(s)"
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
    --time="${WALLTIME}" \
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
