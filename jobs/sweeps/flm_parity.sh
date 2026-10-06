#!/bin/bash

# Where FLM (src/models/fmlm/algo.py) does something we don't, one knob at a time. Each
# step pits the stack adopted so far against the stack plus one new knob:
#
#   1. uniform-time   flow.time_input=uniform    the denoiser sees the draw u, not t = tau^-1(u)
#   2. strat          flow.stratified_t=true     one t draw per 1/B slice of the batch
#   3. (reduction / padding arms, once the plot from
#       scripts/diagnostics/plot_loss_reduction.py has said which are worth running)
#
# A knob that beats the control by more than the control pair's own seed spread goes into
# KEEP, and the next step runs on top of it — the control included:
#
#   ARMS="control uniform-time" ./jobs/sweeps/flm_parity.sh                          # step 1
#   KEEP="flow.time_input=uniform" ARMS="control strat" ./jobs/sweeps/flm_parity.sh  # step 2
#
# The control is `sim-e5` from schedule_fulldata.sh, the skilled-lake-22 reproduction, so
# each step's control also checks that the pipeline still lands where it did.
#
# Both integrators are scored at every sampling epoch on the same x0
# (sampled/interpolant/wer, sampled/euler/wer); sampled/wer stays the interpolant's.
#
# Usage:
#   ./jobs/sweeps/flm_parity.sh                     # dry run: print, submit nothing
#   SUBMIT=1 ./jobs/sweeps/flm_parity.sh            # actually submit
#   SEEDS="0" ARMS="uniform-time" SUBMIT=1 ./jobs/...   # trim to a subset

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

CONFIG="${CONFIG:-configs/audio_cond.yaml}"
PROJECT="${PROJECT:-flow-asr-flm-parity}"
ARMS="${ARMS:-control uniform-time}"
KEEP="${KEEP:-}"
SEEDS="${SEEDS:-0 1}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-512}"
PROBE_N="${PROBE_N:-8192}"
# schedule_fulldata.sh measured ~80 s/epoch for sim-e5, 6.6 h for 300 epochs; 12 h covers
# that with the second integrator on top. See the note there on why over-requesting is the
# safe direction.
export WALLTIME="${WALLTIME:-12:00:00}"

# Held at skilled-lake-22's values so the control is a real reproduction.
COMMON=(
    "data.subset_n.train=null"
    "data.batch_size=${BATCH}"
    "data.eval_batch_size=${EVAL_BATCH}"
    "schedule.probe_n_utts=${PROBE_N}"
    "schedule.refresh_curve_from=simulated"
    "schedule.refresh_every_n_epochs=5"
    "train.num_epochs=${EPOCHS}"
    "train.denoiser_lr=0.002"
    "train.audio_decoder_lr=0.001"
    "sampling.log_methods=[interpolant,euler]"
    "logging.project=${PROJECT}"
)

arm_overrides() {
    case "$1" in
        control)      echo "" ;;
        uniform-time) echo "flow.time_input=uniform" ;;
        strat)        echo "flow.stratified_t=true" ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

cd "${REPO_DIR}"

# Tag adopted knobs into the group so steps do not share a wandb group by accident.
stack="base"
[[ -n "${KEEP}" ]] && stack="$(echo "${KEEP}" | tr ' =' '+-' | sed 's/flow\.//g')"

n=0
for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        overrides=(
            "seed=${seed}"
            "${COMMON[@]}"
            ${KEEP}
            $(arm_overrides "${arm}")
            "logging.group=${stack}/${arm}"
            "logging.name=${arm}-s${seed}"
        )
        n=$((n + 1))
        echo "=== [${n}] ${stack}/${arm}-s${seed}"
        if [[ -n "${SUBMIT}" ]]; then
            ./jobs/submit/train.sh "${CONFIG}" "${overrides[@]}"
        else
            echo "    ./jobs/submit/train.sh ${CONFIG} ${overrides[*]}"
        fi
    done
done

echo
if [[ -n "${SUBMIT}" ]]; then
    echo "submitted ${n} jobs to ${PROJECT}"
else
    echo "${n} jobs NOT submitted (dry run). Re-run with SUBMIT=1 to send them."
    echo "At ~7 h each that is ~$((n * 7)) GPU-hours."
fi
