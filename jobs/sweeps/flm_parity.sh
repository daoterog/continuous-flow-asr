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
# Step 1 is done (flm-parity, 300 epochs x 2 seeds): no arm beat the control's seed
# spread, and Euler sampled ~1.5 points worse than the interpolant in all 8 runs, so only
# the interpolant is scored now.
#
# Runs on the standard tiers (jobs/common/sweep.sh): a 100-epoch screen by default, two
# runs per GPU; TIER=confirm for 300 epochs x 2 seeds.
#
# Usage:
#   ./jobs/sweeps/flm_parity.sh                     # dry run: plan + SBU, submit nothing
#   SUBMIT=1 ./jobs/sweeps/flm_parity.sh            # screen, gated
#   TIER=confirm ARMS="control strat" SUBMIT=1 ./jobs/sweeps/flm_parity.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# shellcheck source=../common/sweep.sh
source "${REPO_DIR}/jobs/common/sweep.sh"
tier_defaults

CONFIG="${CONFIG:-configs/recipes/fulldata.yaml}"
PROJECT="${PROJECT:-flow-asr-flm-parity}"
ARMS="${ARMS:-control uniform-time}"
KEEP="${KEEP:-}"

# Everything that defines the run lives in the recipe ($CONFIG): model, data, LRs,
# schedule. The machine and the run kind come from configs/profiles/. Only logging and the
# arms vary here.
COMMON=(
    "logging.project=${PROJECT}"
    ${EPOCH_OVERRIDE}
)

arm_overrides() {
    case "$1" in
        control)       echo "" ;;
        uniform-time)  echo "flow.time_input=uniform" ;;
        strat)         echo "flow.stratified_t=true" ;;
        strat-uniform) echo "flow.time_input=uniform flow.stratified_t=true" ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

cd "${REPO_DIR}"

# Tag adopted knobs into the group so steps do not share a wandb group by accident.
stack="base"
[[ -n "${KEEP}" ]] && stack="$(echo "${KEEP}" | tr ' =' '+-' | sed 's/flow\.//g')"

for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        queue_run "${stack}/${arm}-s${seed}" "seed=${seed}" "${COMMON[@]}" ${KEEP} \
            $(arm_overrides "${arm}") "logging.group=${TIER}${EPOCHS}/${stack}/${arm}" \
            "logging.name=${arm}-e${EPOCHS}-s${seed}"
    done
done
flush_runs
