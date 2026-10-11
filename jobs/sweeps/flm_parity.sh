#!/bin/bash

# Where FLM (src/models/fmlm/algo.py) does something we don't, each tested as an independent
# arm against one shared control on the cluster recipe (configs/recipes/fulldata.yaml:
# denoiser small + mu micro, full latents, batch 128, stratified t, token-level loss).
#
#   control         the recipe as is (pad_loss_weight 1.0)
#   uniform-time    flow.time_input=uniform           the denoiser sees the draw u, not t
#   pad0.1          flow.pad_loss_weight=0.1           less weight on padding, first pad too
#   pad0.1-decay8   ... + flow.pad_decay=8             first pad (the stop decision) kept
#                                                      at full weight, the tail decays to 0.1
#
# The pad arms rely on the recipe's token-level reduction: under `sequence` a short
# utterance's tokens would count up to ~6x more once pads are down-weighted, which is what
# confounded the earlier pad-decay run (21.1 vs 16.8).
#
# Read each arm against the control at epoch 100 (sampled WER, 16 steps, full val):
# >= 2 points better adopts it, >= 2 worse drops it — seed noise on small + micro is ~1-2.
# For the pad arms, also check samples still stop at the right length (WER vs
# denoiser_perr): an under-learnt stop token shows up as insertions.
#
# Earlier rounds of this sweep ran on the tiny denoiser at stride 4 and are not comparable:
# there, uniform time split by seed (-2.2 / +1.3), stratified t was flat (now a recipe
# default), and Euler sampled ~1.5 points worse than the interpolant.
#
# KEEP adds adopted knobs to every arm, the control included, for a next stage.
#
# Usage:
#   ./jobs/sweeps/flm_parity.sh                     # dry run: plan + SBU, submit nothing
#   SUBMIT=1 ./jobs/sweeps/flm_parity.sh            # screen, gated
#   KEEP="flow.time_input=uniform" ARMS="control ..." SUBMIT=1 ./jobs/sweeps/flm_parity.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# shellcheck source=../common/sweep.sh
source "${REPO_DIR}/jobs/common/sweep.sh"
tier_defaults

CONFIG="${CONFIG:-configs/recipes/fulldata.yaml}"
PROJECT="${PROJECT:-flow-asr-flm-parity}"
ARMS="${ARMS:-control uniform-time pad0.1 pad0.1-decay8}"
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
        pad0.1)        echo "flow.pad_loss_weight=0.1" ;;
        pad0.1-decay8) echo "flow.pad_loss_weight=0.1 flow.pad_decay=8" ;;
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
