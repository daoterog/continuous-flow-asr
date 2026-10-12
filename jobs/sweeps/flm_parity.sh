#!/bin/bash

# Where FLM (src/models/fmlm/algo.py) does something we don't, each tested as an independent
# arm against one shared control on the cluster recipe (configs/recipes/fulldata.yaml).
#
# Stage 1 (100 epochs, seed 1, sampled WER @ e99) — both winners are now in the recipe:
#   control 17.99 | uniform time 15.17 (-2.8) | pad 0.1 flat 14.94 (-3.1)
# so `control` below is the stage-1 stack: small + micro, full latents, stratified t,
# token-level loss, uniform time input, pad weight 0.1.
#
# Stage 2 arms, each one change from that control:
#
#   control     the recipe as is
#   ema         train.ema_decay=0.999           validate, sample and save an EMA of the
#                                               weights, as FLM does (half-life ~700 steps)
#   softcap     model.denoiser.logit_cap=30     logits soft-capped by 30*tanh(x/30), as FLM
#   decay8      flow.pad_decay=8                first pad (the stop decision) back at full
#                                               weight, the tail decaying to 0.1
#   lr1e-3      train.denoiser_lr=1e-3          an LR bracket around the recipe's 2e-3
#   lr4e-3      train.denoiser_lr=4e-3
#
# Stage 2 results (sampled WER @ e99, control 15.49): softcap 15.29 (adopted, free),
# lr4e-3 16.01, lr1e-3 17.76, decay8 18.68 (all dropped); ema was OOM-killed at epoch 20 —
# a host-memory problem, not EMA's. Stage 3 reruns it on the stack + softcap:
#
#   STAGE=stage3 ARMS="control ema" SUBMIT=1 ./jobs/sweeps/flm_parity.sh
#
# Read each arm against the control at epoch 100 (sampled WER, 16 steps, full val):
# >= 2 points better adopts it, >= 2 worse drops it — seed noise on small + micro is ~1-2.
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
ARMS="${ARMS:-control ema softcap decay8 lr1e-3 lr4e-3}"
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
        control) echo "" ;;
        ema)     echo "train.ema_decay=0.999" ;;
        softcap) echo "model.denoiser.logit_cap=30" ;;
        decay8)  echo "flow.pad_decay=8" ;;
        lr1e-3)  echo "train.denoiser_lr=1e-3" ;;
        lr4e-3)  echo "train.denoiser_lr=4e-3" ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

cd "${REPO_DIR}"

# Tag the stage (and any adopted knobs) into the group so stages never share a wandb group.
stack="${STAGE:-stage2}"
[[ -n "${KEEP}" ]] && stack="${stack}+$(echo "${KEEP}" | tr ' =' '+-' | sed 's/flow\.//g')"

for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        queue_run "${stack}/${arm}-s${seed}" "seed=${seed}" "${COMMON[@]}" ${KEEP} \
            $(arm_overrides "${arm}") "logging.group=${TIER}${EPOCHS}/${stack}/${arm}" \
            "logging.name=${stack}-${arm}-e${EPOCHS}-s${seed}"
    done
done
flush_runs
