#!/bin/bash

# NOTE (2026-10-10): the overfitting below was measured on the 256x2 mu that the
# flm-parity runs inherited by accident. The recipe now trains mu `micro` (256x1), whose
# train/val gap was only 0.036 (EXPERIMENTS row 13), and row 14 already found dropout 0.1
# makes micro worse. Re-run this only if a stride-1 small + micro control shows mu
# overfitting again.
#
# mu memorises the training set: in the flm-parity controls its val CE bottoms out at
# epoch ~60 (0.40) and climbs to ~1.85 by 300, while its train P_e goes to 0.000 and its
# val P_e sits at ~0.50. It has no dropout (configs/mu/micro.yaml) and peaks at 1e-3 at
# epoch 90 under OneCycle — so it overfits while its LR is still rising. This sweeps the
# two levers that act there, mu's peak LR and its dropout; the denoiser is untouched.
#
#   control        mu at 1e-3, no dropout, as in every run so far
#   lr3e-4         train.audio_decoder_lr=3e-4
#   do0.1          model.audio_decoder.dropout=0.1
#   do0.3          model.audio_decoder.dropout=0.3
#   lr3e-4-do0.1   both
#   lr3e-4-do0.3   both
#
# Runs on the standard tiers (jobs/common/sweep.sh): a 100-epoch screen by default, two
# runs per GPU, then TIER=confirm on the winner. A screen carries its own control, since a
# 100-epoch run is only comparable to another 100-epoch run.
#
# One seed is enough to read mu: across the two control seeds its val P_e was 0.496 vs
# 0.499 and its val-CE minimum 0.396 at epoch 58 vs 0.396 at 59. Sampled WER is noisier
# (6 points between seeds), so read it paired against the same-seed control, and treat
# only a large gap as a result.
#
# Read, per arm: mu_loss/val_overall (minimum, and how far it climbs after it),
# mu_perr/val_overall at the end, the train/val gap, then sampled_interpolant/wer.
#
# Usage:
#   ./jobs/sweeps/mu_reg.sh                                     # dry run: plan + SBU
#   SUBMIT=1 ./jobs/sweeps/mu_reg.sh                            # screen, gated
#   TIER=confirm ARMS="control do0.3" SUBMIT=1 ./jobs/sweeps/mu_reg.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# shellcheck source=../common/sweep.sh
source "${REPO_DIR}/jobs/common/sweep.sh"
tier_defaults

CONFIG="${CONFIG:-configs/recipes/fulldata.yaml}"
PROJECT="${PROJECT:-flow-asr-mu-reg}"
ARMS="${ARMS:-control lr3e-4 do0.1 do0.3 lr3e-4-do0.1 lr3e-4-do0.3}"

# Everything that defines the run lives in the recipe ($CONFIG): model, data, LRs,
# schedule. The machine and the run kind come from configs/profiles/. Only logging and the
# arms vary here.
COMMON=(
    "logging.project=${PROJECT}"
    ${EPOCH_OVERRIDE}
)

arm_overrides() {
    case "$1" in
        control)      echo "train.audio_decoder_lr=1e-3" ;;
        lr3e-4)       echo "train.audio_decoder_lr=3e-4" ;;
        do0.1)        echo "train.audio_decoder_lr=1e-3 model.audio_decoder.dropout=0.1" ;;
        do0.3)        echo "train.audio_decoder_lr=1e-3 model.audio_decoder.dropout=0.3" ;;
        lr3e-4-do0.1) echo "train.audio_decoder_lr=3e-4 model.audio_decoder.dropout=0.1" ;;
        lr3e-4-do0.3) echo "train.audio_decoder_lr=3e-4 model.audio_decoder.dropout=0.3" ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

cd "${REPO_DIR}"

for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        queue_run "${arm}-s${seed}" "seed=${seed}" "${COMMON[@]}" $(arm_overrides "${arm}") \
            "logging.group=${TIER}${EPOCHS}/${arm}" "logging.name=${arm}-e${EPOCHS}-s${seed}"
    done
done
flush_runs
