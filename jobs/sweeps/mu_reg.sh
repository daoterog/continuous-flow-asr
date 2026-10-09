#!/bin/bash

# mu memorises the training set: in the flm-parity controls its val CE bottoms out at
# epoch ~60 (0.40) and climbs to ~1.85 by 300, while its train P_e goes to 0.000 and its
# val P_e sits at ~0.50. It has no dropout (configs/mu/micro.yaml) and peaks at 1e-3 at
# epoch 90 under OneCycle — so it overfits while its LR is still rising. This sweeps the
# two levers that act there, mu's peak LR and its dropout; the denoiser is untouched.
#
#   lr3e-4         train.audio_decoder_lr=3e-4
#   do0.1          model.audio_decoder.dropout=0.1
#   do0.3          model.audio_decoder.dropout=0.3
#   lr3e-4-do0.1   both
#   lr3e-4-do0.3   both
#
# The control is the existing flm-parity `control` run of the same seed (defaults are
# unchanged), so no control is resubmitted here.
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
#   ./jobs/sweeps/mu_reg.sh                         # dry run
#   SUBMIT=1 ./jobs/sweeps/mu_reg.sh                # submit
#   ARMS="do0.3 lr3e-4-do0.3" SUBMIT=1 ./jobs/...   # trim

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

CONFIG="${CONFIG:-configs/audio_cond.yaml}"
PROJECT="${PROJECT:-flow-asr-mu-reg}"
ARMS="${ARMS:-lr3e-4 do0.1 do0.3 lr3e-4-do0.1 lr3e-4-do0.3}"
SEEDS="${SEEDS:-1}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-512}"
PROBE_N="${PROBE_N:-8192}"
# The flm-parity runs finished 300 epochs inside 12 h with both samplers on; this logs the
# interpolant only, so it is a little faster still.
export WALLTIME="${WALLTIME:-12:00:00}"

# The flm-parity control's overrides, so its runs are this sweep's control.
COMMON=(
    "data.subset_n.train=null"
    "data.batch_size=${BATCH}"
    "data.eval_batch_size=${EVAL_BATCH}"
    "schedule.probe_n_utts=${PROBE_N}"
    "schedule.refresh_curve_from=simulated"
    "schedule.refresh_every_n_epochs=5"
    "train.num_epochs=${EPOCHS}"
    "train.denoiser_lr=0.002"
    "logging.project=${PROJECT}"
)

arm_overrides() {
    case "$1" in
        lr3e-4)       echo "train.audio_decoder_lr=3e-4" ;;
        do0.1)        echo "train.audio_decoder_lr=1e-3 model.audio_decoder.dropout=0.1" ;;
        do0.3)        echo "train.audio_decoder_lr=1e-3 model.audio_decoder.dropout=0.3" ;;
        lr3e-4-do0.1) echo "train.audio_decoder_lr=3e-4 model.audio_decoder.dropout=0.1" ;;
        lr3e-4-do0.3) echo "train.audio_decoder_lr=3e-4 model.audio_decoder.dropout=0.3" ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

cd "${REPO_DIR}"

n=0
for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        overrides=(
            "seed=${seed}"
            "${COMMON[@]}"
            $(arm_overrides "${arm}")
            "logging.group=${arm}"
            "logging.name=${arm}-s${seed}"
        )
        n=$((n + 1))
        echo "=== [${n}] ${arm}-s${seed}"
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
    echo "${n} jobs NOT submitted (dry run). SUBMIT=1 to send them."
    echo "~7 h each, so ~$((n * 7)) GPU-hours."
fi
