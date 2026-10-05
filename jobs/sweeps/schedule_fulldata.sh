#!/bin/bash

# Two questions, on the full train.100, one job per cell.
#
#   1. Is drawing t uniformly viable at full data?  -> the `uniform` arm.
#      At 4096 utterances it lost badly (93.3 / 94.2 against simulated's 73.2 / 72.3,
#      EXPERIMENTS row 27), but that is a regime where the model never sees enough of the
#      low-t region to learn it. With 14x the data and 300 epochs it may not need tau at
#      all, and that is worth knowing before any more is invested in the schedule.
#
#   2. How often should the measured curve be refreshed?  -> `meas-e1/e10/e30`.
#      The cadences are spread wide on purpose. At small scale e3/e5/e10 all landed inside
#      the 7-point noise floor (row 31), so a narrow spread cannot answer this; 1, 10 and
#      30 are 300, 30 and 10 refreshes over the run.
#
# `sim-e5` is the control and the most important job here. It reproduces skilled-lake-22
# (10.46 sampled WER, 3.6 h), so it says whether the pipeline still lands where it did
# before anything is concluded about the arms. The Oct 3 batch had no such anchor and no
# seed replication, and could not be read at all once `subset_n.train: 2000` was found.
#
# Every cell runs at two seeds. The noise floor at this scale is unmeasured — the control
# pair is what measures it, and no gap smaller than that pair's own spread is readable.
#
# Usage:
#   ./jobs/sweeps/schedule_fulldata.sh                 # dry run: print, submit nothing
#   SUBMIT=1 ./jobs/sweeps/schedule_fulldata.sh        # actually submit
#   ARMS="uniform sim-e5" SUBMIT=1 ./jobs/sweeps/...   # trim to a subset
#   SEEDS="0" SUBMIT=1 ./jobs/sweeps/...               # one seed only

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

CONFIG="${CONFIG:-configs/audio_cond.yaml}"
PROJECT="${PROJECT:-flow-asr-fulldata-schedule}"
ARMS="${ARMS:-uniform sim-e5 meas-e1 meas-e10 meas-e30}"
SEEDS="${SEEDS:-0 1}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-128}"

# Val, test and the measured probe run under no_grad, so a larger batch fits than training
# takes. It does not go faster for it: measured on a 3060, one probe is flat from chunk 32
# to 128 and slower above, because the (B, canvas_len, vocab_size) field already saturates
# the card. Unmeasured on an A100 — until it is, follow BATCH rather than guess, so the
# cadence arms are not separated by an untested throughput change.
# Worth its own cheap job: ABLATE_EVAL_BATCH=1 below.
EVAL_BATCH="${EVAL_BATCH:-${BATCH}}"

# skilled-lake-22: 66,600 steps in 3.6 h is 5.1 steps/s at batch 128. Sized a little under
# that, because fiery-resonance-37 took 20.5 h for the same work at batch 64 when GPFS was
# contended, and ten jobs queued together is exactly when that happens.
export STEPS_PER_SEC="${STEPS_PER_SEC:-4}"

# Held fixed at skilled-lake-22's values so the control is a real reproduction.
COMMON=(
    "data.subset_n.train=null"
    "data.batch_size=${BATCH}"
    "data.eval_batch_size=${EVAL_BATCH}"
    "train.num_epochs=${EPOCHS}"
    "train.denoiser_lr=0.002"
    "train.audio_decoder_lr=0.001"
    "logging.project=${PROJECT}"
)

arm_overrides() {
    case "$1" in
        uniform)  echo "schedule.enabled=false" ;;
        sim-e5)   echo "schedule.refresh_curve_from=simulated schedule.refresh_every_n_epochs=5" ;;
        meas-e1)  echo "schedule.refresh_curve_from=measured schedule.refresh_every_n_epochs=1" ;;
        meas-e10) echo "schedule.refresh_curve_from=measured schedule.refresh_every_n_epochs=10" ;;
        meas-e30) echo "schedule.refresh_curve_from=measured schedule.refresh_every_n_epochs=30" ;;
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
    echo "${n} jobs NOT submitted (dry run). Re-run with SUBMIT=1 to send them."
    echo "At ~4 h each that is ~$((n * 4)) GPU-hours."
fi
