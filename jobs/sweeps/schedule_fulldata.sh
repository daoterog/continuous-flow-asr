#!/bin/bash

# Predates the budget standards (jobs/common/sweep.sh): one run per GPU job at its own
# epoch count. Each submit still prints its SBU, is gated against the cap and lands in
# the ledger, but new sweeps should use the screen/confirm tiers instead.
#
# Its CONFIG is a plain variant, so on Snellius train.py now refuses it (cluster runs must
# come from configs/recipes/). That is deliberate: these runs trained base.yaml's tiny
# denoiser at stride 4. To rerun one, point CONFIG at a recipe, or pass recipe=<name> to
# reproduce the old setting on purpose.

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
ARMS="${ARMS:-uniform sim-e5 meas-e10 warm-e1 warm-e10 warm-e30}"
SEEDS="${SEEDS:-0 1}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-128}"

# Val, test and the measured probe run under no_grad, so a larger batch both fits and pays.
# Measured on the A100 (jobs/batch/bench_probe.sh): 0.836 ms/utt at chunk 32 against
# 0.645 at 512, 24% off, flat above. 512 takes all but 1% of that for 5.7 GiB, which sits
# alongside training; 2048 wants 22 GiB of the card's 40 to save another 1%.
EVAL_BATCH="${EVAL_BATCH:-512}"

# 2048 was chosen for a binned estimator that needed 150 bins kept populated. Centered
# isotonic pools adaptively, so the constraint is gone and precision is 1/sqrt(n) with no
# plateau: against the closed-form curve, 8192 halves the error of 2048 at t0
# (0.0015 against 0.0077) and is 1.46x better overall.
#
# Affordable because the probe costs exactly what it forwards — 5.29 s at chunk 512, 6.5%
# of an 81 s epoch, so 26 min added to a 300-epoch run at e1 and 2.6 min at e10. Held the
# same across every arm, so it improves all of them rather than separating any.
#
# Not higher: 8192 is 29% of the ~28,416-utterance split, and the point of redrawing each
# refresh is that successive curves are independent samples. 16384 is 58% of it.
PROBE_N="${PROBE_N:-8192}"

# Pinned rather than left to submit/train.sh's estimator, which would also rebuild the
# dataset on the login node once per job to count steps.
#
# Measured, from the first full-data batch, rather than derived. Every one of those ten
# jobs hit a 6 h limit and stopped between epoch 199 and 273 of 300 — and reported FINISHED
# in wandb, because Lightning's SLURM handler exits cleanly on the signal. Their own
# seconds-per-epoch, and what 300 epochs would have needed:
#
#     uniform / sim-e5      79.0 - 79.8 s      6.58 - 6.65 h
#     meas-e10 / meas-e30   80.3 - 82.4 s      6.69 - 6.86 h
#     meas-e1               95.0 - 108.5 s     7.92 - 9.04 h
#
# An earlier version of this comment budgeted 4 h from a warm-GPFS rate of 43 s/epoch. That
# rate is real but does not describe these jobs: at probe_n 8192 with train WER bins on,
# the steady state is ~80 s/epoch, and refresh-every-epoch adds another 15-30 on top.
#
# 12 h is 1.33x the worst arm. Deliberately generous, because the two directions are not
# symmetric: over-request costs queue priority, under-request silently returns a truncated
# run that reads as a result — and on top of that OneCycleLR is sized for the full 300
# epochs, so a job cut at 271 stops with the LR still 11,578x above its intended final
# value, having never had the anneal that was supposed to consolidate it.
export WALLTIME="${WALLTIME:-12:00:00}"

# Held fixed at skilled-lake-22's values so the control is a real reproduction.
COMMON=(
    "data.subset_n.train=null"
    "data.batch_size=${BATCH}"
    "data.eval_batch_size=${EVAL_BATCH}"
    "schedule.probe_n_utts=${PROBE_N}"
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
        warm-e1)  echo "schedule.refresh_curve_from=warmup schedule.refresh_every_n_epochs=1" ;;
        warm-e10) echo "schedule.refresh_curve_from=warmup schedule.refresh_every_n_epochs=10" ;;
        warm-e30) echo "schedule.refresh_curve_from=warmup schedule.refresh_every_n_epochs=30" ;;
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
