#!/bin/bash

# Predates the budget standards (jobs/common/sweep.sh): one run per GPU job at its own
# epoch count. Each submit still prints its SBU, is gated against the cap and lands in
# the ledger, but new sweeps should use the screen/confirm tiers instead.
#
# Its CONFIG is a plain variant, so on Snellius train.py now refuses it (cluster runs must
# come from configs/recipes/). That is deliberate: these runs trained base.yaml's tiny
# denoiser at stride 4. To rerun one, point CONFIG at a recipe, or pass recipe=<name> to
# reproduce the old setting on purpose.

# How long should warmup last, and does the refresh cadence interact with it?
#
# `measured_max_median_t` is the knob that decides when the handover fires: the measured
# curve is refused while it would draw half its batch above that t. The observed medians
# across five earlier runs were
#
#     epoch 1    0.9931 - 0.9998        epoch 3    0.8247 - 0.8452
#     epoch 2    0.8584 - 0.8729        epoch 5    0.8054 - 0.8158
#
# still falling at epoch 5, so where a gate lands the switch depends on a trajectory we
# have only seen the start of. Hence the values: 0.95 switches as soon as the degenerate
# first curve is gone, 0.78 sits below the simulated curve's own median of 0.767 and may
# never switch at all, and the two between should bracket the interesting range.
#
# `measured_min_recovery` is deliberately NOT swept. It reads rec.max(), the top of the
# curve at t ~ 1 where the field is already nearly clean, which is high for any model from
# epoch 1 — in the runs above the peak was 1.000 even for curves the median gate refused.
# It cannot separate these cases, and raising it would not help: the degenerate curve's
# problem is that its peak is high.
#
# 30 epochs, no sampling: this measures when the schedule switches and what it does after,
# not WER. Read tau/source_measured for the handover epoch, tau/t_q50 for the shape it
# switched into, and tau/mu_accuracy_ema for what the fallback was being rebuilt at.
#
#   ./jobs/sweeps/warmup_hps.sh                        # dry run
#   SUBMIT=1 ./jobs/sweeps/warmup_hps.sh               # submit
#   GATES="0.95 0.82" CADENCES="1" SUBMIT=1 ./jobs/... # trim

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

CONFIG="${CONFIG:-configs/audio_cond.yaml}"
PROJECT="${PROJECT:-flow-asr-warmup-hps}"
GATES="${GATES:-0.95 0.88 0.82 0.78}"
CADENCES="${CADENCES:-1 10}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-512}"
PROBE_N="${PROBE_N:-8192}"
SEEDS="${SEEDS:-0}"

# 30 epochs is ~10 cold at 130 s plus 20 warm at 43 s, about 36 min, plus startup. 1h30
# is ~2.3x that, which covers a contended node without sitting in the queue for it.
export WALLTIME="${WALLTIME:-01:30:00}"

cd "${REPO_DIR}"

n=0
for gate in ${GATES}; do
    for every in ${CADENCES}; do
        for seed in ${SEEDS}; do
            name="g${gate}-e${every}"
            [[ "${SEEDS}" == *" "* ]] && name="${name}-s${seed}"
            overrides=(
                "seed=${seed}"
                "data.subset_n.train=null"
                "data.batch_size=${BATCH}"
                "data.eval_batch_size=${EVAL_BATCH}"
                "train.num_epochs=${EPOCHS}"
                "train.denoiser_lr=0.002"
                "train.audio_decoder_lr=0.001"
                "schedule.refresh_curve_from=warmup"
                "schedule.refresh_every_n_epochs=${every}"
                "schedule.measured_max_median_t=${gate}"
                "schedule.probe_n_utts=${PROBE_N}"
                "sampling.enabled=false"
                "logging.project=${PROJECT}"
                "logging.group=gate-${gate}"
                "logging.name=${name}"
            )
            n=$((n + 1))
            echo "=== [${n}] ${name}"
            if [[ -n "${SUBMIT}" ]]; then
                ./jobs/submit/train.sh "${CONFIG}" "${overrides[@]}"
            else
                echo "    ./jobs/submit/train.sh ${CONFIG} ${overrides[*]}"
            fi
        done
    done
done

echo
if [[ -n "${SUBMIT}" ]]; then
    echo "submitted ${n} jobs to ${PROJECT}"
else
    echo "${n} jobs NOT submitted (dry run). SUBMIT=1 to send them."
    echo "~45 min each, so ~$(awk -v n=${n} 'BEGIN{printf "%.0f", n*0.75}') GPU-hours."
fi
