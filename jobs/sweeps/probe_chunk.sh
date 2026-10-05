#!/bin/bash

# Does a larger eval batch make the measured probe faster on an A100?
#
# On a 3060 it does not: one 2048-utterance probe at the 512x8 denoiser is flat from chunk
# 32 to 128 (5.12 / 5.08 / 5.16 s) and slower above, because the
# (B, canvas_len, vocab_size) field already saturates the card. An A100 has far more to
# saturate and 40 GiB to do it in, so the answer may differ — and it decides whether a
# frequent refresh is cheap or is the thing that makes `meas-e1` lose.
#
# Short on purpose: 6 epochs with a refresh every epoch is 5 probes, which is plenty to
# time, and the WER these produce is meaningless. Read the walltime, not the result.
#
# Usage:
#   ./jobs/sweeps/probe_chunk.sh                  # dry run
#   SUBMIT=1 ./jobs/sweeps/probe_chunk.sh         # submit
#
# Then compare the per-epoch heartbeat spacing across the runs, or the total walltime:
#   sacct -j <id> --format=JobID,JobName%22,Elapsed,State

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

CONFIG="${CONFIG:-configs/audio_cond.yaml}"
PROJECT="${PROJECT:-flow-asr-probe-chunk}"
CHUNKS="${CHUNKS:-32 128 256 512 1024}"
BATCH="${BATCH:-128}"
EPOCHS="${EPOCHS:-6}"
PROBE_N="${PROBE_N:-2048}"

# 6 epochs on full data is ~1,332 steps; the floor in submit/train.sh covers it, but the
# probe is the point of the job so leave room for the slowest chunk.
export STEPS_PER_SEC="${STEPS_PER_SEC:-4}"

cd "${REPO_DIR}"

n=0
for chunk in ${CHUNKS}; do
    overrides=(
        "seed=0"
        "data.subset_n.train=null"
        "data.batch_size=${BATCH}"
        "data.eval_batch_size=${chunk}"
        "train.num_epochs=${EPOCHS}"
        "train.denoiser_lr=0.002"
        "train.audio_decoder_lr=0.001"
        "schedule.refresh_curve_from=measured"
        "schedule.refresh_every_n_epochs=1"
        "schedule.probe_n_utts=${PROBE_N}"
        # The model is nowhere near recovering anything in 6 epochs, and a skipped refresh
        # still runs the whole sweep — which is what is being timed. Zero so the fit and
        # the inversion run too, not just the forward passes.
        "schedule.measured_min_recovery=0.0"
        "sampling.enabled=false"
        "logging.project=${PROJECT}"
        "logging.group=probe-chunk"
        "logging.name=chunk-${chunk}"
    )
    n=$((n + 1))
    echo "=== [${n}] chunk-${chunk}"
    if [[ -n "${SUBMIT}" ]]; then
        ./jobs/submit/train.sh "${CONFIG}" "${overrides[@]}"
    else
        echo "    ./jobs/submit/train.sh ${CONFIG} ${overrides[*]}"
    fi
done

echo
if [[ -n "${SUBMIT}" ]]; then
    echo "submitted ${n} jobs to ${PROJECT}"
else
    echo "${n} jobs NOT submitted (dry run). SUBMIT=1 to send them."
fi
