#!/bin/bash

# Predates the budget standards (jobs/common/sweep.sh): one run per GPU job at its own
# epoch count. Each submit still prints its SBU, is gated against the cap and lands in
# the ledger, but new sweeps should use the screen/confirm tiers instead.
#
# Its CONFIG is a plain variant, so on Snellius train.py now refuses it (cluster runs must
# come from configs/recipes/). That is deliberate: these runs trained base.yaml's tiny
# denoiser at stride 4. To rerun one, point CONFIG at a recipe, or pass recipe=<name> to
# reproduce the old setting on purpose.

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

# 6 epochs on full data ran in ~11 min, every epoch paying the cold-cache rate because the
# job ends before the cache fills. 40 min is ~3.5x that, which is the right margin for a
# job this short — the estimator's own floor is 15 min and would be tight.
#
# Superseded by jobs/batch/bench_probe.sh, which times the probe directly instead of
# inferring it from whole training runs. Kept because it is the end-to-end check that the
# measured path survives a real run at each chunk, which the synthetic benchmark does not
# cover.
export WALLTIME="${WALLTIME:-00:40:00}"

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
