#!/bin/bash

# Does flow-map distillation close the one-step gap, and is FMLM's batch split the right one
# for our regime?
#
# The teacher integrates well and answers badly in one shot. On the full val split its
# interpolant ladder is 39.05 / 16.26 / 11.49 / 10.32 WER at 1 / 2 / 4 / 16 steps, and the
# student starts *as* the teacher, so that is also its starting ladder. The claim to look for
# is `sampled_interpolant_1/wer` falling from ~39 toward ~10, with `sampled_interpolant_16/wer`
# holding near 10.3 as the regression guard.
#
#   Read 39.05, not the 51.75 in older notes. The probe scores the FIRST n_utts of the val
#   split while data.subset_n draws a RANDOM subsample, so a run with subset_n.val=64 scores a
#   different 64 utterances. These arms run on the whole split.
#
# The dense every-step read is `offdiag_perr/train_bins_gap`: the student's argmax against the
# true tokens, binned by how much of the decoding one jump covers. Unlike either loss term —
# the off-diagonal one scores the student against its own rollout, so it falls both when
# distillation works and when the student collapses to a constant map — it is anchored to
# truth. A working flow map flattens it; `_low_gap` rising toward `_high_gap` is collapse.
#
# The arms are the split, because the measured terms are badly out of balance. At init, on the
# training distribution:
#
#     input_perr     0.538     what the interpolant at s already carried
#     diagonal_perr  0.0085    the student is a 99.1% denoiser before training starts
#     boundary_perr  0.491     the one-step jump, which is the whole point
#
# FMLM spends half the batch on the diagonal and 1.6% on the boundary. Those are their
# numbers, not a guess, but they were set for a regime where the diagonal is not already
# solved. So:
#
#   control       0.5   / 0.016 / 0.484     FMLM's own, the reference
#   split-bndry   0.4   / 0.1   / 0.5       6x the boundary, the diagonal still held down
#   split-even    0.2   / 0.1   / 0.7       how far the diagonal can be cut before the
#                                           student drifts, which diagonal_perr reports
#
# Runs on the standard tiers (jobs/common/sweep.sh): a 100-epoch screen by default, two runs
# per GPU; TIER=confirm for 300 epochs x 2 seeds. Epochs come from the tier's --run profile,
# so nothing here pins train.num_epochs.
#
# SEC_PER_EPOCH must be set explicitly to submit — see the guard below.
#
# Usage:
#   ./jobs/sweeps/distill.sh                                  # dry run: plan + SBU
#   SEC_PER_EPOCH=60 SUBMIT=1 ./jobs/sweeps/distill.sh        # screen, gated
#   TIER=confirm ARMS="control split-bndry" SEC_PER_EPOCH=60 SUBMIT=1 ./jobs/sweeps/distill.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# Read before sourcing: budget.sh defaults SEC_PER_EPOCH, after which there is no way to
# tell a measured value from the default. The guard below needs that distinction.
SEC_PER_EPOCH_GIVEN="${SEC_PER_EPOCH:-}"
# shellcheck source=../common/sweep.sh
source "${REPO_DIR}/jobs/common/sweep.sh"
tier_defaults

# The recipe rather than configs/flow_mapping.yaml, which train.build_config refuses on the
# cluster; the recipe adds nothing but its own name.
CONFIG="${CONFIG:-configs/recipes/distill.yaml}"
PROJECT="${PROJECT:-flow-asr-distill}"
ARMS="${ARMS:-control split-bndry split-even}"
TEACHER="${TEACHER:-runs/q5bsvywz/components}"

cd "${REPO_DIR}"

if [[ ! -d "${TEACHER}" ]]; then
    echo "ERROR: no teacher at ${TEACHER}. Fetch it with" >&2
    echo "  python -m scripts.checkpointing.fetch_components $(basename "$(dirname "${TEACHER}")")" >&2
    exit 1
fi

# Read off the teacher's manifest rather than hardcoded, so a different TEACHER brings its
# own. grep rather than python: this runs on a login node, where the venv may not be active,
# and `audio_stride` appears exactly once in a manifest.
TEACHER_STRIDE="${TEACHER_STRIDE:-$(
    grep -m1 -oP 'audio_stride:\s*\K[0-9]+' "${TEACHER}/manifest.yaml"
)}"
if [[ -z "${TEACHER_STRIDE}" ]]; then
    echo "ERROR: no data.audio_stride in ${TEACHER}/manifest.yaml; pass TEACHER_STRIDE=" >&2
    exit 1
fi

# No cost model describes a distillation epoch. SEC_PER_EPOCH defaults to 25, measured for
# audio_cond at the profile's batch WITH compile and bf16; distillation runs compile off
# (map_step indexes by group with nonzero, a data-dependent shape) and does more work per
# step — half the batch also runs the frozen teacher, the rest a two-step rollout. A local
# measurement put the step at 1.64x, but that was compile-off on both sides, so it does not
# describe the ratio against the compiled baseline the default came from.
#
# Walltime is 2x the estimate, so under-pricing by more than 2x returns a truncated run that
# reads as a result — and OneCycleLR is sized for the full schedule, so a job cut early stops
# with the LR nowhere near its intended final value. Hence: measure, then submit.
#
#   python -m scripts.diagnostics.bench_train --suite distill --env snellius
#
# (or ./jobs/submit/bench_train.sh, which runs it on the node). Take the s/epoch of the
# `distill` row and pass it as SEC_PER_EPOCH.
if [[ -n "${SUBMIT}" && -z "${SEC_PER_EPOCH_GIVEN}" ]]; then
    cat >&2 <<'MSG'
ERROR: SEC_PER_EPOCH is still the audio_cond default (25 s), which does not describe a
       distillation epoch — compile is off here and the step does more work. Pricing and
       walltime would both be wrong, and a truncated run reads as a result.

       Measure it first:
           python -m scripts.diagnostics.bench_train --suite distill --env snellius
       then re-run this with the measured value:
           SEC_PER_EPOCH=<s/epoch> SUBMIT=1 ./jobs/sweeps/distill.sh

       The dry run (no SUBMIT) works without it; its SBU figure is a placeholder.
MSG
    exit 1
fi

# The whole split, the teacher to distil from, and the student's own rate. Everything that
# depends on the machine or the run kind — batch, eval batch, epochs, eval cadence, the
# sampling ladder — comes from configs/profiles/ and configs/flow_mapping.yaml.
COMMON=(
    "data.subset_n.train=null"
    "data.subset_n.val=null"
    "distill.teacher=${TEACHER}"
    # The teacher's stride, not the profile's. Every teacher so far was trained at stride 4
    # and the snellius profile now sets 1 — and the two differ by 4x in frames at the same
    # width (stride pools over frames), so nothing would raise: the teacher would simply be
    # fed 1,500 frames where it learned 375, and the student distilled against whatever came
    # out. `train.validate_distill_teacher` enforces the match, so changing TEACHER to a
    # stride-1 checkpoint will fail here until this is changed with it.
    "data.audio_stride=${TEACHER_STRIDE}"
    "train.two_time_denoiser_lr=0.00005"
    "logging.project=${PROJECT}"
    ${EPOCH_OVERRIDE}
)

# Three dotlist entries rather than one inline mapping: each fraction is visible in the dry
# run, and `train.validate_distill_split` sees all three whichever way they arrive.
arm_overrides() {
    case "$1" in
        control)     echo "" ;;
        split-bndry) echo "distill.split.diagonal=0.4 distill.split.boundary=0.1 distill.split.offdiag=0.5" ;;
        split-even)  echo "distill.split.diagonal=0.2 distill.split.boundary=0.1 distill.split.offdiag=0.7" ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
        queue_run "${arm}-s${seed}" "seed=${seed}" "${COMMON[@]}" \
            $(arm_overrides "${arm}") "logging.group=${TIER}${EPOCHS}/${arm}" \
            "logging.name=${arm}-e${EPOCHS}-s${seed}"
    done
done
flush_runs
