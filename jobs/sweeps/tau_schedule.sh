#!/bin/bash

# How much of the batch, and of the sampler, to spend away from the transition: two knobs,
# bracketed one at a time against one control on the cluster recipe
# (configs/recipes/fulldata.yaml), then combined.
#
# The knobs (scripts/schedule/project_tau.py draws every arm below before it is paid for):
#
#   floor<x>   schedule.uniform_floor=x     a share x of every batch drawn uniformly on
#                                           [0, 1], under the simulated curve's shape. The
#                                           control puts ~5% of a batch below t=0.5 and ~1%
#                                           above 0.85; floor 0.3 puts 19% and 5%.
#   sigma<x>   flow.alpha.scale=x           noise at x(1-t), so x_0 ~ N(mu, x^2 I). mu shows
#                                           through the noise and the field carries data
#                                           earlier, so the curve moves left as mu learns —
#                                           the shift toward the audio, made by the path.
#
# And the regularisers, since every round-1 arm memorises train.100 at the transition
# (train loss ~0.001 against ~0.1 on val):
#
#   drop<x>    model.denoiser.dropout=x     the recipe's 0.1, raised
#   wd<x>      train.weight_decay=x         the recipe's 0.01, raised
#   mudrop<x>  model.audio_decoder.dropout  mu's dropout; for configs/recipes/mu_only.yaml,
#                                           where mu is the whole model
#   mulr<x>    train.audio_decoder_lr       mu's learning rate, likewise
#   mask       train.audio_mask=...         SpecAugment on the Whisper latents: 2 time and
#                                           2 channel masks per utterance, each up to 10% of
#                                           the axis (src/data/augment.py)
#   adrop<p>   train.audio_dropout          a share p of rows train with no audio (and a
#                                           uniform prior in mu's place), so they must be
#                                           solved from the field: text modelling
#   adroplin<p>                             the same at p * t: more often nearer the text
#   meas<k>    schedule.refresh_curve_from  measured tau, re-read every k epochs: the draw
#                                           follows the denoiser's own error rather than the
#                                           field's (scripts/diagnostics/probe_audio_info.py
#                                           puts it at t~0-0.6, where simulated draws ~5%)
#   measval<k>                              the same, read off val: on train the model
#                                           memorises, recovery saturates just above t=0,
#                                           and the curve collapses there (median t 0.006
#                                           by epoch 90 in the `final` round)
#   draft<p>   train.draft_corruption       a share p of real tokens in the field swapped
#                                           for random ones, the loss still on the truth:
#                                           the denoiser learns to correct the field
#   draftmu<p>                              the same, swapped for samples from mu: the
#                                           acoustically plausible errors a draft makes
#   draftmudec<p>                           mu's swaps at p * (1 - t): a draft is worse the
#                                           earlier it is
#   linear     flow.gamma=null + no mu      the linear interpolant recipe on this stack: no
#                                           audio decoder, unit noise, no draft-mu (which
#                                           needs a mu). Ours against it is "is mu needed?"
#   cfgs       sampling.cfg_strengths       score every sampling epoch, and the final
#                                           variants, at w = 1, 1.5, 2, 3 (classifier-free
#                                           guidance; pair with adrop<p> to train it)
#   notau      schedule.enabled=false       t drawn uniformly. For a sigma too small for the
#                                           simulated curve, which collapses to a step at
#                                           t = 0.5 once the field decodes deterministically
#   lvar<l>    flow.learned_noise           the source's sigma learned per position and token
#                                           off mu's trunk, held to its target by a KL of
#                                           weight l (0: none, the paper's collapse). Target
#                                           1 as in the paper; carries uniform_floor 0.01, so
#                                           the tau rebuilt at a shrunken sigma still starts
#                                           at t = 0
#   lvartarget<x>                           that target sigma_0; put it after lvar<l>
#
# Arms join knobs with `+` (sigma0.5+floor0.1); `control` is the recipe as is.
#
# Round 1, each knob alone (6 runs, 3 jobs):
#   control floor0.3 floor0.4 sigma0.7 sigma0.5 sigma0.4
# Learned source variance (the control is why-control-e100-s5, y9hyu9p0, at seed 5):
#   ROUND=lvar1 SEEDS=5 ARMS="lvar0.1 lvar0.1+lvartarget0.7 lvar0.1+lvartarget0.4 lvar0+lvartarget0.4" SUBMIT=1 ...
# Regularisation screen:
#   ROUND=reg1 ARMS="control drop0.2 mask mask+drop0.2 adrop0.2 adroplin0.4" SUBMIT=1 ...
# Round 2, the winners combined, e.g. with sigma 0.5 and floor 0.3 winning:
#   ROUND=round2 ARMS="control sigma0.5 sigma0.5+floor0.1 sigma0.5+floor0.3" SUBMIT=1 ...
#
# Every arm samples on its own grid and on the control's (sampling.grids): a floor or a
# sigma moves the sampler's steps as well as the training draw, and the `_reference` WER
# is what separates the two. Every arm also logs audio_only/val_recovery[_no_mu|_wrong_mu]
# and val_copies_wrong_mu, the denoiser's recovery at t=0 under mu, no mu, and another
# utterance's mu: whether the knob made the model read the audio, not just whether WER
# moved.
#
# Read each arm against the control at epoch 100 (sampled WER, 16 steps, full val):
# >= 2 points better is a winner — seed noise on small + micro is ~1-2.
#
# KEEP adds adopted knobs to every arm, the control included (train.ema_decay=0.999 if
# stage 3 adopts the EMA).
#
# Usage:
#   ./jobs/sweeps/tau_schedule.sh                   # dry run: plan + SBU, submit nothing
#   SUBMIT=1 ./jobs/sweeps/tau_schedule.sh          # round 1, gated

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
# shellcheck source=../common/sweep.sh
source "${REPO_DIR}/jobs/common/sweep.sh"
tier_defaults

CONFIG="${CONFIG:-configs/recipes/fulldata.yaml}"
PROJECT="${PROJECT:-flow-asr-tau-schedule}"
ARMS="${ARMS:-control floor0.3 floor0.4 sigma0.7 sigma0.5 sigma0.4}"
ROUND="${ROUND:-round1}"
KEEP="${KEEP:-}"

COMMON=(
    "logging.project=${PROJECT}"
    "sampling.grids=[live,reference]"
    ${EPOCH_OVERRIDE}
)

# One knob's override. sigma rewrites the whole alpha node: base.yaml holds it as the
# string `linear_down`, which a dotted `flow.alpha.scale=` cannot be set under. JSON, so
# the override has no spaces and survives the batch script's word splitting.
knob_override() {
    case "$1" in
        control) echo "" ;;
        floor*)  echo "schedule.uniform_floor=${1#floor}" ;;
        sigma*)  echo "flow.alpha={\"type\":\"linear_down\",\"scale\":${1#sigma}}" ;;
        drop*)   echo "model.denoiser.dropout=${1#drop}" ;;
        mudrop*) echo "model.audio_decoder.dropout=${1#mudrop}" ;;
        mulr*)   echo "train.audio_decoder_lr=${1#mulr}" ;;
        wd*)     echo "train.weight_decay=${1#wd}" ;;
        measval*) echo "schedule.refresh_curve_from=measured schedule.refresh_every_n_epochs=${1#measval} schedule.refresh_split=val" ;;
        meas*)   echo "schedule.refresh_curve_from=measured schedule.refresh_every_n_epochs=${1#meas}" ;;
        notau)   echo "schedule.enabled=false" ;;
        linear)  echo "flow.gamma=null model.audio_decoder=null flow.alpha=linear_down train.draft_corruption={\"p\":0}" ;;
        cfgs)    echo "sampling.cfg_strengths=[1,1.5,2,3]" ;;
        draftmudec*) echo "train.draft_corruption={\"p\":${1#draftmudec},\"source\":\"mu\",\"schedule\":\"decreasing\"}" ;;
        draftmu*) echo "train.draft_corruption={\"p\":${1#draftmu},\"source\":\"mu\"}" ;;
        draft*)  echo "train.draft_corruption={\"p\":${1#draft},\"source\":\"uniform\"}" ;;
        adroplin*) echo "train.audio_dropout={\"p\":${1#adroplin},\"schedule\":\"linear\"}" ;;
        adrop*)  echo "train.audio_dropout={\"p\":${1#adrop},\"schedule\":\"constant\"}" ;;
        lvartarget*) echo "flow.learned_noise.target_scale=${1#lvartarget}" ;;
        lvar*)   echo "flow.learned_noise={\"kl_weight\":${1#lvar},\"target_scale\":1.0,\"min_scale\":0.05} schedule.uniform_floor=0.01" ;;
        mask)    echo "train.audio_mask={\"time_masks\":2,\"time_width\":0.1,\"feature_masks\":2,\"feature_width\":0.1}" ;;
        *) echo "unknown knob: $1" >&2; exit 1 ;;
    esac
}

arm_overrides() {
    local knob
    for knob in ${1//+/ }; do
        knob_override "${knob}"
    done
}

cd "${REPO_DIR}"

stack="${ROUND}"
[[ -n "${KEEP}" ]] && stack="${stack}+$(echo "${KEEP}" | tr ' =' '+-' | sed 's/train\.//g')"

for arm in ${ARMS}; do
    arm_overrides "${arm}" >/dev/null  # fail on an unknown knob before anything is queued
    for seed in ${SEEDS}; do
        queue_run "${stack}/${arm}-s${seed}" "seed=${seed}" "${COMMON[@]}" ${KEEP} \
            $(arm_overrides "${arm}") "logging.group=${TIER}${EPOCHS}/${stack}/${arm}" \
            "logging.name=${stack}-${arm}-e${EPOCHS}-s${seed}"
    done
done
flush_runs
