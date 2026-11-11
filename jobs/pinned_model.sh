# The checkpoint every job runs, and the guard that keeps it that way. Source it, do not
# execute it. Sourcing also resolves MODEL, so it is the only place the string lives.
#
# Features live under features/<model>/... and manifests are named MODEL_<slug>_DATASET_...,
# so two checkpoints' artifacts coexist instead of overwriting each other. That is what
# makes a mix easy to miss: nothing fails, nothing is overwritten, and the numbers look
# ordinary — you just end up with two half-populated result sets that average into
# something meaningless. MODEL is ordinary inherited environment (every submit script
# reads ${MODEL:-...} and both sbatch calls use --export=ALL), so one stale `export` is
# all it takes.
#
# Every script that loads a checkpoint calls require_pinned_model after resolving MODEL.
#
# To run a different one deliberately, say so and keep its output separate:
#   ALLOW_MODEL=1 MODEL=openai/whisper-small RESULTS_DIR=/scratch-shared/... \
#       ./jobs/submit/eval.sh

PINNED_MODEL="openai/whisper-large-v3"
# ${MODEL-...}, not ${MODEL:-...}: only an *unset* MODEL takes the default. An explicitly
# empty one is a mistake (a misspelled variable upstream, an --export that lost its value)
# and is reported by require_pinned_model rather than quietly coerced to the pin.
MODEL="${MODEL-${PINNED_MODEL}}"

# Aborts the sourcing script unless MODEL is the pinned checkpoint or ALLOW_MODEL is set.
require_pinned_model() {
    if [[ -z "${MODEL}" ]]; then
        # Reachable when a caller passes MODEL="" explicitly: argparse accepts an empty
        # --model, and score_results treats an empty model_id as "do not filter", which
        # pools every checkpoint in the results directory into one WER.
        echo "ERROR: MODEL is empty. Unset it to get ${PINNED_MODEL}, or set it to a" \
             "real checkpoint." >&2
        exit 1
    fi
    if [[ "${MODEL}" != "${PINNED_MODEL}" && -z "${ALLOW_MODEL}" ]]; then
        echo "ERROR: MODEL is '${MODEL}', not the pinned '${PINNED_MODEL}'." >&2
        echo "       Results from two checkpoints in one directory do not combine into" >&2
        echo "       anything meaningful. If this is deliberate, re-run with ALLOW_MODEL=1" >&2
        echo "       and a separate RESULTS_DIR. If it is not, check for an exported" >&2
        echo "       MODEL in your shell (env | grep '^MODEL=')." >&2
        exit 1
    fi
}
