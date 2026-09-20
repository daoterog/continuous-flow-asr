# Shared environment setup for every batch job. Source it, do not execute it:
#
#   source "$HOME/continuous-flow-asr/jobs/batch/_env.sh"
#
# An absolute path, not one relative to $BASH_SOURCE: SLURM may copy the batch script to a
# spool directory on the compute node, so a relative source would resolve to nothing.
#
# Override the repo location with REPO_HOME and the conda env with CONDA_ENV.

REPO_HOME="${REPO_HOME:-$HOME/continuous-flow-asr}"

module purge
module load 2025

cd "${REPO_HOME}"

# Put the env's interpreter on PATH directly instead of activating it. Both
# `source activate flow_asr` and `conda activate flow_asr` silently no-op in a batch shell
# and leave `python` pointing at the Anaconda3 module's interpreter, which has no torch —
# the module's conda does not resolve env names under ~/.conda/envs. The Anaconda3 module
# is not loaded at all now, since nothing here needs it.
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/.conda/envs/${CONDA_ENV:-flow_asr}}"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"

# Pip wheels such as torchcodec carry no RPATH into the env, so libtorchcodec would
# resolve libstdc++ against Snellius's /lib64 — too old for the GLIBCXX that conda's own
# libicuuc needs.
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# HF_TOKEN for authenticated Hub requests (higher rate limits on model/dataset pulls).
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "WARNING: no .env found; Hub requests will be unauthenticated" >&2
fi

# Last, so it sees the final MODEL: .env is sourced under `set -a`, which exports every
# name it defines, so a MODEL left in there would otherwise slip past the guard.
# Defines PINNED_MODEL, resolves MODEL, and provides require_pinned_model — which the
# jobs that load a checkpoint call for themselves; the download jobs do not.
# shellcheck disable=SC1091
source "${REPO_HOME}/jobs/pinned_model.sh"
