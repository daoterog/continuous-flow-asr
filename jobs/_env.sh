# Shared environment setup for every batch job. Source it, do not execute it:
#
#   source "$HOME/continuous-flow-asr/jobs/_env.sh"
#
# An absolute path, not one relative to $BASH_SOURCE: SLURM may copy the batch script to a
# spool directory on the compute node, so a relative source would resolve to nothing.
#
# Override the repo location with REPO_HOME and the conda env with CONDA_ENV.

REPO_HOME="${REPO_HOME:-$HOME/continuous-flow-asr}"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "${REPO_HOME}"

source activate "${CONDA_ENV:-flow_asr}"

# Conda relies on RPATH and does not set LD_LIBRARY_PATH, but pip wheels such as torchcodec
# carry no RPATH into the env, so libtorchcodec resolves libstdc++ against Snellius's
# /lib64 — too old for the GLIBCXX that conda's own libicuuc needs.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# HF_TOKEN for authenticated Hub requests (higher rate limits on model/dataset pulls).
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "WARNING: no .env found; Hub requests will be unauthenticated" >&2
fi
