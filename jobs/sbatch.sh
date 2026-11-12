#!/bin/bash

# sbatch, refusing a GPU job that would be billed for more GPUs than it asks for
# (jobs/common/slurm_guard.sh). Use it wherever you would type sbatch:
#
#   ./jobs/sbatch.sh --export=ALL,COMPONENTS=runs/<id>/components jobs/batch/sample_eval.sh
#   ./jobs/sbatch.sh --partition=gpu_mig --cpus-per-task=9 --mem=60G jobs/batch/sample_eval.sh

set -eo pipefail
# shellcheck source=common/slurm_guard.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common/slurm_guard.sh"
guarded_sbatch "$@"
