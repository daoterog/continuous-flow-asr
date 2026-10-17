#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=sample_trajectory
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
# 120G with one GPU and 18 cores keeps the job billed as one GPU (jobs/common/budget.sh).
#SBATCH --mem=120G
# Sized from FLOPs, not measured: the default models cost ~1,500 network calls per
# utterance per split (1+2+...+64 steps per weight, guided weights twice over), ~5e17 FLOPs
# over val and test — about 1.5 h on an A100, so 3 h is room, at most ~384 SBU.
#SBATCH --time=03:00:00
#SBATCH --output=slurm_sample_trajectory_%j.out

# The paper's sampling figure (scripts/evaluation/sample_trajectory.py): final WER at each
# step count, WER along one 16-step trajectory, and the transcripts it passes, for every
# model in MODELS. Guidance follows each run's own training: unguided and guided for a run
# trained with audio dropout, unguided only otherwise.
#
#   sbatch jobs/batch/sample_trajectory.sh
#   sbatch --export=ALL,MODELS="ours:runs/<id>/components" jobs/batch/sample_trajectory.sh
#
# Optional env vars:
#   MODELS          — space-separated label:components_dir[@w,...]
#                     (default: the four families of the WER table, plus Ours trained with
#                     audio dropout 0.05)
#   TRAJ_ROLES      — splits to score                  (default "val test")
#   TRAJ_STEPS      — step counts                      (default "1 2 4 8 16 32 64")
#   TRAJ_CFG        — guided weights for runs trained with audio dropout (default "1.5 2")
#   TRAJ_OUT        — where the CSVs land              (default paper/data)
#   TRAJ_SHOW       — utterances whose every step is written (default 25); the
#                     examples table picks the one that forms most visibly among them
#   TRAJ_OVERRIDES  — dotlist overrides on top of each run's own config. The default
#                     forces the whole split, as sample_eval.sh does.

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

MODELS="${MODELS:-ours:runs/y9hyu9p0/components \
linear:runs/16keaex7/components \
linear-adrop0.1:runs/1nz9v8pz/components@2 \
ours-adrop0.05:runs/6n9h2pko/components@1.5 \
mu-only:runs/txuowwbe/components}"
TRAJ_OVERRIDES="${TRAJ_OVERRIDES:-data.subset_n.val=null data.subset_n.test=null}"

echo "models    : ${MODELS}"
echo "roles     : ${TRAJ_ROLES:-val test}   steps ${TRAJ_STEPS:-1 2 4 8 16 32 64}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# data.env=snellius because the features are on scratch, not in the repo's data/.
srun python -m scripts.evaluation.sample_trajectory \
    --model ${MODELS} \
    --role ${TRAJ_ROLES:-val test} \
    --steps ${TRAJ_STEPS:-1 2 4 8 16 32 64} \
    --cfg-strengths ${TRAJ_CFG:-1.5 2} \
    --out "${TRAJ_OUT:-paper/data}" \
    --n-show "${TRAJ_SHOW:-25}" \
    data.env=snellius ${TRAJ_OVERRIDES}
