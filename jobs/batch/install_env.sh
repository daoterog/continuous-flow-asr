#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=ins_env
#SBATCH --ntasks=1
# One whole MIG slice (9 cores, 60 GiB): half an A100 share's price, and a GPU is all the
# install needs, so conda resolves the CUDA build of torch rather than a CPU one.
#SBATCH --cpus-per-task=9
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/continuous-flow-asr"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f conda_environment.yaml