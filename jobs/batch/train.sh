#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=flow_train
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
# Sized so the features fit the page cache, not so the process fits. Full clean/train.100
# is 102 GiB of fp16 latents; at 64G it did not cache and every epoch re-read it off GPFS
# at ~219 utts/s against 657 utts/s warm — a 3x walltime penalty.
#
# The features are mmap'd, so their pages are charged to this job's cgroup along with the
# workers. Budget for all of it, not just the data: 102 GiB mapped + ~24 GiB for 16 worker
# processes + ~8 GiB for the main process, CUDA's host allocations and the non-reclaimable
# pinned staging buffers. 120G was 14 GiB short of that and the OOM killer took a worker
# mid-epoch ("killed by signal: Killed"), non-deterministically — clean mapped pages are
# reclaimable, so the job survives it sometimes and dies others.
#
# This only stretches to train.100. Untrimmed train.360 is 372 GiB and cannot be cached at
# any sane request; pre-pooling the features at audio_stride would cut every figure here
# by 4x and is the prerequisite for training on more than train.100.
#SBATCH --mem=160G
#SBATCH --time=01:00:00
# A placeholder only. submit/train.sh overrides --time with a value derived from the
# config's own step count and a measured steps-per-second, so this is what you get if
# you sbatch this file directly — which you should not.
#SBATCH --output=slurm_output_%j.out
# BEGIN/END/FAIL are start, clean finish and crash. TIME_LIMIT_90 warns at 90% of the
# walltime, which is the one that matters here: a run cut off at 90 of 300 epochs looks
# "finished" in wandb because Lightning's SLURM handler exits cleanly on the signal.
# REQUEUE fires when that handler resubmits, which otherwise happens silently and starts
# a second wandb run under a new name.
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90,REQUEUE
#SBATCH --mail-user=daniel.otero.gomez@student.uva.nl

# One or more training runs sharing this job's GPU. Submit via submit/train.sh — do not
# run this script directly.
#
# Required env vars (set by submit/train.sh through the environment):
#   TRAIN_CONFIG        — configs/<name>.yaml
#   TRAIN_N             — how many runs (default 1)
#   TRAIN_RUN           — the run-kind profile, `sweep` or `full` (default full); the
#                         machine profile is always `--where snellius` here
#   TRAIN_OVERRIDES_<i> — run i's space-separated OmegaConf dotlist overrides, may be empty
#                         (TRAIN_OVERRIDES is still read for a single run, as before)
#
# Why several runs per job: a job is billed for the largest of its node shares (1 GPU,
# 18 cores or 120 GiB each), and the 160G below is already 2 shares — so a one-run job pays
# for 2 GPUs and uses 1. jobs/submit/train.sh asks for one GPU and 18 cores per run
# (--gpus, --cpus-per-task and --mem on the sbatch line override the defaults here), and
# each run below is pinned to its own GPU with the profile's full 16 workers. The runs mmap
# the same feature files, so the page cache holds them once: 102 GiB plus ~20 GiB per run.

set -eo pipefail

# shellcheck disable=SC1091
source "${REPO_HOME:-$HOME/continuous-flow-asr}/jobs/batch/_env.sh"

# wandb runs on the compute node, which has no terminal to log in from. It reads
# WANDB_API_KEY from the environment (_env.sh sources .env) or ~/.netrc, which is shared
# with the login node. Without either it would block on a prompt and burn the whole
# reservation, so fail here instead.
if [[ -z "${WANDB_API_KEY}" ]] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
    echo "ERROR: no wandb credentials. Put WANDB_API_KEY in .env, or run 'wandb login'" >&2
    echo "       on the login node so ~/.netrc is written." >&2
    exit 1
fi

TRAIN_N="${TRAIN_N:-1}"
TRAIN_OVERRIDES_0="${TRAIN_OVERRIDES_0-${TRAIN_OVERRIDES}}"

echo "config    : ${TRAIN_CONFIG}"
echo "runs      : ${TRAIN_N}"
echo "node      : $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Whether train.960's ~1.1 TB of latents could be staged to node-local disk instead of read
# off GPFS every epoch.
echo "local disk: $(df -h "${TMPDIR:-/tmp}" | tail -1)"

# What depends on running here — the scratch data root, workers, eval batch, compile, the
# per-step metrics — comes from configs/profiles/where/snellius.yaml, not from overrides.
PROFILE=(--where snellius --run "${TRAIN_RUN:-full}")
echo "profile   : ${PROFILE[*]}"

# STAGE_DATA=1 (set by SCRATCH_NODE=1 in submit/train.sh): copy every feature directory the
# run reads onto this node's local NVMe and train from there. train.960's 1.1 TB does not fit
# in RAM, and GPFS gives one node 0.2-0.8 GB/s of random reads whatever the loader does
# (mmfsd at 718% CPU with 256 reads in flight, the GPU 0-10% busy). A bulk sequential copy is
# what GPFS streams best, and after it every read is local. The volume is
# /scratch-node/$USER.$SLURM_JOB_ID (1.8 TB for a 1-GPU H100 job, 2026-10-20) and is deleted
# when the job ends.
if [[ -n "${STAGE_DATA}" ]]; then
    STAGE_DIR="${STAGE_DIR:-/scratch-node/${USER}.${SLURM_JOB_ID}}"
    if [[ ! -d "${STAGE_DIR}" ]]; then
        echo "ERROR: STAGE_DATA needs node-local NVMe at ${STAGE_DIR}; submit with" >&2
        echo "       --constraint=scratch-node (SCRATCH_NODE=1 ./jobs/submit/train.sh ...)" >&2
        exit 1
    fi
    # shellcheck disable=SC2086
    mapfile -t LISTING < <(python train.py --config "${TRAIN_CONFIG}" "${PROFILE[@]}" \
        ${TRAIN_OVERRIDES_0} --print-data-dirs)
    SRC_ROOT="$(printf '%s\n' "${LISTING[@]}" | awk '$1 == "root" {print $2}')"
    mapfile -t SRC_DIRS < <(printf '%s\n' "${LISTING[@]}" | awk '$1 == "dir" {print $2}')
    if [[ -z "${SRC_ROOT}" || ${#SRC_DIRS[@]} -eq 0 ]]; then
        echo "ERROR: could not list the run's feature directories" >&2
        exit 1
    fi
    DST_ROOT="${STAGE_DIR}/data"

    # Summed file sizes, not du: du also counts directory entries, which differ in number.
    need=$(find "${SRC_DIRS[@]}" -maxdepth 1 -type f -printf "%s\n" | awk '{s += $1} END {print s}')
    have=$(df -B1 --output=avail "${STAGE_DIR}" | tail -1)
    echo "stage     : ${#SRC_DIRS[@]} directories, $((need / 2**30)) GiB -> ${DST_ROOT}" \
        "($((have / 2**30)) GiB free)"
    if ((need > have)); then
        echo "ERROR: the features do not fit on ${STAGE_DIR}" >&2
        exit 1
    fi

    t0=$(date +%s)
    for d in "${SRC_DIRS[@]}"; do
        mkdir -p "${DST_ROOT}/${d#"${SRC_ROOT}"/}"
    done
    # One cp per file, 16 at a time: shards are ~10 GB each, and parallel sequential
    # streams are what GPFS delivers fastest.
    for d in "${SRC_DIRS[@]}"; do
        find "${d}" -maxdepth 1 -type f -printf "%p\0${DST_ROOT}/${d#"${SRC_ROOT}"/}/\0"
    done | xargs -0 -n 2 -P 16 cp
    t1=$(date +%s)

    got=$(find "${DST_ROOT}" -type f -printf "%s\n" | awk '{s += $1} END {print s}')
    if ((got != need)); then
        echo "ERROR: staged ${got} bytes, expected ${need}" >&2
        exit 1
    fi
    secs=$((t1 - t0 > 0 ? t1 - t0 : 1))
    echo "stage     : copied in ${secs}s, $(awk -v b="${need}" -v s="${secs}" \
        'BEGIN {printf "%.2f", b / s / 1e9}') GB/s"

    # Every run in this job reads the staged copy.
    for ((i = 0; i < ${TRAIN_N:-1}; i++)); do
        var="TRAIN_OVERRIDES_${i}"
        printf -v "${var}" '%s' "${!var} data.data_root=${DST_ROOT}"
    done
fi

if [[ "${TRAIN_N}" -eq 1 ]]; then
    echo "overrides : ${TRAIN_OVERRIDES_0}"
    srun python train.py --config "${TRAIN_CONFIG}" "${PROFILE[@]}" ${TRAIN_OVERRIDES_0}
    exit $?
fi

# Each run logs to its own file next to this job's .out; this one only says which is which.
# Run i gets the i-th GPU Slurm handed this job — the i-th entry of CUDA_VISIBLE_DEVICES,
# which need not be 0..N-1.
IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TRAIN_N - 1)))}"
if ((${#GPU_IDS[@]} < TRAIN_N)); then
    echo "ERROR: ${TRAIN_N} runs but only ${#GPU_IDS[@]} GPU(s) visible (${CUDA_VISIBLE_DEVICES:-unset})" >&2
    exit 1
fi
PIDS=()
for ((i = 0; i < TRAIN_N; i++)); do
    var="TRAIN_OVERRIDES_${i}"
    LOG="slurm_output_${SLURM_JOB_ID:-local}_${i}.out"
    echo "run ${i}     : GPU ${GPU_IDS[$i]}  ${!var}  -> ${LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" \
        python train.py --config "${TRAIN_CONFIG}" "${PROFILE[@]}" ${!var} > "${LOG}" 2>&1 &
    PIDS+=($!)
done

failed=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "run ${i}     : finished"
    else
        echo "run ${i}     : FAILED (exit $?), see slurm_output_${SLURM_JOB_ID:-local}_${i}.out"
        failed=1
    fi
done
exit "${failed}"
