#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --job-name=sanity
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_sanity_%j.out

# Pre-flight for the Whisper precompute. Runs every step the real job does — env,
# decoder, data, a tiny encode, a read-back — on the cheapest GPU partition in ~10
# minutes, so a broken environment costs one MIG slice instead of a full A100 array.
#
#   sbatch jobs/sanity_check.sh
#   CONFIGS=clean SPLITS=validation sbatch jobs/sanity_check.sh
#
# Exits non-zero if any check fails; the summary lists exactly which.

# No -u: conda's activate script reads unset variables and would abort under it.
set -eo pipefail

CONFIGS="${CONFIGS:-clean}"
SPLITS="${SPLITS:-train.100}"
MODEL="${MODEL:-openai/whisper-large-v3}"

module purge
module load 2025
module load Anaconda3/2025.06-1

cd "$HOME/continuous-flow-asr"
source activate "${CONDA_ENV:-flow_asr}"

# Conda relies on RPATH and does not set LD_LIBRARY_PATH, but pip wheels such as
# torchcodec carry no RPATH into the env, so its libtorchcodec resolves libstdc++
# against Snellius's /lib64 — too old for the GLIBCXX_3.4.30 conda's libicuuc needs.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

FAILED=0
check() {
    local name="$1"; shift
    echo
    echo "── ${name} ──────────────────────────────────────────────"
    if "$@"; then
        echo "PASS: ${name}"
    else
        echo "FAIL: ${name}" >&2
        FAILED=1
    fi
}

check "GPU and torch" python - <<'EOF'
import torch
print("torch     ", torch.__version__)
print("cuda built", torch.version.cuda)
assert torch.cuda.is_available(), "no CUDA device visible"
p = torch.cuda.get_device_properties(0)
print("gpu       ", p.name, f"{p.total_memory/2**30:.1f} GiB")
torch.zeros(8, device="cuda").sum().item()   # prove the runtime actually works
EOF

# The failure that killed the first real submission: conda's ffmpeg present but
# unloadable, discovered only after 3 GB of weights had been fetched.
check "FFmpeg / torchcodec" python -m scripts.check_ffmpeg

check "LibriSpeech present" python -m scripts.download_librispeech \
    --env snellius --configs ${CONFIGS} --splits ${SPLITS} --verify

check "Whisper weights reachable" python - "$MODEL" <<'EOF'
import sys
from transformers import WhisperFeatureExtractor
fe = WhisperFeatureExtractor.from_pretrained(sys.argv[1])
print("feature extractor OK:", fe.feature_size, "mels,", fe.chunk_length, "s window")
EOF

# A real encode of 8 utterances, then a read-back of what it wrote. --limit output is
# never mistaken for a finished shard (is_complete compares against the parquet row
# count), so this cannot poison a later full run even if cleanup below is skipped.
check "end-to-end encode" python -m scripts.precompute_whisper \
    --env snellius --model "${MODEL}" \
    --configs ${CONFIGS} --splits ${SPLITS} \
    --limit 8 --batch-size 8 --overwrite

check "read back features" python - "$MODEL" "$CONFIGS" "$SPLITS" <<'EOF'
import json, sys
import numpy as np
from pathlib import Path
model, configs, splits = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
root = Path("/scratch-shared/cont-flow-asr/data/features") / model.split("/")[-1]
found = 0
for c in configs:
    for s in splits:
        for idx in sorted((root / c / s).glob("*.json")):
            meta = json.loads(idx.read_text())
            x = np.load(idx.with_suffix(".npy"), mmap_mode="r")
            u = meta["utts"][0]
            feat = x[u["offset"] : u["offset"] + u["length"]]
            assert feat.shape == (u["length"], meta["d_model"]), feat.shape
            assert np.isfinite(feat).all(), "non-finite values in features"
            assert x.shape[0] == sum(v["length"] for v in meta["utts"])
            print(f"{c}/{s}/{idx.stem}: {x.shape} {x.dtype}, "
                  f"{len(meta['utts'])} utts, first={feat.shape} '{u['text'][:40]}'")
            found += 1
assert found, "no feature files were written"
EOF

# Remove the 8-utterance stubs so the real run starts from a clean tree.
echo
echo "── cleanup ──────────────────────────────────────────────"
python - "$MODEL" "$CONFIGS" "$SPLITS" <<'EOF'
import sys
from pathlib import Path
model, configs, splits = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
root = Path("/scratch-shared/cont-flow-asr/data/features") / model.split("/")[-1]
for c in configs:
    for s in splits:
        d = root / c / s
        for p in sorted(d.glob("*.npy")) + sorted(d.glob("*.json")):
            p.unlink(); print("removed", p)
EOF

echo
echo "═════════════════════════════════════════════════════════"
if ((FAILED)); then
    echo "SANITY CHECK FAILED — fix the above before submitting the array job." >&2
    exit 1
fi
echo "SANITY CHECK PASSED — safe to run ./jobs/submit_whisper.sh"
