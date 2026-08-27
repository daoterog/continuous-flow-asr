#!/usr/bin/env bash
set -euo pipefail

# Mirror the uv-managed .venv into a conda environment YAML for Snellius.
#
#   ./uv_to_conda.sh [REPO_DIR]      ENV_NAME overrides the conda env name.
#
# Targets CUDA GPU nodes only: torch must resolve to a CUDA build, and the
# generated YAML pins torch/torchcodec with their local version tags so pip
# cannot substitute a PyPI wheel built against a different CUDA.
REPO_DIR="${1:-$(pwd)}"
ENV_NAME="${ENV_NAME:-flow_asr}"

# Prefer using the env's python directly (avoid relying on activation)
PY_BIN="$REPO_DIR/.venv/bin/python"
ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [ ! -x "$PY_BIN" ]; then
  if [ -f "$ACTIVATE" ]; then
    # shellcheck source=/dev/null
    echo "source $ACTIVATE"
    source "$ACTIVATE" || {
      echo "Error: Could not activate virtual environment."
      exit 1
    }
    PY_BIN="$(which python)"
  else
    echo "Error: Python not found in virtualenv and activate script missing."
    exit 1
  fi
fi

cd "$REPO_DIR" || exit

echo "uv sync"
uv sync

PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')

# Read the versions uv actually resolved, so the conda env mirrors the .venv
# instead of a stale pin.  "2.13.0+cu130" -> version 2.13.0, tag cu130.
TORCH_VERSION_FULL=$("$PY_BIN" -c 'import torch; print(torch.__version__)')
CUDA_TAG="${TORCH_VERSION_FULL#*+}"
if [ "$CUDA_TAG" = "$TORCH_VERSION_FULL" ]; then
  echo "Error: torch $TORCH_VERSION_FULL carries no CUDA local version tag." >&2
  echo "       This env targets CUDA GPU nodes; check [tool.uv.sources] in" >&2
  echo "       pyproject.toml points torch at a pytorch CUDA index." >&2
  exit 1
fi
echo "Detected torch=$TORCH_VERSION_FULL (index tag: $CUDA_TAG)"

# torchcodec is ABI-locked to both torch and the CUDA major version: a mismatch
# surfaces as an undefined c10::cuda symbol or a missing libnvrtc.so.<major>.
# It must therefore come from the same index as torch, pinned with its local
# version tag — PyPI's build targets a different CUDA and will not load.
TORCHCODEC_VERSION_FULL=$("$PY_BIN" -c 'import torchcodec; print(torchcodec.__version__)')
echo "Detected torchcodec=$TORCHCODEC_VERSION_FULL"

# torch/torchcodec are pinned explicitly below; the nvidia-*/triton wheels come
# in as torch's own dependencies and pinning them causes solver conflicts.
PIP_FREEZE=$(uv pip list --format freeze \
  | grep -vE '^(torch|torchcodec)==' \
  | grep -vE '^(nvidia-|cuda-toolkit==|cuda-bindings==|cuda-pathfinder==|triton==)')

CONDA_YAML_PATH="$REPO_DIR/conda_environment.yaml"

{
  echo "name: $ENV_NAME"
  echo "channels:"
  echo "  - conda-forge"
  echo "  - defaults"
  echo "dependencies:"
  echo "  - python=$PY_VER"
  echo "  - pip"
  # torchcodec decodes audio through FFmpeg (any of 4-9) and will not import
  # without those shared libraries present.
  echo "  - ffmpeg"
  echo "  - pip:"
  echo "    - \"--index-url=https://download.pytorch.org/whl/$CUDA_TAG\""
  echo '    - "--extra-index-url=https://pypi.org/simple"'
  # Both pinned with their local CUDA tag (e.g. 2.13.0+cu130) rather than the
  # bare version: without it pip may satisfy the pin from PyPI, whose build
  # targets a different CUDA and fails to load at import time.
  echo "    - \"torch==$TORCH_VERSION_FULL\""
  echo "    - \"torchcodec==$TORCHCODEC_VERSION_FULL\""

  if [ -z "$PIP_FREEZE" ]; then
    echo "    # no additional pip packages detected"
  else
    while IFS= read -r pkg; do
      [ -z "$pkg" ] && continue
      printf '    - "%s"\n' "$(printf '%s' "$pkg" | sed 's/"/\\"/g')"
    done <<< "$PIP_FREEZE"
  fi
} > "$CONDA_YAML_PATH"

echo "Generated conda environment YAML at: $CONDA_YAML_PATH"

# Deactivate current uv env
echo "deactivate"
deactivate || true