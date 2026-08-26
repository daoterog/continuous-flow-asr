#!/usr/bin/env bash
set -euo pipefail

# Repository dir
REPO_DIR="${1:-$(pwd)}"

# Prefer using the env's python/pip directly (avoid relying on activation)
PY_BIN="$REPO_DIR/.venv/bin/python"
PIP_BIN="$REPO_DIR/.venv/bin/pip"
ACTIVATE="$REPO_DIR/.venv/bin/activate"

if [ ! -x "$PY_BIN" ] || [ ! -x "$PIP_BIN" ]; then
  # Try sourcing activate as a fallback (keeps original behaviour)
  if [ -f "$ACTIVATE" ]; then
    # shellcheck source=/dev/null
    echo "source $ACTIVATE"
    source "$ACTIVATE" || {
      echo "Error: Could not activate virtual environment."
      exit 1
    }
    PY_BIN="$(which python)"
  else
    echo "Error: Python/pip not found in virtualenv and activate script missing."
    exit 1
  fi
fi

# Pick the torch build. pyproject.toml locks a CUDA and a CPU resolution as two
# conflicting dependency groups; a lockfile cannot know about the hardware, so
# the choice is made here. Fall back to the CPU wheels when there is no usable
# GPU (login node, CI runner, laptop) — nvidia-smi must both exist and report a
# device, since the binary is present but fails on Snellius login nodes.
SYNC_ARGS=()
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
  echo "GPU detected, syncing the CUDA build"
else
  echo "No GPU detected, falling back to the CPU build"
  SYNC_ARGS=(--no-group cuda --group cpu)
fi

# Run sync env from $REPO_DIR
echo "cd $REPO_DIR && uv sync ${SYNC_ARGS[*]}"
cd "$REPO_DIR" || exit # '|| exit' ensures the script stops if cd fails
uv sync "${SYNC_ARGS[@]}"

# Determine Python major.minor version from the env
echo ""$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))'"
PY_VER=$("$PY_BIN" -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')

# Determine the torch version/CUDA build actually resolved by uv, so the
# conda env mirrors it instead of installing a different, stale pin.
TORCH_VERSION_FULL=$("$PY_BIN" -c 'import torch; print(torch.__version__)')
TORCH_VERSION="${TORCH_VERSION_FULL%%+*}"
TORCH_CUDA_TAG="${TORCH_VERSION_FULL#*+}"
if [ "$TORCH_CUDA_TAG" = "$TORCH_VERSION_FULL" ]; then
  TORCH_CUDA_TAG="cpu"
fi
echo "Detected torch=$TORCH_VERSION (index tag: $TORCH_CUDA_TAG)"

TORCHVISION_VERSION_FULL=$("$PY_BIN" -c 'import torchvision; print(torchvision.__version__)')
TORCHVISION_VERSION="${TORCHVISION_VERSION_FULL%%+*}"
echo "Detected ($TORCHVISION_VERSION_FULL) torchvision=$TORCHVISION_VERSION"

# Get pip freeze output (keeps VCS and editable lines)
echo "uv pip list --format freeze"
PIP_FREEZE=$(uv pip list --format freeze)

# Remove problematic packages — torch will pull in the correct
# nvidia-*/cuda versions itself, pinning them causes solver conflicts
PIP_FREEZE=$(echo "$PIP_FREEZE" | grep -v '^torch==' | grep -vE '^(nvidia-|cuda-toolkit==|cuda-bindings==|cuda-pathfinder==|triton==)')

CONDA_YAML_PATH="$REPO_DIR/conda_environment.yaml"

# Write conda YAML
{
  echo "name: env"
  echo "channels:"
  echo "  - conda-forge"
  echo "  - defaults"
  echo "dependencies:"
  echo "  - python=$PY_VER"
  echo "  - pip"
  echo "  - pip:"
  # ---- PyTorch (matches the build resolved by uv, in this case 2.9.0+cu128) ----
  echo "    - \"--index-url=https://download.pytorch.org/whl/$TORCH_CUDA_TAG\""
  echo '    - "--extra-index-url=https://pypi.org/simple"'
  echo "    - \"torch==$TORCH_VERSION\""
  echo "    - \"torchvision==$TORCHVISION_VERSION\""
  # ---- PyG ecosystem (same find-links page uv actually resolved against) ----
  echo "    - \"--find-links=https://data.pyg.org/whl/torch-$TORCH_VERSION_FULL.html\""
  echo '    - "pyg-lib"'
  echo '    - "torch-scatter"'
  echo '    - "torch-sparse"'
  echo '    - "torch-cluster"'
  # Remove torch-spline-conv from the conda env if using torch>2.9
  echo '    - "torch-spline-conv"'

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