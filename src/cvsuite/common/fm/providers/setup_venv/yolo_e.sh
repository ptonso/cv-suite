#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$1"

CACHE_ROOT="$(dirname "${VENV_DIR}")"
WEIGHTS_DIR="${CACHE_ROOT}/weights"
TMP_DIR="${WEIGHTS_DIR}/tmp"
PIP_CACHE_DIR="${WEIGHTS_DIR}/pip"
mkdir -p "${TMP_DIR}" "${PIP_CACHE_DIR}"
export TMPDIR="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TMP="${TMP_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"
pip install -U pip wheel setuptools --no-cache-dir
# Install CPU torch by default, but allow explicit CUDA requests to override the wheel index.
TORCH_INDEX_URL="${VT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
pip install torch torchvision --index-url "${TORCH_INDEX_URL}" --no-cache-dir
pip install ultralytics huggingface_hub numpy pillow opencv-python-headless --no-cache-dir
