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
TORCH_VERSION="${VT_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${VT_TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${VT_TORCHAUDIO_VERSION:-2.7.1}"
TORCH_INDEX_URL="${VT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}" \
  --no-cache-dir
# MiniCPM-V remote code currently expects the pre-v5 transformers init contract.
pip install "transformers<5" accelerate pillow pyyaml sentencepiece --no-cache-dir

pip install opencv-python-headless==4.10.0.84
