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

# Install CUDA torch
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

# Install diffusers stack
pip install \
  "accelerate>=1.6,<2" \
  "diffusers>=0.35,<1" \
  "huggingface-hub>=0.31,<1" \
  "transformers>=4.51,<5" \
  "safetensors>=0.5,<1" \
  "sentencepiece>=0.2,<1" \
  "einops>=0.8,<1" \
  "numpy<2.1" \
  pillow \
  pyyaml \
  --no-cache-dir
pip install opencv-python-headless==4.10.0.84 --no-cache-dir

# Install bitsandbytes for NF4 quantization
pip install "bitsandbytes>=0.46,<1" --no-cache-dir
