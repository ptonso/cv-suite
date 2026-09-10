#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$1"
CACHE_ROOT="$(dirname "${VENV_DIR}")"
WEIGHTS_DIR="${CACHE_ROOT}/weights"
TMP_DIR="${WEIGHTS_DIR}/tmp"
PIP_CACHE_DIR="${WEIGHTS_DIR}/pip"
PKG_ROOT="${CACHE_ROOT}/pkgs"

mkdir -p "${TMP_DIR}" "${PIP_CACHE_DIR}" "${PKG_ROOT}"
export TMPDIR="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TMP="${TMP_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install -U pip wheel "setuptools<81" --no-cache-dir

DIM_DIR="${PKG_ROOT}/DIM"
if [ ! -d "${DIM_DIR}" ]; then
  git clone --depth 1 https://github.com/showlab/DIM.git "${DIM_DIR}"
fi

TORCH_INDEX_URL="${VT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${VT_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${VT_TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${VT_TORCHAUDIO_VERSION:-2.7.1}"

pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}" \
  --no-cache-dir

# Install build dependencies
pip install \
  "setuptools>=65.0,<81" \
  "wheel" \
  "cmake" \
  --no-cache-dir

pip install \
  "transformers==4.51.3" \
  "datasets==3.5.0" \
  "diffusers" \
  "open_clip_torch" \
  "pyrallis" \
  "omegaconf" \
  "termcolor" \
  "imageio" \
  "matplotlib" \
  "httpx==0.23.2" \
  --no-cache-dir

# Additional diffusers / accelerate stack
pip install \
  "accelerate>=1.6,<2" \
  "huggingface-hub>=0.31,<1" \
  "safetensors>=0.5,<1" \
  "sentencepiece>=0.2,<1" \
  "einops>=0.8,<1" \
  "bitsandbytes>=0.43,<1" \
  "numpy<2.1" \
  pillow \
  pyyaml \
  --no-cache-dir

pip install opencv-python-headless==4.10.0.84 --no-cache-dir
