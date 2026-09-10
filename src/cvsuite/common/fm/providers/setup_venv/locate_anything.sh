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

NUMPY_SPEC="${VT_NUMPY_SPEC:-numpy==1.26.4}"
pip install "${NUMPY_SPEC}" --only-binary=numpy --no-cache-dir

pip install \
  "transformers==4.57.1" \
  "Pillow==11.1.0" \
  "opencv-python-headless==4.11.0.86" \
  "decord==0.6.0" \
  "lmdb==1.7.5" \
  accelerate \
  peft \
  huggingface-hub \
  safetensors \
  pyyaml \
  --no-cache-dir
