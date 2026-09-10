#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$1"

CACHE_ROOT="$(dirname "${VENV_DIR}")"
WEIGHTS_DIR="${CACHE_ROOT}/weights"
TMP_DIR="${WEIGHTS_DIR}/tmp"
PIP_CACHE_DIR="${WEIGHTS_DIR}/pip"
SAM2_CACHE="${WEIGHTS_DIR}/sam2"
SAM2_CFG_CACHE="${SAM2_CACHE}/configs"
SAM2_CKPT_CACHE="${SAM2_CACHE}/checkpoints"
mkdir -p "${TMP_DIR}" "${PIP_CACHE_DIR}" "${SAM2_CFG_CACHE}" "${SAM2_CKPT_CACHE}"
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
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}" --no-cache-dir
pip install transformers pillow opencv-python numpy pyyaml hydra-core omegaconf iopath --no-cache-dir

# Install sam2 without deps so it reuses the already installed torch stack.
pip install --no-deps --no-build-isolation --no-cache-dir git+https://github.com/facebookresearch/sam2.git


SAM2_PKG_DIR="$("${VENV_DIR}/bin/python" - <<'PY'
from pathlib import Path
import sam2

print(Path(sam2.__file__).resolve().parent)
PY
)"

SRC="${SAM2_PKG_DIR}/configs"
DST="${SAM2_CACHE}/configs"
if [ -d "${SRC}" ] && [ ! -L "${SRC}" ]; then
  mkdir -p "${DST}"
  cp -a "${SRC}/." "${DST}/"
  rm -rf "${SRC}"
fi
ln -sfn "${DST}" "${SRC}"
