#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="$1"

CACHE_ROOT="$(dirname "${VENV_DIR}")"
WEIGHTS_DIR="${CACHE_ROOT}/weights"
TMP_DIR="${WEIGHTS_DIR}/tmp"
PIP_CACHE_DIR="${WEIGHTS_DIR}/pip"
CHECKPOINTS_DIR="${WEIGHTS_DIR}/checkpoints/paddle"
mkdir -p "${TMP_DIR}" "${PIP_CACHE_DIR}" "${CHECKPOINTS_DIR}"
export TMPDIR="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TMP="${TMP_DIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"
pip install -U pip wheel setuptools
pip install "numpy<2"

PY_TAG="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
PLATFORM_TAG="$("${VENV_DIR}/bin/python" -c 'import platform, sys; print("linux_x86_64" if sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"} else "")')"

PADDLE_WHL_URL="${PADDLE_WHL_URL:-}"
PADDLE_USE_GPU_WHEEL="${PADDLE_USE_GPU_WHEEL:-1}"
PADDLE_PIP_SPEC="${PADDLE_PIP_SPEC:-paddlepaddle==3.2.0}"
PADDLE_GPU_PIP_SPEC="${PADDLE_GPU_PIP_SPEC:-paddlepaddle-gpu==3.2.0}"
PADDLE_INDEX_URL="${PADDLE_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cpu/}"
PADDLE_GPU_INDEX_URL="${PADDLE_GPU_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cu118/}"
PADDLE_OCR_SPEC="${PADDLE_OCR_SPEC:-paddleocr[all]}"
export PADDLE_PDX_MODEL_SOURCE="${PADDLE_PDX_MODEL_SOURCE:-BOS}"
if [ -z "${PADDLE_WHL_URL}" ] && [ "${PADDLE_USE_GPU_WHEEL}" != "0" ] && [ "${PY_TAG}" = "cp310" ] && [ "${PLATFORM_TAG}" = "linux_x86_64" ]; then
  PADDLE_WHL_URL="https://paddle-whl.bj.bcebos.com/stable/cu118/paddlepaddle-gpu/paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl"
fi

if [ -n "${PADDLE_WHL_URL}" ]; then
  WHEEL_NAME="$(basename "${PADDLE_WHL_URL}")"
  WHEEL_PATH="${CHECKPOINTS_DIR}/${WHEEL_NAME}"
  if [ ! -f "${WHEEL_PATH}" ]; then
    if command -v wget >/dev/null 2>&1; then
      wget -c -nc -P "${CHECKPOINTS_DIR}" "${PADDLE_WHL_URL}"
    else
      curl -L "${PADDLE_WHL_URL}" -o "${WHEEL_PATH}"
    fi
  fi
  pip install "${WHEEL_PATH}"
elif [ "${PADDLE_USE_GPU_WHEEL}" != "0" ]; then
  pip install "${PADDLE_GPU_PIP_SPEC}" -i "${PADDLE_GPU_INDEX_URL}"
else
  pip install "${PADDLE_PIP_SPEC}" -i "${PADDLE_INDEX_URL}"
fi

pip install "${PADDLE_OCR_SPEC}"
pip install "numpy<2"
