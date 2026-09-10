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
# sam3 currently imports pkg_resources at runtime via model_builder.py, so keep
# setuptools on a version that still ships that module.
pip install -U pip wheel "setuptools==68.1.2"

# NumPy 1.25.x has no Python 3.12 wheel, which forces a broken source build path.
pip install "numpy<2"
# PyTorch's Blackwell support starts with the CUDA 12.8 wheels. RTX 50-series
# GPUs such as the 5070 Ti need this wheel line (or newer) to avoid missing
# sm_120 kernels.
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"


# Upstream checkouts live beside the venv, not inside it: the runtime rebuilds the venv
# from scratch on a failed health check, and re-cloning sam3 every time is pure waste.
PKG_ROOT="${CACHE_ROOT}/pkgs"
SAM3_DIR="${PKG_ROOT}/sam3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_HELPER="${SCRIPT_DIR}/../../core/sam3_patch.py"
mkdir -p "${PKG_ROOT}"

if [ ! -d "${SAM3_DIR}" ]; then
  git clone --depth 1 https://github.com/facebookresearch/sam3.git "${SAM3_DIR}"
fi

cd "${SAM3_DIR}"
"${VENV_DIR}/bin/python" "${PATCH_HELPER}" "${SAM3_DIR}"
# Install sam3 as a regular package; editable installs can still rely on
# pkg_resources-style runtime hooks that are absent in fresh Python 3.12 venvs.
pip install .
pip install \
  "accelerate>=1.6,<2" \
  einops \
  decord \
  psutil \
  pycocotools \
  opencv-python-headless==4.10.0.84
