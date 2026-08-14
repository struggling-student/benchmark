#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG_PATH="${BENCH_CONFIG:-}"
WITH_DEV=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="${2:?--config requires a file}"; shift 2 ;;
        --with-dev) WITH_DEV=1; shift ;;
        -h|--help)
            printf 'Usage: %s --config FILE [--with-dev]\n' "$0"
            exit 0
            ;;
        *) die "Unknown argument: $1"; exit 2 ;;
    esac
done

load_cluster_config "${CONFIG_PATH}"
require_value PYTHON_EXECUTABLE
require_value VENV_PATH

if ! command -v "${PYTHON_EXECUTABLE}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
    die "Configured Python executable is unavailable: ${PYTHON_EXECUTABLE}"
    exit 1
fi

printf 'Selected Python:\n'
"${PYTHON_EXECUTABLE}" -VV
printf '\nDetected NVIDIA driver / GPU visibility (informational on a login node):\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || warn "nvidia-smi exists but cannot query a GPU in this shell."
else
    warn "nvidia-smi is unavailable here. Inspect it inside a GPU allocation before choosing a vLLM build."
fi
printf '\nDetected CUDA compiler toolkit (informational):\n'
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    warn "nvcc is not on PATH. Verify whether the selected vLLM distribution requires a local toolkit."
fi
printf 'Creating virtual environment at %s\n' "${VENV_PATH}"
"${PYTHON_EXECUTABLE}" -m venv "${VENV_PATH}"
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

install_target="${REPO_ROOT}"
if [[ "${WITH_DEV}" == "1" ]]; then
    install_target="${REPO_ROOT}[dev]"
fi
python -m pip install -e "${install_target}"

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    python -VV
    python -m pip --version
    python -m pip freeze
} > "${VENV_PATH}/setup_versions.txt"

printf '\nThe benchmark runner is installed. Backend runtimes remain independently managed.\n'
printf 'Configure native vLLM/llama.cpp paths or pinned Docker/Apptainer images, then run\n'
printf 'scripts/preflight_check.sh with a model, workload, profile, provider, and variant.\n'
