#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG_PATH="${BENCH_CONFIG:-}"
EXPERIMENT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="${2:?--config requires a file}"; shift 2 ;;
        --experiment) EXPERIMENT="${2:?--experiment requires a YAML file}"; shift 2 ;;
        -h|--help)
            printf 'Usage: %s --config FILE [--experiment EXPERIMENT.yaml]\n' "$0"
            exit 0
            ;;
        *) die "Unknown argument: $1"; exit 2 ;;
    esac
done

load_cluster_config "${CONFIG_PATH}"
activate_bench_venv

printf '=== Python ===\n'
python -VV
python -m pip --version

printf '\n=== NVIDIA driver / CUDA visibility ===\n'
if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi is unavailable; optional GPU telemetry and some driver metadata will be null."
elif ! nvidia-smi; then
    warn "nvidia-smi could not query a GPU; optional telemetry will be unavailable. PyTorch CUDA usability is checked separately."
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    warn "nvcc is not on PATH. A compiler toolkit is not always required by a vLLM wheel; verify the installed build's requirements."
fi

printf '\n=== vLLM ===\n'
if ! python -c 'import vllm; print(vllm.__version__)'; then
    die "vLLM is not importable in ${VENV_PATH}. Follow the current official installation guide for the detected Python/driver/CUDA environment."
    exit 1
fi
if ! python -c 'import torch; print(f"torch={torch.__version__}"); print(f"torch_cuda_build={torch.version.cuda}"); print(f"torch_cuda_available={torch.cuda.is_available()}"); raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    die "PyTorch cannot use CUDA in this allocation. Verify driver/CUDA/PyTorch/vLLM compatibility against current official documentation."
    exit 1
fi
if ! python -m pip check; then
    die "Installed Python packages have incompatible dependency requirements (pip check failed)."
    exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
    die "The vllm CLI is not available in the active virtual environment."
    exit 1
fi
vllm --help >/dev/null

if [[ -n "${EXPERIMENT}" ]]; then
    benchmark_type="$(config_value "${EXPERIMENT}" benchmark_type)"
    python -m llm_bench.cli validate-config "${EXPERIMENT}" --benchmark-type "${benchmark_type}"
    tensor_parallel_size="$(config_value "${EXPERIMENT}" tensor_parallel_size)"
    visible_gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
    if (( visible_gpu_count < tensor_parallel_size )); then
        die "Experiment requests tensor_parallel_size=${tensor_parallel_size}, but PyTorch sees only ${visible_gpu_count} GPU(s). Request enough GPUs on one node or lower the setting."
        exit 1
    fi
    if [[ "${SLURM_NNODES:-1}" =~ ^[0-9]+$ ]] && (( SLURM_NNODES > 1 )); then
        die "Multi-node inference is not implemented in this initial repository; request one Slurm node."
        exit 1
    fi
    case "${benchmark_type}" in
        smoke|offline) vllm bench throughput --help >/dev/null ;;
        serving)
            if ! vllm serve --help=all >/dev/null 2>&1; then
                vllm serve --help >/dev/null
            fi
            vllm bench serve --help >/dev/null
            command -v curl >/dev/null 2>&1 || { die "curl is required for the bounded server health check."; exit 1; }
            ;;
        *) die "Unsupported benchmark_type in ${EXPERIMENT}: ${benchmark_type}"; exit 1 ;;
    esac
fi

for path_name in HF_HOME MODEL_CACHE_DIR; do
    require_value "${path_name}" || exit 1
    path_value="${!path_name}"
    [[ "${path_value}" == /* ]] || { die "${path_name} must be an absolute path: ${path_value}"; exit 1; }
    mkdir -p "${path_value}" || { die "Cannot create ${path_name}: ${path_value}"; exit 1; }
    [[ -w "${path_value}" ]] || { die "${path_name} is not writable: ${path_value}"; exit 1; }
    resolved_path="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${path_value}")"
    if [[ "${resolved_path}" == "${REPO_ROOT}" || "${resolved_path}" == "${REPO_ROOT}/"* ]]; then
        die "${path_name} must be outside the Git repository: ${resolved_path}"
        exit 1
    fi
done

require_value RESULTS_ROOT || exit 1
[[ "${RESULTS_ROOT}" == /* ]] || { die "RESULTS_ROOT must be an absolute path: ${RESULTS_ROOT}"; exit 1; }
mkdir -p "${RESULTS_ROOT}" || { die "Cannot create RESULTS_ROOT: ${RESULTS_ROOT}"; exit 1; }
[[ -w "${RESULTS_ROOT}" ]] || { die "RESULTS_ROOT is not writable: ${RESULTS_ROOT}"; exit 1; }
resolved_results="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "${RESULTS_ROOT}")"
if [[ "${resolved_results}" == "${REPO_ROOT}" ]] || \
   [[ "${resolved_results}" == "${REPO_ROOT}/"* && "${resolved_results}" != "${REPO_ROOT}/results" ]]; then
    die "Inside the checkout, RESULTS_ROOT may only be the ignored ${REPO_ROOT}/results directory."
    exit 1
fi

{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    python -VV
    python -c 'import torch, vllm; print(f"vllm={vllm.__version__}"); print(f"torch={torch.__version__}"); print(f"torch_cuda_build={torch.version.cuda}"); print(f"torch_cuda_available={torch.cuda.is_available()}")'
    python -m pip --version
    python -m pip freeze
} > "${VENV_PATH}/preflight_versions.txt"

printf '\nPreflight passed. This verifies visibility, not that a particular model fits the allocated GPU.\n'
