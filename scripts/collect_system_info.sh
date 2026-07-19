#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="${2:?--output-dir requires a directory}"; shift 2 ;;
        -h|--help)
            printf 'Usage: %s --output-dir RUN_DIRECTORY\n' "$0"
            exit 0
            ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "${OUTPUT_DIR}" ]] || { printf 'ERROR: --output-dir is required.\n' >&2; exit 2; }
SYSTEM_DIR="${OUTPUT_DIR}/system"
mkdir -p "${SYSTEM_DIR}"

hostname > "${SYSTEM_DIR}/hostname.txt" 2>&1 || true
(command -v lscpu >/dev/null 2>&1 && lscpu || printf 'lscpu unavailable\n') > "${SYSTEM_DIR}/lscpu.txt" 2>&1
(command -v numactl >/dev/null 2>&1 && numactl --hardware || printf 'numactl unavailable\n') > "${SYSTEM_DIR}/numactl.txt" 2>&1
VISIBLE_SOURCE=""
VISIBLE_DEVICES=""
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    VISIBLE_SOURCE=CUDA_VISIBLE_DEVICES
    VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
elif [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
    VISIBLE_SOURCE=NVIDIA_VISIBLE_DEVICES
    VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES//[[:space:]]/}"
elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    VISIBLE_SOURCE=SLURM_STEP_GPUS
    VISIBLE_DEVICES="${SLURM_STEP_GPUS//[[:space:]]/}"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    VISIBLE_SOURCE=SLURM_JOB_GPUS
    VISIBLE_DEVICES="${SLURM_JOB_GPUS//[[:space:]]/}"
fi
if [[ -n "${VISIBLE_DEVICES}" && "${VISIBLE_DEVICES}" != "all" && \
      "${VISIBLE_DEVICES}" != "none" && "${VISIBLE_DEVICES}" != "void" && \
      "${VISIBLE_DEVICES}" =~ ^[A-Za-z0-9_.,:/-]+$ ]]; then
    printf 'scope_source=%s\ndevices=%s\n' "${VISIBLE_SOURCE}" "${VISIBLE_DEVICES}" > "${SYSTEM_DIR}/nvidia_smi_scope.txt"
    nvidia-smi --id="${VISIBLE_DEVICES}" > "${SYSTEM_DIR}/nvidia_smi.txt" 2>&1 || \
        printf 'nvidia-smi scoped query failed\n' > "${SYSTEM_DIR}/nvidia_smi.txt"
    nvidia-smi --id="${VISIBLE_DEVICES}" \
        --query-gpu=timestamp,index,uuid,name,driver_version,memory.total,power.limit \
        --format=csv > "${SYSTEM_DIR}/nvidia_smi_query.csv" 2>&1 || \
        printf 'nvidia-smi query unsupported or failed\n' > "${SYSTEM_DIR}/nvidia_smi_query.csv"
else
    printf 'scope=disabled; reason=no usable allocated-GPU visibility list\n' > "${SYSTEM_DIR}/nvidia_smi_scope.txt"
    printf 'nvidia-smi inventory omitted to avoid recording unallocated node GPUs\n' > "${SYSTEM_DIR}/nvidia_smi.txt"
    printf 'nvidia-smi scoped query unavailable\n' > "${SYSTEM_DIR}/nvidia_smi_query.csv"
fi
(python -VV; python -m pip --version) > "${SYSTEM_DIR}/python_version.txt" 2>&1 || true
python -m pip freeze > "${SYSTEM_DIR}/pip_freeze.txt" 2>&1 || true
(type module >/dev/null 2>&1 && module list || printf 'environment modules unavailable or none loaded\n') > "${SYSTEM_DIR}/environment_modules.txt" 2>&1
env | LC_ALL=C sort | awk -F= '$1 ~ /^SLURM_/ {print}' > "${SYSTEM_DIR}/slurm_environment.txt"
