#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE=""
INTERVAL_MS=1000
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_FILE="${2:?--output requires a CSV path}"; shift 2 ;;
        --interval-ms) INTERVAL_MS="${2:?--interval-ms requires an integer}"; shift 2 ;;
        -h|--help)
            printf 'Usage: %s --output FILE [--interval-ms MILLISECONDS]\n' "$0"
            exit 0
            ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "${OUTPUT_FILE}" ]] || { printf 'ERROR: --output is required.\n' >&2; exit 2; }
[[ "${INTERVAL_MS}" =~ ^[1-9][0-9]*$ ]] || { printf 'ERROR: interval must be a positive integer.\n' >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT_FILE}")"

HEADER='timestamp,gpu_index,gpu_uuid,gpu_utilization_percent,memory_utilization_percent,memory_used_mib,memory_total_mib,power_draw_watts,sm_clock_mhz,memory_clock_mhz,temperature_celsius'
printf '%s\n' "${HEADER}" > "${OUTPUT_FILE}"
SCOPE_FILE="${OUTPUT_FILE%.csv}.scope.txt"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'scope=disabled; reason=nvidia-smi unavailable\n' > "${SCOPE_FILE}"
    printf 'WARNING: nvidia-smi unavailable; telemetry file contains only its header.\n' >&2
    exit 0
fi

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
if [[ -z "${VISIBLE_DEVICES}" || "${VISIBLE_DEVICES}" == "all" || \
      "${VISIBLE_DEVICES}" == "none" || "${VISIBLE_DEVICES}" == "void" || \
      ! "${VISIBLE_DEVICES}" =~ ^[A-Za-z0-9_.,:/-]+$ ]]; then
    printf 'scope=disabled; reason=no usable allocated-GPU visibility environment variable\n' > "${SCOPE_FILE}"
    printf 'WARNING: no usable allocated-GPU visibility list is set; telemetry is disabled rather than sampling every node GPU.\n' >&2
    exit 0
fi
printf 'scope_source=%s\ndevices=%s\n' "${VISIBLE_SOURCE}" "${VISIBLE_DEVICES}" > "${SCOPE_FILE}"

QUERY='timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu'
STOP=0
trap 'STOP=1' TERM INT
interval_seconds="$(awk -v ms="${INTERVAL_MS}" 'BEGIN { printf "%.3f", ms / 1000 }')"

while [[ "${STOP}" == "0" ]]; do
    if ! nvidia-smi --id="${VISIBLE_DEVICES}" --query-gpu="${QUERY}" \
        --format=csv,noheader,nounits >> "${OUTPUT_FILE}" 2>/dev/null; then
        printf 'WARNING: GPU telemetry query failed or contains an unsupported metric; continuing without telemetry.\n' >&2
        exit 0
    fi
    sleep "${interval_seconds}" &
    wait $! || true
done
exit 0
