#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG_PATH="${BENCH_CONFIG:-}"
MODEL=""
WORKLOAD=""
PROFILE=""
PROVIDER=""
VARIANT=""
RESULT_DIR=""
DRY_RUN=0
LOG_LEVEL="${LLM_BENCH_LOG_LEVEL:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="${2:?--config requires a file}"; shift 2 ;;
        --model) MODEL="${2:?--model requires a YAML file}"; shift 2 ;;
        --workload) WORKLOAD="${2:?--workload requires a YAML file}"; shift 2 ;;
        --profile) PROFILE="${2:?--profile requires a YAML file}"; shift 2 ;;
        --provider) PROVIDER="${2:?--provider requires a value}"; shift 2 ;;
        --variant) VARIANT="${2:?--variant requires a value}"; shift 2 ;;
        --result-dir) RESULT_DIR="${2:?--result-dir requires a directory}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --log-level) LOG_LEVEL="${2:?--log-level requires a value}"; shift 2 ;;
        -h|--help)
            printf 'Usage: %s --config FILE --model MODEL.yaml --workload WORKLOAD.yaml --profile PROFILE.yaml --provider native|docker|apptainer --variant VARIANT [--result-dir DIR] [--dry-run] [--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL]\n' "$0"
            exit 0
            ;;
        *) die "Unknown argument: $1"; exit 2 ;;
    esac
done
for name in MODEL WORKLOAD PROFILE PROVIDER VARIANT; do
    [[ -n "${!name}" ]] || { die "--$(printf '%s' "${name}" | tr '[:upper:]' '[:lower:]' | tr '_' '-') is required"; exit 2; }
done

load_cluster_config "${CONFIG_PATH}"
activate_bench_venv
command=(python -m llm_bench.cli)
[[ -n "${LOG_LEVEL}" ]] && command+=(--log-level "${LOG_LEVEL}")
command+=(
    run
    --model "${MODEL}"
    --workload "${WORKLOAD}"
    --profile "${PROFILE}"
    --provider "${PROVIDER}"
    --variant "${VARIANT}"
    --repository "${REPO_ROOT}"
)
[[ -n "${MODEL_ARTIFACT_ROOT:-}" ]] && command+=(--artifact-root "${MODEL_ARTIFACT_ROOT}")
[[ -n "${RESULTS_ROOT:-}" ]] && command+=(--results-root "${RESULTS_ROOT}")
[[ -n "${RESULT_DIR}" ]] && command+=(--run-dir "${RESULT_DIR}")
[[ "${DRY_RUN}" == "1" ]] && command+=(--dry-run)
"${command[@]}"
