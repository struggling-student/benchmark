#!/usr/bin/env bash
# Shared shell helpers. This file is sourced by the user-facing scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    return 1
}

warn() {
    printf 'WARNING: %s\n' "$*" >&2
}

require_value() {
    local name="$1"
    local value="${!name:-}"
    if [[ -z "${value}" || "${value}" == *'<'*'>'* ]]; then
        die "${name} is unset or still contains a placeholder in the cluster configuration."
    fi
}

load_cluster_config() {
    local config_path="$1"
    if [[ -z "${config_path}" ]]; then
        die "No cluster configuration supplied. Use --config FILE or set BENCH_CONFIG."
        return 1
    fi
    if [[ ! -r "${config_path}" ]]; then
        die "Cluster configuration is not readable: ${config_path}"
        return 1
    fi

    # Cluster configuration is trusted shell input. Export its cache variables to
    # Python/vLLM child processes without ever printing their values.
    set -a
    # shellcheck disable=SC1090
    source "${config_path}"
    set +a

    # Keep the downloader and vLLM/Hugging Face Hub on the same configured cache.
    # The recorded cluster configuration is the source of truth, not ambient state.
    if [[ -n "${MODEL_CACHE_DIR:-}" && "${MODEL_CACHE_DIR}" != *'<'*'>'* ]]; then
        export HF_HUB_CACHE="${MODEL_CACHE_DIR}"
    fi

    if [[ -n "${MODULE_COMMANDS:-}" ]]; then
        printf 'Applying configured environment-module commands.\n'
        eval "${MODULE_COMMANDS}"
    fi
}

activate_bench_venv() {
    require_value VENV_PATH || return 1
    if [[ ! -r "${VENV_PATH}/bin/activate" ]]; then
        die "Virtual environment not found at ${VENV_PATH}; run scripts/create_venv.sh first."
        return 1
    fi
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
}

config_value() {
    local experiment="$1"
    local field="$2"
    python -m llm_bench.cli get-config "${experiment}" "${field}"
}

is_null_value() {
    local value="${1:-}"
    [[ -z "${value}" || "${value}" == "null" || "${value}" == "None" ]]
}

make_result_dir() {
    local experiment="$1"
    local benchmark_type="$2"
    local root="${RESULTS_ROOT:-${REPO_ROOT}/results}"
    local date_part experiment_stem
    require_value RESULTS_ROOT || return 1
    date_part="$(date -u +%Y-%m-%d)"
    experiment_stem="$(basename "${experiment}" .yaml)"
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}-${benchmark_type}-${experiment_stem}-$$"
    RESULT_DIR="${root}/${date_part}/${RUN_ID}"
    mkdir -p "${RESULT_DIR}"
    export RESULT_DIR RUN_ID
}

enable_run_logging() {
    local run_dir="$1"
    if [[ "${LLM_BENCH_LOGGING_ACTIVE:-0}" != "1" ]]; then
        export LLM_BENCH_LOGGING_ACTIVE=1
        exec > >(tee -a "${run_dir}/stdout.log") \
             2> >(tee -a "${run_dir}/stderr.log" >&2)
    fi
}

require_help_flag() {
    local help_file="$1"
    local flag="$2"
    if ! grep -Fq -- "${flag}" "${help_file}"; then
        die "Installed vLLM does not advertise ${flag} in $(basename "${help_file}"). Check the recorded help and current official vLLM documentation."
        return 1
    fi
}

stop_background_process() {
    local pid="${1:-}"
    [[ -z "${pid}" ]] && return 0
    # The grouped stderr redirection also suppresses Bash's expected "Terminated"
    # job notification when a deliberately stopped background process exits.
    {
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            local attempt
            for attempt in 1 2 3 4 5; do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "${pid}" 2>/dev/null; then
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        fi
        wait "${pid}" 2>/dev/null || true
    } 2>/dev/null
}

stop_process_group() {
    local leader_pid="${1:-}"
    local process_group="${2:-}"
    [[ -z "${leader_pid}" || -z "${process_group}" ]] && return 0
    if [[ ! "${leader_pid}" =~ ^[1-9][0-9]*$ || \
          ! "${process_group}" =~ ^[1-9][0-9]*$ ]]; then
        warn "Refusing to signal an invalid process-group identity."
        return 1
    fi

    # Never signal the calling shell's process group, even if a corrupted PID is
    # supplied. The serving launcher creates a separate session before using this.
    local shell_group
    shell_group="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
    if [[ -n "${shell_group}" && "${process_group}" == "${shell_group}" ]]; then
        warn "Refusing to terminate the benchmark shell's own process group."
        return 1
    fi

    {
        if kill -0 -- "-${process_group}" 2>/dev/null; then
            kill -TERM -- "-${process_group}" 2>/dev/null || true
            local attempt
            for attempt in 1 2 3 4 5; do
                kill -0 -- "-${process_group}" 2>/dev/null || break
                sleep 1
            done
            if kill -0 -- "-${process_group}" 2>/dev/null; then
                kill -KILL -- "-${process_group}" 2>/dev/null || true
            fi
        fi
        wait "${leader_pid}" 2>/dev/null || true
    } 2>/dev/null
}
