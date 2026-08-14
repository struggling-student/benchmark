#!/usr/bin/env bash
# Shared environment helpers for the generic benchmark entry points.

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

    # The private cluster file is trusted shell input. It must never contain tokens.
    set -a
    # shellcheck disable=SC1090
    source "${config_path}"
    set +a

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
