#!/usr/bin/env bash
# Run Intel MLC's bandwidth and latency matrices. Unlike STREAM, MLC manages
# its own per-NUMA-node placement internally -- `--bandwidth_matrix` and
# `--latency_matrix` already sweep every node pair on their own, so this
# script does not wrap it in numactl. `--latency_matrix` may need root for
# some counters; if it isn't available, the failure is captured as a warning
# in the written record rather than aborting the whole run.
#
#   scripts/membench/run_mlc.sh --bin ~/mlc/Linux/mlc --target flat \
#       --output-dir ~/membench-results [--config CONFIG]
set -euo pipefail

# _common.sh recomputes its own SCRIPT_DIR/REPO_ROOT globals when sourced,
# clobbering any same-named variables in the caller -- so this repo's
# membench directory is captured under a distinct name.
MEMBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MEMBENCH_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_common.sh"

BIN="${MLC_BIN:-}"
TARGET=""
OUTPUT_DIR=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin) BIN="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ -n "${BIN}" ]] || die "--bin (or \$MLC_BIN) is required; run build_mlc.sh first."
[[ -x "${BIN}" ]] || die "MLC binary not executable: ${BIN}"
[[ -n "${TARGET}" ]] || die "--target is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"

if [[ -n "${CONFIG}" ]]; then
    load_cluster_config "${CONFIG}"
    activate_bench_venv
fi

# Prefer this checkout's own `llm_bench` over whatever an editable install in
# the shared venv happens to point at -- the venv may have been `pip install
# -e`'d from a different checkout than the one running this script.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_matrix() {
    local kind="$1" flag="$2" tool="$3"
    local output status=0
    printf 'MLC %s (%s)\n' "${kind}" "${flag}" >&2
    if ! output="$("${BIN}" "${flag}" 2>&1)"; then
        status=$?
        # Retry once, non-interactively, with sudo -- only after a real
        # failure, and only if it can succeed without prompting for a
        # password (this must never hang a script waiting on stdin).
        if [[ "${EUID:-$(id -u)}" -ne 0 ]] && command -v sudo >/dev/null \
            && sudo -n true 2>/dev/null; then
            printf 'Retrying %s with sudo -n\n' "${kind}" >&2
            if output="$(sudo -n "${BIN}" "${flag}" 2>&1)"; then
                status=0
            fi
        fi
    fi
    printf '%s\n' "${output}" >&2
    if [[ "${status}" -ne 0 ]]; then
        printf 'WARNING: MLC %s exited %d (may need root); recording as a warning.\n' \
            "${kind}" "${status}" >&2
    fi
    printf '%s' "${output}" | python3 -m llm_bench.membench \
        --tool "${tool}" --target "${TARGET}" --output-dir "${OUTPUT_DIR}" \
        --param "kind=${kind}"
}

run_matrix "bandwidth_matrix" "--bandwidth_matrix" "mlc-bandwidth"
run_matrix "latency_matrix" "--latency_matrix" "mlc-latency"
