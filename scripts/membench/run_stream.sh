#!/usr/bin/env bash
# Run one compiled STREAM binary bound to a specific NUMA node pair, then parse
# its output and write a membench JSON result via `python -m llm_bench.membench`.
#
# --cpu-bind and --mem-bind are separate on purpose: for a clean single-tier
# bandwidth reading you generally want both pinned to the same node (e.g.
# --cpu-bind 0 --mem-bind 2 tests socket 0's cores against socket 0's HBM), but
# run_concurrent_stream.sh also uses this script with independent binds to
# probe cross-socket/cross-tier behaviour deliberately.
#
#   scripts/membench/run_stream.sh --size below_cliff --cpu-bind 0 --mem-bind 2 \
#       --target flat --output-dir ~/membench-results [--config CONFIG] [--threads 56]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_common.sh"

SIZE=""
CPU_BIND=""
MEM_BIND=""
TARGET=""
OUTPUT_DIR=""
CONFIG=""
THREADS=""
BIN_DIR="${SCRIPT_DIR}/build"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="$2"; shift 2 ;;
        --cpu-bind) CPU_BIND="$2"; shift 2 ;;
        --mem-bind) MEM_BIND="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --bin-dir) BIN_DIR="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ -n "${SIZE}" ]] || die "--size is required"
[[ -n "${CPU_BIND}" ]] || die "--cpu-bind is required"
[[ -n "${MEM_BIND}" ]] || die "--mem-bind is required"
[[ -n "${TARGET}" ]] || die "--target is required"
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"

BIN="${BIN_DIR}/stream_${SIZE}"
[[ -x "${BIN}" ]] || die "Missing ${BIN}; run scripts/membench/build_stream.sh first."
command -v numactl >/dev/null || die "numactl is required."

if [[ -n "${CONFIG}" ]]; then
    load_cluster_config "${CONFIG}"
    activate_bench_venv
fi

# Prefer this checkout's own `llm_bench` over whatever an editable install in
# the shared venv happens to point at -- the venv may have been `pip install
# -e`'d from a different checkout than the one running this script.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_PROC_BIND=close
export OMP_PLACES=cores
[[ -n "${THREADS}" ]] && export OMP_NUM_THREADS="${THREADS}"

printf 'STREAM %s: cpu-bind=%s mem-bind=%s target=%s\n' \
    "${SIZE}" "${CPU_BIND}" "${MEM_BIND}" "${TARGET}" >&2

OUTPUT="$(numactl --cpunodebind="${CPU_BIND}" --membind="${MEM_BIND}" "${BIN}")"
printf '%s\n' "${OUTPUT}" >&2

PARSE_ARGS=(
    --tool stream --target "${TARGET}" --output-dir "${OUTPUT_DIR}"
    --cpu-bind "${CPU_BIND}" --mem-bind "${MEM_BIND}"
    --param "array_size=${SIZE}"
)
[[ -n "${THREADS}" ]] && PARSE_ARGS+=(--threads "${THREADS}")

printf '%s' "${OUTPUT}" | python3 -m llm_bench.membench "${PARSE_ARGS[@]}"
