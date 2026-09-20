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
# --cpu-list is an alternative to --cpu-bind: it pins to a specific physical
# core range (numactl --physcpubind) instead of an entire NUMA node
# (--cpunodebind), so two concurrent instances can be given disjoint cores on
# the same socket -- e.g. one on cores 0-27 driving DDR, another on cores
# 28-55 driving HBM, with neither starved of CPU by the other. --threads is
# required alongside --cpu-list, since a subset of a node's cores must not be
# oversubscribed by OpenMP defaulting to the whole node's core count.
#
#   scripts/membench/run_stream.sh --size below_cliff --cpu-bind 0 --mem-bind 2 \
#       --target flat --output-dir ~/membench-results [--config CONFIG] [--threads 56]
#   scripts/membench/run_stream.sh --size small --cpu-list 0-27 --mem-bind 2 \
#       --threads 28 --target flat --output-dir ~/membench-results
set -euo pipefail

# _common.sh recomputes its own SCRIPT_DIR/REPO_ROOT globals when sourced,
# clobbering any same-named variables in the caller -- so this repo's
# membench directory is captured under a distinct name.
MEMBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MEMBENCH_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/_common.sh"

SIZE=""
CPU_BIND=""
CPU_LIST=""
MEM_BIND=""
TARGET=""
OUTPUT_DIR=""
CONFIG=""
THREADS=""
BIN_DIR="${MEMBENCH_DIR}/build"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="$2"; shift 2 ;;
        --cpu-bind) CPU_BIND="$2"; shift 2 ;;
        --cpu-list) CPU_LIST="$2"; shift 2 ;;
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
if [[ -n "${CPU_BIND}" && -n "${CPU_LIST}" ]]; then
    die "--cpu-bind and --cpu-list are mutually exclusive"
fi
[[ -n "${CPU_BIND}" || -n "${CPU_LIST}" ]] || die "one of --cpu-bind or --cpu-list is required"
[[ -n "${CPU_LIST}" && -z "${THREADS}" ]] && die "--threads is required alongside --cpu-list"
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

if [[ -n "${CPU_LIST}" ]]; then
    CPU_FLAG="--physcpubind=${CPU_LIST}"
    CPU_RECORD="${CPU_LIST}"
else
    CPU_FLAG="--cpunodebind=${CPU_BIND}"
    CPU_RECORD="${CPU_BIND}"
fi

printf 'STREAM %s: cpu=%s mem-bind=%s target=%s\n' \
    "${SIZE}" "${CPU_RECORD}" "${MEM_BIND}" "${TARGET}" >&2

OUTPUT="$(numactl "${CPU_FLAG}" --membind="${MEM_BIND}" "${BIN}")"
printf '%s\n' "${OUTPUT}" >&2

PARSE_ARGS=(
    --tool stream --target "${TARGET}" --output-dir "${OUTPUT_DIR}"
    --cpu-bind "${CPU_RECORD}" --mem-bind "${MEM_BIND}"
    --param "array_size=${SIZE}"
)
[[ -n "${THREADS}" ]] && PARSE_ARGS+=(--threads "${THREADS}")

printf '%s' "${OUTPUT}" | python3 -m llm_bench.membench "${PARSE_ARGS[@]}"
