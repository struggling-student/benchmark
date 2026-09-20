#!/usr/bin/env bash
# Launch two STREAM instances simultaneously against independent (cpu, memory)
# bindings and report each Triad rate plus their sum. This is the direct test
# of "does concurrent access to two separate memory controllers beat either
# alone."
#
# Use --a-cpu-list/--b-cpu-list (with --threads) rather than
# --a-cpu-bind/--b-cpu-bind whenever both legs target the same NUMA node's
# cores (e.g. socket 0's DDR vs socket 0's HBM): --cpu-bind pins to the whole
# node, so two legs on the *same* node would fight over all of its cores.
# --cpu-list splits it into disjoint physical core ranges instead, so neither
# leg is CPU-starved by the other -- e.g.:
#
#   scripts/membench/run_concurrent_stream.sh --size small --threads 28 \
#       --a-cpu-list 0-27 --a-mem-bind 0 --b-cpu-list 28-55 --b-mem-bind 2 \
#       --target flat --output-dir ~/membench-results [--config CONFIG]
#
# --a-cpu-bind/--b-cpu-bind (whole-node binding) remains useful for legs on
# genuinely different nodes, e.g. cross-socket comparisons:
#
#   scripts/membench/run_concurrent_stream.sh --size small \
#       --a-cpu-bind 0 --a-mem-bind 0 --b-cpu-bind 1 --b-mem-bind 3 \
#       --target flat --output-dir ~/membench-results [--config CONFIG]
set -euo pipefail

# _common.sh recomputes its own SCRIPT_DIR/REPO_ROOT globals when sourced,
# clobbering any same-named variables in the caller -- so this repo's
# membench directory is captured under a distinct name.
MEMBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${MEMBENCH_DIR}/../_common.sh"

SIZE=""
A_CPU_BIND=""; A_CPU_LIST=""; A_MEM=""
B_CPU_BIND=""; B_CPU_LIST=""; B_MEM=""
TARGET=""
OUTPUT_DIR=""
CONFIG=""
THREADS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="$2"; shift 2 ;;
        --a-cpu-bind) A_CPU_BIND="$2"; shift 2 ;;
        --a-cpu-list) A_CPU_LIST="$2"; shift 2 ;;
        --a-mem-bind) A_MEM="$2"; shift 2 ;;
        --b-cpu-bind) B_CPU_BIND="$2"; shift 2 ;;
        --b-cpu-list) B_CPU_LIST="$2"; shift 2 ;;
        --b-mem-bind) B_MEM="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

for v in SIZE A_MEM B_MEM TARGET OUTPUT_DIR; do
    [[ -n "${!v}" ]] || die "missing required argument for ${v}"
done
[[ -n "${A_CPU_BIND}" || -n "${A_CPU_LIST}" ]] || die "one of --a-cpu-bind or --a-cpu-list is required"
[[ -n "${B_CPU_BIND}" || -n "${B_CPU_LIST}" ]] || die "one of --b-cpu-bind or --b-cpu-list is required"

A_CPU_ARGS=(--cpu-bind "${A_CPU_BIND}")
[[ -n "${A_CPU_LIST}" ]] && A_CPU_ARGS=(--cpu-list "${A_CPU_LIST}")
B_CPU_ARGS=(--cpu-bind "${B_CPU_BIND}")
[[ -n "${B_CPU_LIST}" ]] && B_CPU_ARGS=(--cpu-list "${B_CPU_LIST}")

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(--size "${SIZE}" --target "${TARGET}" --output-dir "${OUTPUT_DIR}")
[[ -n "${CONFIG}" ]] && COMMON_ARGS+=(--config "${CONFIG}")
[[ -n "${THREADS}" ]] && COMMON_ARGS+=(--threads "${THREADS}")

printf 'Launching concurrent STREAM pair: A(%s %s,mem=%s) B(%s %s,mem=%s)\n' \
    "${A_CPU_ARGS[@]}" "${A_MEM}" "${B_CPU_ARGS[@]}" "${B_MEM}" >&2

"${MEMBENCH_DIR}/run_stream.sh" "${COMMON_ARGS[@]}" "${A_CPU_ARGS[@]}" --mem-bind "${A_MEM}" \
    >"${OUTPUT_DIR}/.concurrent_a.stdout" 2>&1 &
PID_A=$!
"${MEMBENCH_DIR}/run_stream.sh" "${COMMON_ARGS[@]}" "${B_CPU_ARGS[@]}" --mem-bind "${B_MEM}" \
    >"${OUTPUT_DIR}/.concurrent_b.stdout" 2>&1 &
PID_B=$!

STATUS=0
wait "${PID_A}" || STATUS=$?
wait "${PID_B}" || STATUS=$?

cat "${OUTPUT_DIR}/.concurrent_a.stdout" >&2
cat "${OUTPUT_DIR}/.concurrent_b.stdout" >&2
PATH_A="$(tail -n1 "${OUTPUT_DIR}/.concurrent_a.stdout")"
PATH_B="$(tail -n1 "${OUTPUT_DIR}/.concurrent_b.stdout")"
rm -f "${OUTPUT_DIR}/.concurrent_a.stdout" "${OUTPUT_DIR}/.concurrent_b.stdout"

if [[ "${STATUS}" -ne 0 ]]; then
    die "one or both concurrent STREAM runs failed; see output above"
fi

python3 - "${PATH_A}" "${PATH_B}" <<'PY'
import json
import sys

path_a, path_b = sys.argv[1], sys.argv[2]
a = json.load(open(path_a))
b = json.load(open(path_b))
triad_a = a["results"]["triad"]["best_mbps"]
triad_b = b["results"]["triad"]["best_mbps"]
print(f"A (cpu={a['cpu_bind']}, mem={a['mem_bind']}): {triad_a:.1f} MB/s Triad -> {path_a}")
print(f"B (cpu={b['cpu_bind']}, mem={b['mem_bind']}): {triad_b:.1f} MB/s Triad -> {path_b}")
print(f"Sum: {triad_a + triad_b:.1f} MB/s")
PY
