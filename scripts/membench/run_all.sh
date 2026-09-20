#!/usr/bin/env bash
# Single entry point: run the STREAM array-size sweep (+ concurrent-tier test
# on flat) and the MLC bandwidth/latency matrices, writing everything under
# --output-dir. Node ids default to CRESCO8's confirmed topology (node 0 =
# socket 0 DDR, node 2 = socket 0 HBM on the flat node cresco8-hbm14) -- pass
# --ddr-node/--hbm-node to override if a different node's numbering differs.
#
#   scripts/membench/run_all.sh --target flat --output-dir ~/membench-results \
#       --config configs/cluster/cresco8.env [--mlc-bin PATH]
#   scripts/membench/run_all.sh --target cache --output-dir ~/membench-results \
#       --config configs/cluster/cresco8.env
set -euo pipefail

# _common.sh recomputes its own SCRIPT_DIR/REPO_ROOT globals when sourced,
# clobbering any same-named variables in the caller -- so this repo's
# membench directory is captured under a distinct name.
MEMBENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${MEMBENCH_DIR}/../_common.sh"

TARGET=""
OUTPUT_DIR=""
CONFIG=""
DDR_NODE=0
HBM_NODE=2
MLC_BIN="${MLC_BIN:-}"
SKIP_MLC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --ddr-node) DDR_NODE="$2"; shift 2 ;;
        --hbm-node) HBM_NODE="$2"; shift 2 ;;
        --mlc-bin) MLC_BIN="$2"; shift 2 ;;
        --skip-mlc) SKIP_MLC=1; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

case "${TARGET}" in
    cache|flat) ;;
    *) die "--target must be 'cache' or 'flat'" ;;
esac
[[ -n "${OUTPUT_DIR}" ]] || die "--output-dir is required"

mkdir -p "${OUTPUT_DIR}"
COMMON=(--target "${TARGET}" --output-dir "${OUTPUT_DIR}")
[[ -n "${CONFIG}" ]] && COMMON+=(--config "${CONFIG}")

printf '=== STREAM array-size sweep on DDR (node %s) ===\n' "${DDR_NODE}"
for size in small below_cliff above_cliff; do
    "${MEMBENCH_DIR}/run_stream.sh" "${COMMON[@]}" \
        --size "${size}" --cpu-bind "${DDR_NODE}" --mem-bind "${DDR_NODE}"
done

if [[ "${TARGET}" == "flat" ]]; then
    printf '=== STREAM array-size sweep on HBM (node %s) ===\n' "${HBM_NODE}"
    # above_cliff (~96 GiB) does not fit in one 64 GiB HBM node -- HBM-bound
    # numactl would fail with ENOMEM, so it is intentionally skipped here.
    for size in small below_cliff; do
        "${MEMBENCH_DIR}/run_stream.sh" "${COMMON[@]}" \
            --size "${size}" --cpu-bind "${DDR_NODE}" --mem-bind "${HBM_NODE}"
    done

    printf '=== Concurrent STREAM: socket %s DDR + socket %s HBM, simultaneously ===\n' \
        "${DDR_NODE}" "${DDR_NODE}"
    "${MEMBENCH_DIR}/run_concurrent_stream.sh" "${COMMON[@]}" --size small \
        --a-cpu-bind "${DDR_NODE}" --a-mem-bind "${DDR_NODE}" \
        --b-cpu-bind "${DDR_NODE}" --b-mem-bind "${HBM_NODE}"
else
    printf 'Cache mode exposes only DDR-backed NUMA nodes -- there is no second\n'
    printf 'tier to bind to, so the concurrent-tier test is skipped here. The\n'
    printf 'array-size sweep above is how the cache-mode HBM effect shows up:\n'
    printf 'compare small/below_cliff (should ride the transparent HBM cache)\n'
    printf 'against above_cliff (should fall back toward plain DDR).\n'
fi

if [[ "${SKIP_MLC}" -eq 1 ]]; then
    printf 'Skipping MLC (--skip-mlc).\n'
elif [[ -z "${MLC_BIN}" ]]; then
    printf 'WARNING: no --mlc-bin/$MLC_BIN given; skipping MLC. Run build_mlc.sh\n' >&2
    printf 'first, or pass --skip-mlc to silence this warning.\n' >&2
else
    printf '=== MLC bandwidth + latency matrices ===\n'
    "${MEMBENCH_DIR}/run_mlc.sh" "${COMMON[@]}" --bin "${MLC_BIN}"
fi

printf 'Done. Results in %s\n' "${OUTPUT_DIR}"
