#!/usr/bin/env bash
# Cluster-side driver for the memory-bandwidth microbenchmark. Submits
# scripts/membench/run_all.sh to Slurm on a cache-mode node, or runs it
# directly over SSH on the flat-mode node, which sits outside Slurm --
# same staging mechanism as scripts/run_cresco8_bf16_flat_campaign.sh.
#
#   scripts/membench/submit_cresco8_membench.sh --target cache [options]
#   scripts/membench/submit_cresco8_membench.sh --target flat [options]
#
# All cluster-specific paths and hostnames come from the private, gitignored
# configs/cluster/cresco8.env (see docs/01_CLUSTER_PREREQUISITES.md) -- never
# hardcode them here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/cluster/cresco8.env"
TARGET=""
PARTITION="cresco8_hbm_dbg"
NODELIST="cresco8-hbm15"
TIME_LIMIT="00:30:00"
CPUS_PER_TASK=112
MLC_BIN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="${2:?}"; shift 2 ;;
        --config) CONFIG="${2:?}"; shift 2 ;;
        --partition) PARTITION="${2:?}"; shift 2 ;;
        --nodelist) NODELIST="${2:?}"; shift 2 ;;
        --time) TIME_LIMIT="${2:?}"; shift 2 ;;
        --cpus-per-task) CPUS_PER_TASK="${2:?}"; shift 2 ;;
        --mlc-bin) MLC_BIN="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

case "${TARGET}" in
    cache|flat) ;;
    *) printf 'ERROR: --target must be "cache" or "flat"\n' >&2; exit 2 ;;
esac
[[ -r "${CONFIG}" ]] || { printf 'ERROR: unreadable cluster config: %s\n' "${CONFIG}" >&2; exit 2; }

set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

: "${BENCH_REPO_ROOT:?set BENCH_REPO_ROOT in ${CONFIG}}"
: "${RESULTS_ROOT:?set RESULTS_ROOT in ${CONFIG}}"

RUN_ALL="${BENCH_REPO_ROOT}/scripts/membench/run_all.sh"
OUTPUT_DIR="${RESULTS_ROOT}/membench-results"

if [[ "${TARGET}" == "cache" ]]; then
    LOGS="${RESULTS_ROOT}/slurm-logs"
    mkdir -p "${LOGS}"
    sbatch \
        --job-name="membench-cache" \
        --partition="${PARTITION}" \
        --account="${SLURM_ACCOUNT:-enea}" \
        --nodelist="${NODELIST}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --exclusive \
        --time="${TIME_LIMIT}" \
        --output="${LOGS}/%x-%j.out" \
        --error="${LOGS}/%x-%j.err" \
        --wrap="bash '${RUN_ALL}' --target cache --output-dir '${OUTPUT_DIR}' --config '${CONFIG}' ${MLC_BIN:+--mlc-bin \"${MLC_BIN}\"}"
    exit 0
fi

: "${BF16_FLAT_NODE:?set BF16_FLAT_NODE in ${CONFIG}}"
: "${BF16_FLAT_STAGE_REPO:?set BF16_FLAT_STAGE_REPO in ${CONFIG}}"
: "${BF16_FLAT_LOG_ROOT:?set BF16_FLAT_LOG_ROOT in ${CONFIG}}"

command -v rsync >/dev/null || { printf 'ERROR: rsync is required to stage the flat-mode node.\n' >&2; exit 2; }
mkdir -p "${BF16_FLAT_LOG_ROOT}"
rsync -a --exclude=.git "${BENCH_REPO_ROOT}/" "${BF16_FLAT_STAGE_REPO}/"

STAGED_RUN_ALL="${BF16_FLAT_STAGE_REPO}/scripts/membench/run_all.sh"
STAGED_CONFIG="${BF16_FLAT_STAGE_REPO}/configs/cluster/cresco8.env"
LOG_FILE="${BF16_FLAT_LOG_ROOT}/membench-flat-$(date -u +%Y%m%dT%H%M%SZ).log"

ssh "${BF16_FLAT_NODE}" \
    "bash '${STAGED_RUN_ALL}' --target flat --output-dir '${OUTPUT_DIR}' --config '${STAGED_CONFIG}' ${MLC_BIN:+--mlc-bin \"${MLC_BIN}\"}" \
    2>&1 | tee "${LOG_FILE}"
