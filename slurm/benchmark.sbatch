#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${BENCH_CONFIG:-${1:-}}"
[[ -n "${CONFIG_PATH}" && -r "${CONFIG_PATH}" ]] || {
    printf 'ERROR: set BENCH_CONFIG or pass a readable cluster environment file.\n' >&2
    exit 2
}
set -a
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set +a
[[ -n "${BENCH_REPO_ROOT:-}" && -r "${BENCH_REPO_ROOT}/scripts/run_benchmark.sh" ]] || {
    printf 'ERROR: BENCH_REPO_ROOT must identify the benchmark checkout.\n' >&2
    exit 2
}

MODEL="${MODEL_CONFIG:-${2:-}}"
WORKLOAD="${WORKLOAD_CONFIG:-${3:-}}"
PROFILE="${PROFILE_CONFIG:-${4:-}}"
PROVIDER="${BENCH_PROVIDER:-${5:-native}}"
VARIANT="${MODEL_VARIANT:-${6:-f16}}"
for name in MODEL WORKLOAD PROFILE; do
    [[ -n "${!name}" ]] || { printf 'ERROR: %s is required.\n' "${name}" >&2; exit 2; }
done

printf 'Slurm job ID: %s\n' "${SLURM_JOB_ID:-not-set}"
printf 'Hostname: %s\n' "$(hostname)"
printf 'Start date (UTC): %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
exec "${BENCH_REPO_ROOT}/scripts/run_benchmark.sh" \
    --config "${CONFIG_PATH}" \
    --model "${MODEL}" \
    --workload "${WORKLOAD}" \
    --profile "${PROFILE}" \
    --provider "${PROVIDER}" \
    --variant "${VARIANT}"
