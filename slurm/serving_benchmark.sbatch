#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${BENCH_CONFIG:-${1:-}}"
[[ -n "${CONFIG_PATH}" && -r "${CONFIG_PATH}" ]] || { printf 'ERROR: set BENCH_CONFIG to a readable cluster environment file.\n' >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set +a
[[ -n "${BENCH_REPO_ROOT:-}" && "${BENCH_REPO_ROOT}" == /* && -r "${BENCH_REPO_ROOT}/scripts/_common.sh" ]] || { printf 'ERROR: BENCH_REPO_ROOT must be an absolute readable checkout path in BENCH_CONFIG.\n' >&2; exit 2; }
REPO_ROOT="${BENCH_REPO_ROOT}"
# shellcheck source=scripts/_common.sh
source "${REPO_ROOT}/scripts/_common.sh"

EXPERIMENT="${EXPERIMENT_CONFIG:-${2:-}}"
[[ -n "${EXPERIMENT}" ]] || { die "Set EXPERIMENT_CONFIG or pass a serving experiment as the second script argument."; exit 2; }
load_cluster_config "${CONFIG_PATH}"
activate_bench_venv
python -m llm_bench.cli validate-config "${EXPERIMENT}" --benchmark-type serving
make_result_dir "${EXPERIMENT}" serving
enable_run_logging "${RESULT_DIR}"

printf 'Slurm job ID: %s\n' "${SLURM_JOB_ID:-not-set}"
printf 'Hostname: %s\n' "$(hostname)"
printf 'Start date (UTC): %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf 'Result directory: %s\n' "${RESULT_DIR}"

cp "${EXPERIMENT}" "${RESULT_DIR}/experiment.yaml"
python -m llm_bench.cli init-run \
    --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
    --run-id "${RUN_ID}" \
    --repository "${REPO_ROOT}"

finalize_if_running() {
    local exit_code=$?
    trap - EXIT
    if [[ "${exit_code}" != "0" && -f "${RESULT_DIR}/summary.json" ]]; then
        summary_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))' "${RESULT_DIR}/summary.json" 2>/dev/null || true)"
        if [[ "${summary_status}" == "running" ]]; then
            python -m llm_bench.cli normalize-results \
                --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
                --status failed --warning "Slurm job ended before the serving benchmark could finalize." || true
        fi
    fi
    exit "${exit_code}"
}
trap finalize_if_running EXIT

"${REPO_ROOT}/scripts/collect_system_info.sh" --output-dir "${RESULT_DIR}"
"${REPO_ROOT}/scripts/preflight_check.sh" --config "${CONFIG_PATH}" --experiment "${EXPERIMENT}"
export LLM_BENCH_PREFLIGHT_DONE=1
export SKIP_SYSTEM_INFO=1
"${REPO_ROOT}/scripts/run_serving_benchmark.sh" \
    --config "${CONFIG_PATH}" --experiment "${EXPERIMENT}" --result-dir "${RESULT_DIR}"
