#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG_PATH="${BENCH_CONFIG:-}"
EXPERIMENT=""
REQUESTED_RESULT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_PATH="${2:?--config requires a file}"; shift 2 ;;
        --experiment) EXPERIMENT="${2:?--experiment requires a YAML file}"; shift 2 ;;
        --result-dir) REQUESTED_RESULT_DIR="${2:?--result-dir requires a directory}"; shift 2 ;;
        -h|--help)
            printf 'Usage: %s --config FILE --experiment EXPERIMENT.yaml [--result-dir DIR]\n' "$0"
            exit 0
            ;;
        *) die "Unknown argument: $1"; exit 2 ;;
    esac
done
[[ -n "${EXPERIMENT}" ]] || { die "--experiment is required for serving benchmarks."; exit 2; }

load_cluster_config "${CONFIG_PATH}" || exit 1
activate_bench_venv || exit 1
python -m llm_bench.cli validate-config "${EXPERIMENT}" --benchmark-type serving || exit 1

if [[ -n "${REQUESTED_RESULT_DIR}" ]]; then
    RESULT_DIR="${REQUESTED_RESULT_DIR}"
    RUN_ID="$(basename "${RESULT_DIR}")"
    mkdir -p "${RESULT_DIR}"
else
    make_result_dir "${EXPERIMENT}" serving || exit 1
fi
enable_run_logging "${RESULT_DIR}"
printf 'Result directory: %s\n' "${RESULT_DIR}"

cp "${EXPERIMENT}" "${RESULT_DIR}/experiment.yaml"
if [[ ! -f "${RESULT_DIR}/metadata.json" ]]; then
    python -m llm_bench.cli init-run \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --run-id "${RUN_ID}" \
        --repository "${REPO_ROOT}" || exit 1
fi
if [[ "${SKIP_SYSTEM_INFO:-0}" != "1" ]]; then
    "${SCRIPT_DIR}/collect_system_info.sh" --output-dir "${RESULT_DIR}"
fi
if [[ "${LLM_BENCH_PREFLIGHT_DONE:-0}" != "1" ]] && \
   ! "${SCRIPT_DIR}/preflight_check.sh" --config "${CONFIG_PATH}" --experiment "${EXPERIMENT}"; then
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed --warning "Environment preflight failed; the vLLM server was not started." || true
    exit 1
fi

SERVER_HELP="${RESULT_DIR}/vllm_serve_help.txt"
CLIENT_HELP="${RESULT_DIR}/vllm_bench_serve_help.txt"
if vllm serve --help=all > "${SERVER_HELP}" 2>&1; then
    :
elif ! vllm serve --help > "${SERVER_HELP}" 2>&1; then
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed --warning "The installed vLLM does not expose a usable serve command; inspect recorded help." || true
    exit 1
fi
if ! vllm bench serve --help > "${CLIENT_HELP}" 2>&1; then
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed --warning "The installed vLLM does not expose a usable bench serve command; inspect recorded help." || true
    exit 1
fi

for flag in --host --port --dtype --tensor-parallel-size --generation-config; do
    require_help_flag "${SERVER_HELP}" "${flag}" || {
        python -m llm_bench.cli normalize-results \
            --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
            --status failed --warning "The installed vLLM server CLI is missing ${flag}." || true
        exit 1
    }
done
for flag in --backend --model --tokenizer --endpoint --dataset-name --num-prompts \
    --request-rate --max-concurrency --seed \
    --temperature --top-p --ignore-eos \
    --save-result --result-dir --result-filename; do
    require_help_flag "${CLIENT_HELP}" "${flag}" || {
        python -m llm_bench.cli normalize-results \
            --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
            --status failed --warning "The installed vLLM serving-benchmark CLI is missing ${flag}." || true
        exit 1
    }
done
if grep -Fq -- --random-input-len "${CLIENT_HELP}" && grep -Fq -- --random-output-len "${CLIENT_HELP}"; then
    INPUT_LENGTH_FLAG=--random-input-len
    OUTPUT_LENGTH_FLAG=--random-output-len
elif grep -Fq -- --input-len "${CLIENT_HELP}" && grep -Fq -- --output-len "${CLIENT_HELP}"; then
    INPUT_LENGTH_FLAG=--input-len
    OUTPUT_LENGTH_FLAG=--output-len
else
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed --warning "The installed vLLM serving-benchmark CLI has no supported input/output length flag pair." || true
    exit 1
fi

MODEL_ID="$(config_value "${EXPERIMENT}" model_id)"
MODEL_REVISION="$(config_value "${EXPERIMENT}" model_revision)"
TOKENIZER_ID="$(config_value "${EXPERIMENT}" tokenizer_id)"
DTYPE="$(config_value "${EXPERIMENT}" dtype)"
QUANTIZATION="$(config_value "${EXPERIMENT}" quantization)"
TENSOR_PARALLEL_SIZE="$(config_value "${EXPERIMENT}" tensor_parallel_size)"
SEED="$(config_value "${EXPERIMENT}" seed)"
NUMBER_OF_PROMPTS="$(config_value "${EXPERIMENT}" number_of_prompts)"
INPUT_LENGTH="$(config_value "${EXPERIMENT}" input_length)"
OUTPUT_LENGTH="$(config_value "${EXPERIMENT}" output_length)"
GENERATION_CONFIG="$(config_value "${EXPERIMENT}" generation_config)"
TEMPERATURE="$(config_value "${EXPERIMENT}" temperature)"
TOP_P="$(config_value "${EXPERIMENT}" top_p)"
IGNORE_EOS="$(config_value "${EXPERIMENT}" ignore_eos)"
REQUEST_RATE="$(config_value "${EXPERIMENT}" request_rate)"
MAXIMUM_CONCURRENCY="$(config_value "${EXPERIMENT}" maximum_concurrency)"
GPU_MEMORY_UTILIZATION="$(config_value "${EXPERIMENT}" gpu_memory_utilization)"
REPETITIONS="$(config_value "${EXPERIMENT}" repetitions)"
WARMUP_RUNS="$(config_value "${EXPERIMENT}" warmup_runs)"
TELEMETRY_INTERVAL_MS="$(config_value "${EXPERIMENT}" telemetry_interval_ms)"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT_SECONDS:-600}"

fail_before_server() {
    local detail="$1"
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed --warning "${detail}" || true
    die "${detail}"
    exit 1
}
[[ "${PORT}" =~ ^[1-9][0-9]*$ ]] || fail_before_server "VLLM_PORT must be a positive integer."
[[ "${STARTUP_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || \
    fail_before_server "VLLM_STARTUP_TIMEOUT_SECONDS must be a positive integer."

SERVER_CMD=(
    vllm serve "${MODEL_ID}"
    --host "${HOST}"
    --port "${PORT}"
    --dtype "${DTYPE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --generation-config "${GENERATION_CONFIG}"
)
if ! is_null_value "${GPU_MEMORY_UTILIZATION}"; then
    require_help_flag "${SERVER_HELP}" --gpu-memory-utilization || \
        fail_before_server "Installed vLLM cannot apply configured gpu_memory_utilization."
    SERVER_CMD+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if ! is_null_value "${TOKENIZER_ID}"; then
    require_help_flag "${SERVER_HELP}" --tokenizer || \
        fail_before_server "Installed vLLM cannot apply configured tokenizer_id."
    SERVER_CMD+=(--tokenizer "${TOKENIZER_ID}")
fi
if ! is_null_value "${MODEL_REVISION}"; then
    require_help_flag "${SERVER_HELP}" --revision || \
        fail_before_server "Installed vLLM cannot apply configured model_revision."
    SERVER_CMD+=(--revision "${MODEL_REVISION}")
fi
if ! is_null_value "${QUANTIZATION}"; then
    require_help_flag "${SERVER_HELP}" --quantization || \
        fail_before_server "Installed vLLM cannot apply configured quantization."
    SERVER_CMD+=(--quantization "${QUANTIZATION}")
fi

build_client_command() {
    local result_directory="$1"
    local result_filename="$2"
    CLIENT_CMD=(
        vllm bench serve
        --backend vllm
        --model "${MODEL_ID}"
        --tokenizer "${TOKENIZER_ID}"
        --endpoint /v1/completions
        --dataset-name random
        --num-prompts "${NUMBER_OF_PROMPTS}"
        "${INPUT_LENGTH_FLAG}" "${INPUT_LENGTH}"
        "${OUTPUT_LENGTH_FLAG}" "${OUTPUT_LENGTH}"
        --request-rate "${REQUEST_RATE}"
        --max-concurrency "${MAXIMUM_CONCURRENCY}"
        --seed "${SEED}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --ignore-eos
        --save-result
        --result-dir "${result_directory}"
        --result-filename "${result_filename}"
    )
    if grep -Fq -- --base-url "${CLIENT_HELP}"; then
        CLIENT_CMD+=(--base-url "http://${HOST}:${PORT}")
    else
        require_help_flag "${CLIENT_HELP}" --host || return 1
        require_help_flag "${CLIENT_HELP}" --port || return 1
        CLIENT_CMD+=(--host "${HOST}" --port "${PORT}")
    fi
    if grep -Fq -- --percentile-metrics "${CLIENT_HELP}" && \
       grep -Fq -- --metric-percentiles "${CLIENT_HELP}"; then
        CLIENT_CMD+=(
            --percentile-metrics ttft,tpot,itl,e2el
            --metric-percentiles 95,99
        )
    fi
}

SERVER_PID=""
SERVER_PGID=""
TELEMETRY_PID=""
raw_args=()
telemetry_args=()
MODEL_LOAD_ARGS=()
cleanup() {
    stop_background_process "${TELEMETRY_PID}"
    TELEMETRY_PID=""
    stop_process_group "${SERVER_PID}" "${SERVER_PGID}"
    SERVER_PID=""
    SERVER_PGID=""
}
handle_signal() {
    local signal_exit="$1"
    trap - INT TERM
    cleanup
    # Bash 3.2 treats an empty array expansion as unbound under `set -u`.
    set +u
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed "${raw_args[@]}" "${telemetry_args[@]}" "${MODEL_LOAD_ARGS[@]}" \
        --error "Serving benchmark interrupted by signal (exit ${signal_exit})." \
        --warning "Serving benchmark interrupted by a signal; the server was terminated and completed outputs were retained." || true
    set -u
    exit "${signal_exit}"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

if curl --silent --fail --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed \
        --warning "A server already answered on the configured host/port; refused to benchmark an unidentified process." || true
    die "${HOST}:${PORT} already has a healthy server. Choose an unused VLLM_PORT."
    exit 1
fi

printf 'Starting vLLM server on %s:%s; logs: %s\n' "${HOST}" "${PORT}" "${RESULT_DIR}/server.log"
server_start="$(date +%s)"
SESSION_READY="${RESULT_DIR}/.server_process_group_ready"
rm -f "${SESSION_READY}"
python -c 'import os, pathlib, sys; os.setsid(); pathlib.Path(sys.argv[1]).touch(); os.execvp(sys.argv[2], sys.argv[2:])' \
    "${SESSION_READY}" "${SERVER_CMD[@]}" > "${RESULT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
SERVER_PGID="${SERVER_PID}"
session_ready=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -e "${SESSION_READY}" ]]; then
        session_ready=1
        rm -f "${SESSION_READY}"
        break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null || break
    sleep 0.1
done
if [[ "${session_ready}" != "1" ]]; then
    stop_background_process "${SERVER_PID}"
    SERVER_PID=""
    SERVER_PGID=""
    rm -f "${SESSION_READY}"
    fail_before_server "Could not isolate the vLLM server in a dedicated process group; no benchmark was run."
fi
deadline=$((server_start + STARTUP_TIMEOUT))
server_ready=0
while [[ "$(date +%s)" -lt "${deadline}" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
    fi
    if curl --silent --show-error --fail --max-time 5 \
        "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        server_ready=1
        break
    fi
    sleep 2
done
server_ready_time="$(date +%s)"
model_load_time=$((server_ready_time - server_start))
if [[ "${server_ready}" != "1" ]]; then
    warn "vLLM server did not become healthy within ${STARTUP_TIMEOUT}s or exited early."
    tail -n 80 "${RESULT_DIR}/server.log" >&2 || true
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed \
        --warning "Server health check failed; model-load time is unavailable." || true
    exit 1
fi
printf 'Server is healthy after %s seconds.\n' "${model_load_time}"
if ! curl --silent --show-error --fail --max-time 5 \
    "http://${HOST}:${PORT}/v1/models" > "${RESULT_DIR}/server_models.json" || \
   ! python -c 'import json,sys; expected=sys.argv[2]; data=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if any(item.get("id") == expected for item in data.get("data", [])) else 1)' \
       "${RESULT_DIR}/server_models.json" "${MODEL_ID}"; then
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
        --status failed \
        --warning "The healthy server did not report the configured model identity; no client benchmark was run." || true
    die "Server model identity did not match ${MODEL_ID}; inspect server.log and server_models.json."
    exit 1
fi
MODEL_LOAD_ARGS=(--model-load-time-seconds "${model_load_time}")

mkdir -p "${RESULT_DIR}/warmup"
benchmark_rc=0
error_args=()
warning_args=(
    --warning "Serving model-load time is wall time to health readiness and includes server startup overhead."
    --warning "Serving telemetry excludes server/model startup and covers each measured client workload while the server is loaded; it is not whole-node energy."
    --warning "Peak nvidia-smi memory is total memory used/reserved on the visible allocated GPU(s), not an isolated model-weight footprint or minimum memory requirement."
)
for ((run = 1; run <= WARMUP_RUNS; run++)); do
    filename="raw_vllm_output.warmup-$(printf '%03d' "${run}").json"
    printf 'Warm-up run %d/%d (excluded from measured results)\n' "${run}" "${WARMUP_RUNS}"
    if ! build_client_command "${RESULT_DIR}/warmup" "${filename}" || ! "${CLIENT_CMD[@]}"; then
        benchmark_rc=1
        error_args=(--error "Serving warm-up run ${run} failed.")
        warning_args+=(--warning "Warm-up run ${run} failed; measured repetitions were not started.")
        printf '%s\n' "failed" > "${RESULT_DIR}/warmup/warmup-$(printf '%03d' "${run}").status.txt"
        break
    fi
done

measured_duration_total=0
completed_repetitions=0
if [[ "${benchmark_rc}" == "0" ]]; then
    for ((run = 1; run <= REPETITIONS; run++)); do
        suffix=".repetition-$(printf '%03d' "${run}")"
        if [[ "${run}" == "1" ]]; then
            filename="raw_vllm_output.json"
            telemetry_output="${RESULT_DIR}/gpu_telemetry.csv"
        else
            filename="raw_vllm_output${suffix}.json"
            telemetry_output="${RESULT_DIR}/gpu_telemetry${suffix}.csv"
        fi
        raw_output="${RESULT_DIR}/${filename}"
        printf 'Measured repetition %d/%d\n' "${run}" "${REPETITIONS}"
        build_client_command "${RESULT_DIR}" "${filename}" || { benchmark_rc=1; error_args=(--error "Could not construct measured repetition ${run}."); warning_args+=(--warning "Could not construct measured repetition ${run} for the installed vLLM CLI."); break; }

        "${SCRIPT_DIR}/collect_gpu_telemetry.sh" \
            --output "${telemetry_output}" --interval-ms "${TELEMETRY_INTERVAL_MS}" &
        TELEMETRY_PID=$!
        repetition_start="$(date +%s)"
        if "${CLIENT_CMD[@]}"; then
            command_rc=0
        else
            command_rc=$?
        fi
        repetition_end="$(date +%s)"
        stop_background_process "${TELEMETRY_PID}"
        TELEMETRY_PID=""
        measured_duration_total=$((measured_duration_total + repetition_end - repetition_start))

        if [[ "${command_rc}" != "0" || ! -s "${raw_output}" ]]; then
            benchmark_rc="${command_rc}"
            [[ "${benchmark_rc}" != "0" ]] || benchmark_rc=1
            error_args=(--error "Measured serving repetition ${run} failed with vLLM exit code ${command_rc} or missing raw output.")
            warning_args+=(--warning "Measured repetition ${run} failed or did not create its raw JSON output; it was retained and the run is failed.")
            printf 'exit_code=%s\n' "${command_rc}" > "${RESULT_DIR}/repetition-$(printf '%03d' "${run}").status.txt"
            break
        fi
        if [[ -s "${telemetry_output}" ]]; then
            telemetry_args+=(--telemetry "${telemetry_output}")
        fi
        raw_args+=(--raw-output "${raw_output}")
        completed_repetitions=$((completed_repetitions + 1))
    done
fi

status=completed
if [[ "${benchmark_rc}" != "0" || "${completed_repetitions}" != "${REPETITIONS}" ]]; then
    status=failed
fi
duration_args=()
if [[ "${completed_repetitions}" -gt 0 ]]; then
    mean_duration="$(awk -v total="${measured_duration_total}" -v count="${completed_repetitions}" 'BEGIN { printf "%.6f", total / count }')"
    duration_args=(--duration-seconds "${mean_duration}")
fi

normalize_rc=0
set +u
python -m llm_bench.cli normalize-results \
    --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" --benchmark-type serving \
    --status "${status}" "${MODEL_LOAD_ARGS[@]}" \
    "${raw_args[@]}" "${telemetry_args[@]}" "${duration_args[@]}" \
    "${error_args[@]}" "${warning_args[@]}" || normalize_rc=$?
set -u

trap - EXIT INT TERM
cleanup
if [[ "${benchmark_rc}" != "0" ]]; then
    exit "${benchmark_rc}"
fi
if [[ "${normalize_rc}" != "0" ]]; then
    die "Benchmark completed but result normalization failed. Raw outputs remain in ${RESULT_DIR}."
    exit "${normalize_rc}"
fi
final_status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "${RESULT_DIR}/summary.json")"
if [[ "${final_status}" != "completed" ]]; then
    die "vLLM returned successfully, but one or more measured outputs could not be normalized; summary status is ${final_status}."
    exit 1
fi
printf 'Completed serving benchmark: %s\n' "${RESULT_DIR}"
