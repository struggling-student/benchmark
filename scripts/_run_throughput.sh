#!/usr/bin/env bash
# Internal implementation shared by smoke and offline entry points.
set -euo pipefail

EXPECTED_TYPE="${1:?internal benchmark type is required}"
shift
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

if [[ -z "${EXPERIMENT}" && "${EXPECTED_TYPE}" == "smoke" ]]; then
    EXPERIMENT="${REPO_ROOT}/configs/experiments/llama32_1b_smoke.yaml"
fi
[[ -n "${EXPERIMENT}" ]] || { die "--experiment is required for ${EXPECTED_TYPE} benchmarks."; exit 2; }

load_cluster_config "${CONFIG_PATH}" || exit 1
activate_bench_venv || exit 1
python -m llm_bench.cli validate-config "${EXPERIMENT}" --benchmark-type "${EXPECTED_TYPE}" || exit 1

if [[ -n "${REQUESTED_RESULT_DIR}" ]]; then
    RESULT_DIR="${REQUESTED_RESULT_DIR}"
    RUN_ID="$(basename "${RESULT_DIR}")"
    mkdir -p "${RESULT_DIR}"
else
    make_result_dir "${EXPERIMENT}" "${EXPECTED_TYPE}" || exit 1
fi
enable_run_logging "${RESULT_DIR}"
printf 'Result directory: %s\n' "${RESULT_DIR}"

cp "${EXPERIMENT}" "${RESULT_DIR}/experiment.yaml"
if [[ ! -f "${RESULT_DIR}/metadata.json" ]]; then
    python -m llm_bench.cli init-run \
        --config "${EXPERIMENT}" \
        --run-dir "${RESULT_DIR}" \
        --benchmark-type "${EXPECTED_TYPE}" \
        --run-id "${RUN_ID}" \
        --repository "${REPO_ROOT}" || exit 1
fi
if [[ "${SKIP_SYSTEM_INFO:-0}" != "1" ]]; then
    "${SCRIPT_DIR}/collect_system_info.sh" --output-dir "${RESULT_DIR}"
fi
if [[ "${LLM_BENCH_PREFLIGHT_DONE:-0}" != "1" ]] && \
   ! "${SCRIPT_DIR}/preflight_check.sh" --config "${CONFIG_PATH}" --experiment "${EXPERIMENT}"; then
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" \
        --benchmark-type "${EXPECTED_TYPE}" --status failed \
        --warning "Environment preflight failed; the vLLM benchmark was not started." || true
    exit 1
fi

HELP_FILE="${RESULT_DIR}/vllm_bench_throughput_help.txt"
if vllm bench throughput --help=all > "${HELP_FILE}" 2>&1; then
    :
elif ! vllm bench throughput --help > "${HELP_FILE}" 2>&1; then
    warn "Installed vLLM has no usable 'bench throughput' command; see ${HELP_FILE}."
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" \
        --benchmark-type "${EXPECTED_TYPE}" --status failed \
        --warning "Installed vLLM does not expose a usable 'vllm bench throughput' command." || true
    exit 1
fi

for flag in --model --tokenizer --dataset-name --num-prompts \
    --dtype --tensor-parallel-size --seed --output-json; do
    if ! require_help_flag "${HELP_FILE}" "${flag}"; then
        python -m llm_bench.cli normalize-results \
            --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" \
            --benchmark-type "${EXPECTED_TYPE}" --status failed \
            --warning "The installed vLLM throughput CLI is missing required flag ${flag}; no benchmark was run." || true
        exit 1
    fi
done
if grep -Fq -- --random-input-len "${HELP_FILE}" && grep -Fq -- --random-output-len "${HELP_FILE}"; then
    INPUT_LENGTH_FLAG=--random-input-len
    OUTPUT_LENGTH_FLAG=--random-output-len
elif grep -Fq -- --input-len "${HELP_FILE}" && grep -Fq -- --output-len "${HELP_FILE}"; then
    INPUT_LENGTH_FLAG=--input-len
    OUTPUT_LENGTH_FLAG=--output-len
else
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" \
        --benchmark-type "${EXPECTED_TYPE}" --status failed \
        --warning "The installed vLLM throughput CLI has no supported input/output length flag pair." || true
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
MAX_MODEL_LEN="$(config_value "${EXPERIMENT}" max_model_len)"
GPU_MEMORY_UTILIZATION="$(config_value "${EXPERIMENT}" gpu_memory_utilization)"
REPETITIONS="$(config_value "${EXPERIMENT}" repetitions)"
WARMUP_RUNS="$(config_value "${EXPERIMENT}" warmup_runs)"
TELEMETRY_INTERVAL_MS="$(config_value "${EXPERIMENT}" telemetry_interval_ms)"

build_command() {
    local output_json="$1"
    CMD=(
        vllm bench throughput
        --model "${MODEL_ID}"
        --tokenizer "${TOKENIZER_ID}"
        --dataset-name random
        --num-prompts "${NUMBER_OF_PROMPTS}"
        "${INPUT_LENGTH_FLAG}" "${INPUT_LENGTH}"
        "${OUTPUT_LENGTH_FLAG}" "${OUTPUT_LENGTH}"
        --dtype "${DTYPE}"
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
        --seed "${SEED}"
        --output-json "${output_json}"
    )
    if ! is_null_value "${MODEL_REVISION}"; then
        require_help_flag "${HELP_FILE}" --revision || return 1
        CMD+=(--revision "${MODEL_REVISION}")
    fi
    if ! is_null_value "${QUANTIZATION}"; then
        require_help_flag "${HELP_FILE}" --quantization || return 1
        CMD+=(--quantization "${QUANTIZATION}")
    fi
    if ! is_null_value "${GPU_MEMORY_UTILIZATION}"; then
        require_help_flag "${HELP_FILE}" --gpu-memory-utilization || return 1
        CMD+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
    fi
    if ! is_null_value "${MAX_MODEL_LEN}"; then
        require_help_flag "${HELP_FILE}" --max-model-len || return 1
        CMD+=(--max-model-len "${MAX_MODEL_LEN}")
    fi
}

mkdir -p "${RESULT_DIR}/warmup"
benchmark_rc=0
error_args=()
warning_args=(
    --warning "Throughput telemetry covers each measured vLLM CLI process, including initialization/model loading and inference; it excludes warm-ups and is not whole-node energy."
    --warning "Peak nvidia-smi memory is total memory used/reserved on the visible allocated GPU(s), not an isolated model-weight footprint or minimum memory requirement."
)
raw_args=()
telemetry_args=()
TELEMETRY_PID=""
cleanup_telemetry() {
    stop_background_process "${TELEMETRY_PID}"
    TELEMETRY_PID=""
}
handle_signal() {
    local signal_exit="$1"
    trap - INT TERM
    cleanup_telemetry
    # Bash 3.2 treats an empty array expansion as unbound under `set -u`.
    set +u
    python -m llm_bench.cli normalize-results \
        --config "${EXPERIMENT}" --run-dir "${RESULT_DIR}" \
        --benchmark-type "${EXPECTED_TYPE}" --status failed \
        "${raw_args[@]}" "${telemetry_args[@]}" \
        --error "Benchmark interrupted by signal (exit ${signal_exit})." \
        --warning "Benchmark interrupted by a signal; completed measured outputs were retained." || true
    set -u
    exit "${signal_exit}"
}
trap cleanup_telemetry EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
for ((run = 1; run <= WARMUP_RUNS; run++)); do
    warmup_output="${RESULT_DIR}/warmup/raw_vllm_output.warmup-$(printf '%03d' "${run}").json"
    printf 'Warm-up run %d/%d (excluded from measured results)\n' "${run}" "${WARMUP_RUNS}"
    if ! build_command "${warmup_output}" || ! "${CMD[@]}"; then
        benchmark_rc=1
        error_args=(--error "Warm-up run ${run} failed.")
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
            raw_output="${RESULT_DIR}/raw_vllm_output.json"
            telemetry_output="${RESULT_DIR}/gpu_telemetry.csv"
        else
            raw_output="${RESULT_DIR}/raw_vllm_output${suffix}.json"
            telemetry_output="${RESULT_DIR}/gpu_telemetry${suffix}.csv"
        fi
        printf 'Measured repetition %d/%d\n' "${run}" "${REPETITIONS}"
        build_command "${raw_output}" || { benchmark_rc=1; error_args=(--error "Could not construct measured repetition ${run}."); warning_args+=(--warning "Could not construct measured repetition ${run} for the installed vLLM CLI."); break; }

        "${SCRIPT_DIR}/collect_gpu_telemetry.sh" \
            --output "${telemetry_output}" \
            --interval-ms "${TELEMETRY_INTERVAL_MS}" &
        TELEMETRY_PID=$!
        repetition_start="$(date +%s)"
        if "${CMD[@]}"; then
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
            error_args=(--error "Measured repetition ${run} failed with vLLM exit code ${command_rc} or missing raw output.")
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
    --config "${EXPERIMENT}" \
    --run-dir "${RESULT_DIR}" \
    --benchmark-type "${EXPECTED_TYPE}" \
    --status "${status}" \
    "${raw_args[@]}" \
    "${telemetry_args[@]}" \
    "${duration_args[@]}" \
    "${error_args[@]}" \
    "${warning_args[@]}" || normalize_rc=$?
set -u

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
trap - EXIT INT TERM
cleanup_telemetry
printf 'Completed %s benchmark: %s\n' "${EXPECTED_TYPE}" "${RESULT_DIR}"
