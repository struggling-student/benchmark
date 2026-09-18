#!/usr/bin/env bash
# Run the cache-campaign-equivalent BF16 matrix on a flat-mode node reserved
# for memory testing outside Slurm. Invoke on a cluster login node; this starts
# a detached, serial worker on the node named in the private cluster config.
#
#   run_cresco8_bf16_flat_campaign.sh [--dry-run|--foreground|--status]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO}/configs/cluster/cresco8.env"
[[ -r "${CONFIG}" ]] || { printf 'ERROR: unreadable config: %s\n' "${CONFIG}" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "${CONFIG}"
set +a

# The private, ignored config supplies site paths and the target node. Staging
# keeps a detached campaign independent of the login session's AFS token.
: "${BF16_FLAT_STAGE_REPO:?set BF16_FLAT_STAGE_REPO in the private cluster config}"
: "${BF16_FLAT_CAMPAIGN:?set BF16_FLAT_CAMPAIGN in the private cluster config}"
: "${BF16_FLAT_LOG_ROOT:?set BF16_FLAT_LOG_ROOT in the private cluster config}"
: "${BF16_FLAT_NODE:?set BF16_FLAT_NODE in the private cluster config}"
: "${BF16_FLAT_ISA_SHIM:?set BF16_FLAT_ISA_SHIM in the private cluster config}"
SOURCE_REPO="${REPO}"
STAGED_REPO="${BF16_FLAT_STAGE_REPO}"
CAMPAIGN="${BF16_FLAT_CAMPAIGN}"
LOG_ROOT="${BF16_FLAT_LOG_ROOT}"
MODEL="${REPO}/configs/models/llama32_1b.yaml"
NODE="${BF16_FLAT_NODE}"
ISA_SHIM="${BF16_FLAT_ISA_SHIM}"
TMUX_SESSION=llm-bf16-flat

PROFILES=(
    llamacpp_cpu_amx_hbm_flat
    llamacpp_cpu_avx512_hbm_flat
    vllm_cpu_amx_hbm_flat
    vllm_cpu_avx512_hbm_flat
)

# Run shortest cells first so useful coverage accumulates early and backend or
# placement failures surface before the long-context cells consume many hours.
WORKLOADS=(
    fixed_32_32
    fixed_512_8
    fixed_128_32
    fixed_256_32
    fixed_128_128
    fixed_512_32
    fixed_1024_32
    fixed_512_128
    fixed_1024_128
    fixed_30000_10000
)

MODE=launch
case "${1:-}" in
    --worker) MODE=worker ;;
    --dry-run) MODE=dry-run ;;
    --foreground) MODE=foreground ;;
    --status) MODE=status ;;
    "") ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac

short_hostname() {
    hostname -s
}

completed_cell() {
    local workload="$1"
    local profile="$2"
    local candidate
    while IFS= read -r candidate; do
        if grep -q '"status": "completed"' "${candidate}"; then
            return 0
        fi
    done < <(
        find "${CAMPAIGN}" -type f \
            -path "*-llama32_1b-${workload}-${profile}-bf16-*/summary.json" \
            -print 2>/dev/null
    )
    return 1
}

configure_treatment() {
    local profile="$1"
    unset APPTAINER_BIND APPTAINERENV_PYTHONPATH \
        APPTAINERENV_ONEDNN_MAX_CPU_ISA APPTAINERENV_VLLM_USE_V2_MODEL_RUNNER

    unset OMP_PROC_BIND OMP_PLACES
    export APPTAINERENV_HF_HUB_OFFLINE=1

    if [[ "${profile}" == vllm_* ]]; then
        # vLLM 0.29.0's V2 CPU runner is incompatible with its bundled Triton
        # CPU backend. This is the same setting used by the cache campaign.
        export APPTAINERENV_VLLM_USE_V2_MODEL_RUNNER=0
        # OMP_PROC_BIND/OMP_PLACES stay unset for vLLM: either one pins the
        # master thread to the first place when torch loads, collapsing the
        # process CPU mask to one CPU. vLLM reads os.sched_getaffinity() after
        # importing torch to plan per-rank cores, so it then assigns rank 0 a
        # single core and rank 1 none. Measured on this node: [1, 0] cores per
        # rank with these vars, [56, 56] without. See scripts/submit_cresco8.sh.
    else
        # ggml links libgomp and needs an explicit placement policy, otherwise
        # its threads migrate across sockets mid-kernel.
        export OMP_PROC_BIND=close
        export OMP_PLACES=cores
    fi

    if [[ "${profile}" == llamacpp_cpu_avx512_* ]]; then
        # Hide the AMX-bearing Sapphire Rapids ggml backend, leaving the
        # AVX-512 BF16 Cooper Lake backend as the widest available treatment.
        export APPTAINER_BIND="${ISA_SHIM}/empty.so:/app/libggml-cpu-sapphirerapids.so,${ISA_SHIM}/empty.so:/app/libggml-cpu-zen4.so"
    elif [[ "${profile}" == vllm_cpu_avx512_* ]]; then
        # Make vLLM report AMX unavailable and cap oneDNN at AVX-512 BF16.
        export APPTAINER_BIND="${ISA_SHIM}:/opt/isa-shim"
        export APPTAINERENV_PYTHONPATH=/opt/isa-shim
        export APPTAINERENV_ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16
    fi
}

attest_treatment() {
    local workload="$1"
    local profile="$2"
    local output="${LOG_ROOT}/isa-${workload}-${profile}.log"
    local image

    # Keep the same runtime-level ISA evidence as the cache-mode Slurm jobs.
    # The probe is diagnostic: a failed grep must not discard a benchmark cell.
    set +e
    {
        printf 'node=%s profile=%s\n' "$(hostname)" "${profile}"
        printf 'APPTAINER_BIND=%s\n' "${APPTAINER_BIND:-<unset>}"
        printf 'ONEDNN_MAX_CPU_ISA=%s\n' \
            "${APPTAINERENV_ONEDNN_MAX_CPU_ISA:-<unset>}"
        if [[ "${profile}" == llamacpp_* ]]; then
            if [[ "${profile}" == *_amx_* ]]; then
                image="${LLAMA_CPP_AMX_APPTAINER_IMAGE}"
            else
                image="${LLAMA_CPP_AVX512_APPTAINER_IMAGE}"
            fi
            apptainer exec --cleanenv --env LD_LIBRARY_PATH=/app \
                "${image}" /app/llama-bench -h 2>&1 | grep -m1 load_backend
        else
            apptainer exec --cleanenv \
                --env PYTHONPATH="${APPTAINERENV_PYTHONPATH:-}" \
                --env ONEDNN_VERBOSE=1 \
                --env ONEDNN_MAX_CPU_ISA="${APPTAINERENV_ONEDNN_MAX_CPU_ISA:-}" \
                "${VLLM_CPU_APPTAINER_IMAGE}" python3 -c '
import torch
from vllm.model_executor.layers.utils import check_cpu_sgl_kernel
print("torch_amx_tile_reported:", torch.cpu._is_amx_tile_supported())
print("torch_avx512_bf16_reported:", torch.cpu._is_avx512_bf16_supported())
print("vllm_sgl_amx_gemm_selected:", check_cpu_sgl_kernel(4096, 4096, torch.bfloat16))
a = torch.randn(512, 512, dtype=torch.bfloat16)
torch.nn.functional.linear(a, a)
' 2>&1 | grep -E 'reported:|selected:|info,cpu,isa|matmul,brg' |
                sed 's/^onednn_verbose.*matmul,/onednn_matmul_impl: /;s/,undef.*//'
        fi
    } >"${output}" 2>&1
    set -e
}

run_worker() {
    [[ "$(short_hostname)" == "${NODE}" ]] || {
        printf 'ERROR: worker must run on %s (found %s).\n' "${NODE}" "$(short_hostname)" >&2
        exit 2
    }
    mkdir -p "${CAMPAIGN}" "${LOG_ROOT}"

    exec 9>"${CAMPAIGN}/campaign.lock"
    flock -n 9 || {
        printf 'ERROR: another flat-mode campaign worker already holds %s.\n' \
            "${CAMPAIGN}/campaign.lock" >&2
        exit 1
    }

    printf 'Campaign start (UTC): %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf 'Node: %s\nResults: %s\n' "$(hostname)" "${CAMPAIGN}"
    local failures=0
    local workload profile cell_log rc
    for workload in "${WORKLOADS[@]}"; do
        for profile in "${PROFILES[@]}"; do
            if completed_cell "${workload}" "${profile}"; then
                printf 'SKIP completed: %s / %s\n' "${workload}" "${profile}"
                continue
            fi

            configure_treatment "${profile}"
            attest_treatment "${workload}" "${profile}"
            cell_log="${LOG_ROOT}/${workload}-${profile}.log"
            printf 'START %s / %s at %s; log=%s\n' \
                "${workload}" "${profile}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${cell_log}"
            set +e
            BENCH_RESULTS_ROOT="${CAMPAIGN}" \
                "${REPO}/scripts/run_benchmark.sh" \
                --config "${CONFIG}" \
                --model "${MODEL}" \
                --workload "${REPO}/configs/workloads/${workload}.yaml" \
                --profile "${REPO}/configs/profiles/${profile}.yaml" \
                --provider apptainer \
                --variant bf16 >"${cell_log}" 2>&1
            rc=$?
            set -e
            if [[ "${rc}" == 0 ]]; then
                printf 'PASS %s / %s at %s\n' \
                    "${workload}" "${profile}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            else
                failures=$((failures + 1))
                printf 'FAIL %s / %s rc=%d at %s; continuing\n' \
                    "${workload}" "${profile}" "${rc}" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            fi
        done
    done
    printf 'Campaign end (UTC): %s; failed cells: %d\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${failures}"
    [[ "${failures}" == 0 ]]
}

if [[ "${MODE}" == worker || "${MODE}" == foreground ]]; then
    run_worker
elif [[ "${MODE}" == dry-run ]]; then
    printf 'Node: %s\nResults: %s\n' "${NODE}" "${CAMPAIGN}"
    for workload in "${WORKLOADS[@]}"; do
        for profile in "${PROFILES[@]}"; do
            printf '%s %s bf16\n' "${workload}" "${profile}"
        done
    done
elif [[ "${MODE}" == status ]]; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        printf 'launcher=tmux:%s (active)\n' "${TMUX_SESSION}"
    else
        printf 'launcher=tmux:%s (inactive)\n' "${TMUX_SESSION}"
    fi
    ssh "${NODE}" "printf 'node='; hostname; pgrep -af '^bash .*/run_cresco8_bf16_flat_campaign.sh --worker$' || true; printf 'completed='; find '${CAMPAIGN}' -name summary.json -type f -exec grep -l '\"status\": \"completed\"' {} + 2>/dev/null | wc -l; tail -n 20 '${LOG_ROOT}/campaign.log' 2>/dev/null || true"
else
    command -v rsync >/dev/null || {
        printf 'ERROR: rsync is required to stage the detached worker on lpor1.\n' >&2
        exit 2
    }
    command -v tmux >/dev/null || {
        printf 'ERROR: tmux is required to retain the nested SSH session.\n' >&2
        exit 2
    }
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
        printf 'ERROR: tmux session %s already exists.\n' "${TMUX_SESSION}" >&2
        exit 1
    fi
    mkdir -p "${STAGED_REPO}"
    rsync -a --exclude=.git "${SOURCE_REPO}/" "${STAGED_REPO}/"
    mkdir -p "${LOG_ROOT}"
    # Keep the x001 -> hbm14 SSH session alive for the duration of the worker.
    # Its AFS token is required by the cluster Python installation backing the
    # benchmark virtual environment, even though the staged repo lives on lpor1.
    tmux new-session -d -s "${TMUX_SESSION}" \
        "ssh '${NODE}' \"bash '${STAGED_REPO}/scripts/run_cresco8_bf16_flat_campaign.sh' --worker >>'${LOG_ROOT}/campaign.log' 2>&1\""
    printf 'Started flat-mode BF16 campaign on %s (tmux session %s).\n' \
        "${NODE}" "${TMUX_SESSION}"
    printf 'Log: %s/campaign.log\nResults: %s\n' "${LOG_ROOT}" "${CAMPAIGN}"
fi
