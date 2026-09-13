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

# --- CPU ISA treatment -------------------------------------------------------
# Both containers dispatch on CPUID at runtime and pick the widest ISA the host
# offers, so a profile asking for avx512 on a Sapphire Rapids node would still
# execute AMX kernels. The requested target is enforced here and attested just
# below, so the recorded cpu_isa reflects what actually ran.
ISA_SHIM=/lpor1/store_0/usr/crainic/isa-shim
BACKEND_NAME="$(awk '$1=="backend:"{print $2}' "${PROFILE}")"
ISA_TARGET="$(awk '$1=="cpu_isa:"{print $2}' "${PROFILE}")"

if [[ "${ISA_TARGET}" == "avx512" ]]; then
    if [[ "${BACKEND_NAME}" == "llamacpp" ]]; then
        # ggml ships one libggml-cpu-<arch>.so per ISA and loads the
        # highest-scoring one found in the executable directory. Masking
        # sapphirerapids (the only variant carrying AMX tile instructions) and
        # zen4 leaves cooperlake: same AVX-512 and AVX512-BF16, no AMX.
        export APPTAINER_BIND="${ISA_SHIM}/empty.so:/app/libggml-cpu-sapphirerapids.so,${ISA_SHIM}/empty.so:/app/libggml-cpu-zen4.so"
    else
        # vLLM 0.29.0 gates its own AMX kernels on
        # torch.cpu._is_amx_tile_supported() with no environment toggle, so a
        # sitecustomize shim reports AMX absent. oneDNN, which serves every
        # remaining dense GEMM, is capped at avx512_core_bf16.
        export APPTAINER_BIND="${ISA_SHIM}:/opt/isa-shim"
        export APPTAINERENV_PYTHONPATH=/opt/isa-shim
        export APPTAINERENV_ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16
    fi
fi

# Attest the ISA actually reachable under the environment the run will use, on
# this node, in this container. Written to the job log and kept next to it.
# Attestation is diagnostic only: `grep -m1` closes the pipe on its producer, so
# under `set -e -o pipefail` a successful probe would otherwise abort the job.
set +e
ATTEST="${LLM_BENCH_ISA_ATTESTATION:-/lpor1/store_0/usr/crainic/slurm-logs/isa-${SLURM_JOB_ID:-manual}.txt}"
{
    printf 'node=%s backend=%s isa_target=%s\n' "$(hostname)" "${BACKEND_NAME}" "${ISA_TARGET}"
    printf 'APPTAINER_BIND=%s\n' "${APPTAINER_BIND:-<unset>}"
    printf 'ONEDNN_MAX_CPU_ISA=%s\n' "${APPTAINERENV_ONEDNN_MAX_CPU_ISA:-<unset>}"
    if [[ "${BACKEND_NAME}" == "llamacpp" ]]; then
        apptainer exec --cleanenv --env LD_LIBRARY_PATH=/app \
            "${LLAMA_CPP_APPTAINER_IMAGE}" /app/llama-bench -h 2>&1 |
            grep -m1 load_backend
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
' 2>&1 | grep -E 'reported:|selected:|info,cpu,isa|matmul,brg' | sed 's/^onednn_verbose.*matmul,/onednn_matmul_impl: /;s/,undef.*//'
    fi
} 2>&1 | tee "${ATTEST}"
printf 'ISA attestation written to %s\n' "${ATTEST}"
set -e
# -----------------------------------------------------------------------------

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
