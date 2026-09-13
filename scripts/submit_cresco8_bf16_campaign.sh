#!/usr/bin/env bash
# BF16 cross-backend campaign: llama.cpp and vLLM CPU, AMX and AVX-512, on the
# Xeon Max cache-mode nodes, over every shipped literature workload.
#
#   usage: submit_cresco8_bf16_campaign.sh [--dependency AFTEROK_SPEC] [--dry-run]
#
# One variant (bf16) and one model (llama32_1b) throughout, so backend and ISA
# are the only things that move. Results land under their own tree instead of
# the shared RESULTS_ROOT so the campaign can be analysed as a unit.
set -euo pipefail
REPO=/afs/enea.it/por/user/crainic/dev/benchmark
CAMPAIGN=/lpor1/store_0/usr/crainic/results/bf16-xbackend
MODEL="${REPO}/configs/models/llama32_1b.yaml"
PARTITION=cresco8_hbm
# hbm14 has been DOWN since 2026-09-10; hbm[15-16] are the one-hour debug pair.
NODELIST="cresco8-hbm[01-13]"

DEPENDENCY=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dependency) DEPENDENCY="${2:?}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) printf "Unknown argument: %s\n" "$1" >&2; exit 2 ;;
    esac
done

PROFILES=(
    llamacpp_cpu_amx_hbm_cache
    llamacpp_cpu_avx512_hbm_cache
    vllm_cpu_amx_hbm_cache
    vllm_cpu_avx512_hbm_cache
)

# Walltimes are sized from the 2026-09-13 fixed_512_128 cells, where the slowest
# cell (llama.cpp AVX-512) took 44 min for 1 warmup + 3 repetitions.
#
# The QOS caps this user at 5 nodes, so the queue drains 5 jobs at a time and
# submission order decides what runs. The long-context shape goes LAST: submitted
# first, its four cells took 4 of the 5 slots for their full 23 h walltime and
# left the other 36 cells sharing one slot. Shortest-first fills the campaign in.
declare -a WORKLOADS=(
    "fixed_32_32       06:00:00"
    "fixed_512_8       06:00:00"
    "fixed_128_32      06:00:00"
    "fixed_256_32      06:00:00"
    "fixed_128_128     06:00:00"
    "fixed_512_32      06:00:00"
    "fixed_1024_32     08:00:00"
    "fixed_512_128     12:00:00"
    "fixed_1024_128    12:00:00"
    "fixed_30000_10000 23:00:00"
)

mkdir -p "${CAMPAIGN}"
count=0
for entry in "${WORKLOADS[@]}"; do
    read -r workload walltime <<<"${entry}"
    for profile in "${PROFILES[@]}"; do
        command=(
            "${REPO}/scripts/submit_cresco8.sh"
            --model "${MODEL}"
            --workload "${REPO}/configs/workloads/${workload}.yaml"
            --profile "${REPO}/configs/profiles/${profile}.yaml"
            --variant bf16
            --partition "${PARTITION}"
            --nodelist "${NODELIST}"
            --time "${walltime}"
            --results-root "${CAMPAIGN}"
        )
        [[ -n "${DEPENDENCY}" ]] && command+=(--dependency "${DEPENDENCY}")
        if [[ "${DRY_RUN}" == "1" ]]; then
            printf "%s\n" "${command[*]}"
        else
            "${command[@]}"
        fi
        count=$((count + 1))
    done
done
printf "%d job(s) for %s\n" "${count}" "${CAMPAIGN}"
