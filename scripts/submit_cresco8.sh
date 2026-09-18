#!/usr/bin/env bash
# Submit a benchmark on the CRESCO8 Xeon CPU Max HBM nodes.
#
#   scripts/submit_cresco8.sh --profile PROFILE.yaml --variant VARIANT [options]
#   Options include --model, --workload, --partition, --nodelist, --time,
#   --cpus-per-task, --dependency, --log-level, --results-root.
#
# Defaults target the verified cache-mode node cresco8-hbm15 in the one-hour
# debug partition. Discovered with sinfo/scontrol: cresco8_hbm_dbg holds
# cresco8-hbm[15-16] (1 h), cresco8_hbm holds cresco8-hbm[01-14] (1 day).
set -euo pipefail
REPO=/afs/enea.it/por/user/crainic/dev/benchmark
CONFIG="${REPO}/configs/cluster/cresco8.env"
LOGS=/lpor1/store_0/usr/crainic/slurm-logs

PROFILE=""
VARIANT=""
MODEL="${REPO}/configs/models/llama32_1b.yaml"
WORKLOAD="${REPO}/configs/workloads/fixed_32_32.yaml"
PARTITION="cresco8_hbm_dbg"
NODELIST="cresco8-hbm15"
TIME_LIMIT="01:00:00"
# Slurm hands a task one CPU unless --cpus-per-task says otherwise, and
# --exclusive does not change that: the task cgroup is capped at CPU 0 and
# every backend then runs single-threaded regardless of its thread_count.
# The Xeon CPU Max 9480 HBM nodes have 112 cores and no SMT.
CPUS_PER_TASK=112
DEPENDENCY=""
LOG_LEVEL=""
RESULTS_ROOT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)   PROFILE="${2:?}";   shift 2 ;;
        --variant)   VARIANT="${2:?}";   shift 2 ;;
        --model)     MODEL="${2:?}";     shift 2 ;;
        --workload)  WORKLOAD="${2:?}";  shift 2 ;;
        --partition) PARTITION="${2:?}"; shift 2 ;;
        --nodelist)  NODELIST="${2:?}";  shift 2 ;;
        --time)      TIME_LIMIT="${2:?}"; shift 2 ;;
        --cpus-per-task) CPUS_PER_TASK="${2:?}"; shift 2 ;;
        --dependency) DEPENDENCY="${2:?}"; shift 2 ;;
        --log-level) LOG_LEVEL="${2:?}"; shift 2 ;;
        --results-root) RESULTS_ROOT_OVERRIDE="${2:?}"; shift 2 ;;
        -h|--help)   sed -n '2,8p' "$0"; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "${PROFILE}" && -n "${VARIANT}" ]] || {
    printf 'ERROR: --profile and --variant are required.\n' >&2; exit 2; }
[[ -r "${PROFILE}" ]] || { printf 'ERROR: unreadable profile: %s\n' "${PROFILE}" >&2; exit 2; }
mkdir -p "${LOGS}"

PROFILE_NAME="$(basename "${PROFILE}" .yaml)"
WORKLOAD_NAME="$(basename "${WORKLOAD}" .yaml)"

# Container environment is injected through APPTAINERENV_* so it survives the
# provider's `apptainer exec --cleanenv`.
#
#   HF_HUB_OFFLINE=1  CRESCO8 compute nodes have no outbound network. Without it
#                     every Hugging Face file lookup waits out an HTTP timeout
#                     before falling back to the local cache.
#   OMP_PROC_BIND     ggml links libgomp. With no binding policy the OpenMP
#   OMP_PLACES        threads migrate across both sockets for the whole run, so
#                     a thread's working set goes remote mid-kernel. Pinning one
#                     thread per core lifted AMX prefill from 292 to 518 tok/s
#                     on its own and, combined with the profile's
#                     memory_policy/load_mode, brought between-launch CV under
#                     5 % on both prefill and decode. The provider forwards
#                     these through its own allowlist, so they are exported for
#                     the native provider; APPTAINERENV_* covers the container.
#                     llama.cpp ONLY -- see the vLLM block below.
CONTAINER_ENV="APPTAINERENV_HF_HUB_OFFLINE=1"

if grep -q '^backend:[[:space:]]*vllm' "${PROFILE}"; then
    # Do NOT set OMP_PROC_BIND/OMP_PLACES for vLLM. Either one makes the OpenMP
    # runtime pin the master thread to the first place as soon as torch loads,
    # which collapses the process CPU mask to a single CPU. vLLM plans its own
    # per-rank core assignment from os.sched_getaffinity() *after* importing
    # torch (vllm/utils/cpu_resource_utils.py:get_allowed_cpu_list), so it then
    # sees one allowed CPU and hands rank 0 a single core and rank 1 none:
    #
    #   Selected CPU core number (1) should be greater than reserved (1)
    #   local_rank=0, core ids=[0]
    #   local_rank=1, core ids=[]
    #
    # Measured on cresco8-hbm14: with these vars the planner yields [1, 0] cores
    # per rank, without them [56, 56]. Nothing errors -- the job just runs on one
    # core. vLLM does its own binding, so it needs no OpenMP placement policy.
    #
    # The V2 CPU model runner in the v0.29.0 vLLM CPU image prepares prefill
    # inputs with a Triton kernel whose launcher is ABI-incompatible with the
    # bundled Triton CPU backend ("function takes exactly 18 arguments (21
    # given)"); both TP workers die on the first request and the engine hangs.
    # The V1 CPU model runner is the native CPU path and has no such kernel.
    CONTAINER_ENV="${CONTAINER_ENV},APPTAINERENV_VLLM_USE_V2_MODEL_RUNNER=0"
else
    CONTAINER_ENV="${CONTAINER_ENV},OMP_PROC_BIND=close,OMP_PLACES=cores"
fi

# Diagnostic verbosity for this job. Overrides whatever LLM_BENCH_LOG_LEVEL is
# set to in CONFIG for a single submission; leave --log-level off to use that
# default (INFO unless CONFIG says otherwise).
[[ -n "${LOG_LEVEL}" ]] && CONTAINER_ENV="${CONTAINER_ENV},LLM_BENCH_LOG_LEVEL=${LOG_LEVEL}"

# Collect a campaign's runs under one tree instead of the shared RESULTS_ROOT.
# Read after the cluster config is sourced (scripts/_common.sh), so it survives
# the re-source that every entry point performs.
JOB_ENV=""
[[ -n "${RESULTS_ROOT_OVERRIDE}" ]] && JOB_ENV=",BENCH_RESULTS_ROOT=${RESULTS_ROOT_OVERRIDE}"

sbatch \
  --job-name="llmbench-${WORKLOAD_NAME}-${PROFILE_NAME}" \
  --partition="${PARTITION}" \
  --account=enea \
  --nodelist="${NODELIST}" \
  --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" --exclusive \
  ${DEPENDENCY:+--dependency="${DEPENDENCY}"} \
  --time="${TIME_LIMIT}" \
  --output="${LOGS}/%x-%j.out" \
  --error="${LOGS}/%x-%j.err" \
  --export=ALL,BENCH_CONFIG="${CONFIG}",MODEL_CONFIG="${MODEL}",WORKLOAD_CONFIG="${WORKLOAD}",PROFILE_CONFIG="${PROFILE}",BENCH_PROVIDER=apptainer,MODEL_VARIANT="${VARIANT}","${CONTAINER_ENV}""${JOB_ENV}" \
  "${REPO}/slurm/benchmark.sbatch"
