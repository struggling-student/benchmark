# 03 — Running the benchmark matrix

One experiment is the composition of a model manifest, workload, execution profile, provider, and
artifact variant. No benchmark-specific launch script or Slurm file is required.

## Shipped dimensions

Models:

- `configs/models/llama32_1b.yaml`
- `configs/models/llama31_8b.yaml`

Workloads:

- `smoke.yaml`: two short API requests and no measured-performance claim.
- `offline.yaml`: native throughput tools with outer warm-up and three measured repetitions.
- `serving.yaml`: deterministic streamed prompts, fixed generation controls, request rate, and
  concurrency.

Profiles:

- `vllm_gpu.yaml`
- `llamacpp_cpu.yaml` (portable CPU baseline with automatic ISA dispatch)
- `llamacpp_cpu_avx2.yaml` (explicit AVX2 build)
- `llamacpp_cpu_avx512.yaml` (explicit AVX-512 build)
- `llamacpp_cpu_avx512_hbm_flat.yaml` (Xeon Max flat-mode HBM placement)
- `llamacpp_cpu_amx_hbm_flat.yaml` (Xeon Max flat-mode HBM placement)
- `llamacpp_cpu_avx512_hbm_cache.yaml` (CRESCO8 Xeon Max cache-mode AVX-512 treatment)
- `llamacpp_cpu_amx_hbm_cache.yaml` (CRESCO8 Xeon Max cache-mode AMX treatment)
- `llamacpp_cuda.yaml` (change `gpu_layers` from `all` to a number for partial offload).

The two flat profiles use `memory_binding: hbm`. At preflight and launch time this is resolved to
the memory-only NUMA nodes discovered in that allocation; numeric node IDs are never assumed by the
profile. To measure DDR on the same flat-mode node, copy the relevant profile and change
`memory_type` to `ddr5` and `memory_binding` to `ddr`.

On CRESCO8, `cresco8-hbm14` is configured in flat mode. Allocation identity is not trusted as the
only evidence: the benchmark still verifies the topology inside every job before inference starts.

All three providers can execute a compatible profile. Only llama.cpp supports `q8_0` and
`q4_k_m`; quantized-to-F16 results are descriptive.

The generic CPU profiles pin 16 threads to mask `0xffff`. Replace the thread count and mask together
when the allocated CPU topology requires a different placement. The CRESCO8 HBM profiles use all
112 physical cores and leave affinity/NUMA policy unset until the campaign's socket-placement design
is chosen explicitly.

## Hardware profile contract

`hardware_type` selects `cpu`, `gpu`, or `hybrid`. CPU profiles also configure:

- `cpu_isa`: `auto`, `avx2`, `avx512`, or `amx`;
- `cpu_features_required`: extra Linux CPU flags required by that exact build/treatment;
- `memory_type`: a descriptive tier such as `system_memory`, `hbm2e`, or `hbm2e+ddr5`;
- `memory_mode`: `none`, `flat`, or `cache`;
- `memory_binding`: `hbm`, `ddr`, or an explicit numeric NUMA node list. Symbolic tiers are resolved
  from the allocated node's topology and the numeric result is passed to the runtime.

Explicit ISA profiles require a matching ISA-specific runtime variable. Preflight reads the node's
CPU flags and Linux NUMA sysfs before the backend starts. On Xeon Max it detects flat mode from
separate CPU-less HBM NUMA nodes and cache mode from HBM being hidden behind DDR. A mismatch aborts
the run before inference; requested/detected modes, detection method, CPU flags, and the full NUMA
inventory are retained in metadata and the summary.

## Resolve and inspect

Use `validate` for schema/capability composition, `preflight` for installed runtime and artifact
checks, or `run --dry-run` for the complete resolved configuration and provider argv:

```bash
COMMON=(
  --model configs/models/llama32_1b.yaml
  --workload configs/workloads/smoke.yaml
  --profile configs/profiles/llamacpp_cuda.yaml
  --provider apptainer
  --variant f16
  --artifact-root "$MODEL_ARTIFACT_ROOT"
)
llm-bench validate "${COMMON[@]}"
llm-bench preflight "${COMMON[@]}"
llm-bench run --dry-run "${COMMON[@]}"
```

Dry-run performs no inference. It resolves the artifact, checks the provider, backend, CPU ISA and
HBM topology, exposes telemetry capabilities, determines mounts/ports/GPU flags, and prints the exact
argv and hardware evidence as JSON. Run it inside the same Slurm allocation intended for inference.

## Run locally or inside an allocation

```bash
llm-bench run "${COMMON[@]}" --results-root "$RESULTS_ROOT"
```

The Python lifecycle is:

```text
resolve -> preflight -> initialize result -> start backend -> readiness/model check
        -> warm-ups -> measured repetitions -> normalize -> signal-safe cleanup
```

Smoke and serving start an OpenAI-compatible server, verify `/health` and `/v1/models`, generate and
persist deterministic exact-length prompts with the canonical tokenizer, then use the same streaming
client for vLLM and llama.cpp. The client applies explicit maximum tokens, temperature, top-p, EOS,
seed, request-rate, and concurrency controls.

Offline invokes `vllm bench throughput` or `llama-bench`. The requested prompt/generation lengths
and repetition count are mapped to their native flags, original JSON is retained, and llama-bench's
discarded internal warm-up is documented separately from configured outer warm-ups.

The shell wrapper is useful when the environment is stored in the private cluster config:

```bash
bash scripts/run_benchmark.sh \
  --config configs/cluster/sapienza.env \
  --model configs/models/llama32_1b.yaml \
  --workload configs/workloads/serving.yaml \
  --profile configs/profiles/vllm_gpu.yaml \
  --provider native --variant f16
```

## Submit through Slurm

The generic entry point contains no site-specific partition, account, QoS, CPU, memory, GPU, or time
request. Supply verified values to `sbatch`:

```bash
sbatch <SITE_RESOURCE_OPTIONS> \
  --export=ALL,BENCH_CONFIG="$PWD/configs/cluster/sapienza.env",MODEL_CONFIG="$PWD/configs/models/llama32_1b.yaml",WORKLOAD_CONFIG="$PWD/configs/workloads/serving.yaml",PROFILE_CONFIG="$PWD/configs/profiles/llamacpp_cuda.yaml",BENCH_PROVIDER=apptainer,MODEL_VARIANT=f16 \
  slurm/benchmark.sbatch
```

For an array, derive `MODEL_CONFIG`, `WORKLOAD_CONFIG`, or another dimension in a small
site-controlled submission wrapper; keep `slurm/benchmark.sbatch` unchanged. The job calls the same
generic shell and Python runner used interactively.

## Result directory

A completed API run resembles:

```text
RUN_DIRECTORY/
├── metadata.json
├── measurements.json
├── model_artifact_provenance.json
├── resolved_experiment.yaml
├── summary.json
├── workload_manifest.json
├── raw_harness_output.json
├── resource_telemetry.csv
└── server.log
```

Repeated files have `.repetition-NNN` suffixes; warm-ups live under `warmup/`. Offline raw files are
named `raw_backend_output*.json`. Old `raw_vllm_output*.json` files remain discoverable for legacy
results.

The initialized result is written before inference starts. Interruptions and benchmark failures
request process-group/container cleanup and retain a failed partial summary instead of deleting
evidence.

## Comparison rules

Use `llm-bench compare LEFT RIGHT --output-dir OUTPUT` for ordinary like-for-like comparisons.
Use the dashboard's backend-treatment lens for vLLM versus llama.cpp. It requires distinct backends
but matching source/tokenizer revisions, F16 precision, provider, hardware, shared-API method,
workload hash, actual token counts, generation controls, warm-ups, and repetitions.

Native offline, CPU-versus-GPU, provider-mismatched, and quantized-versus-F16 pairs remain visible
but descriptive or incompatible. A shared schema never implies a shared measurement boundary.

Continue with [04 — Metrics and results](04_METRICS_AND_RESULTS.md).
