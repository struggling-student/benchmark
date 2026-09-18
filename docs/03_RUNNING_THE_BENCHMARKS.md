# 03 — Running the benchmark matrix

One experiment is the composition of a model manifest, workload, execution profile, provider, and
artifact variant. No benchmark-specific launch script or Slurm file is required.

## Shipped dimensions

Models:

- `configs/models/llama32_1b.yaml`
- `configs/models/llama31_8b.yaml`

Workloads:

- `fixed_*.yaml`: deterministic offline token shapes reproduced from the inference literature.
  No generic or synthetic-only workload is shipped; see
  [Literature-matched fixed-length workloads](#literature-matched-fixed-length-workloads).

Profiles:

- `vllm_gpu.yaml`
- `vllm_cpu_amx_hbm_cache.yaml` (Xeon Max cache-mode vLLM CPU, BF16/AMX first treatment)
- `vllm_cpu_amx_hbm_flat.yaml` (Xeon Max flat-mode HBM placement, BF16/AMX vLLM CPU)
- `llamacpp_cpu.yaml` (portable CPU baseline with automatic ISA dispatch)
- `llamacpp_cpu_avx2.yaml` (explicit AVX2 build)
- `llamacpp_cpu_avx512.yaml` (explicit AVX-512 build)
- `llamacpp_cpu_avx512_hbm_flat.yaml` (Xeon Max flat-mode HBM placement)
- `llamacpp_cpu_amx_hbm_flat.yaml` (Xeon Max flat-mode HBM placement)
- `llamacpp_cpu_avx512_hbm_cache.yaml` (CRESCO8 Xeon Max cache-mode AVX-512 treatment)
- `llamacpp_cpu_amx_hbm_cache.yaml` (CRESCO8 Xeon Max cache-mode AMX treatment)
- `llamacpp_cuda.yaml` (change `gpu_layers` from `all` to a number for partial offload).

The three flat profiles use `memory_binding: hbm`. At preflight and launch time this is resolved to
the memory-only NUMA nodes discovered in that allocation; numeric node IDs are never assumed by the
profile. To measure DDR on the same flat-mode node, copy the relevant profile and change
`memory_type` to `ddr5` and `memory_binding` to `ddr`.

Flat-mode binding is backend-independent: `memory_binding` is an OS-level NUMA policy that the
providers apply around whichever runtime is launched (`numactl --membind` for the native and
Apptainer providers, `docker run --cpuset-mems` for Docker), so it is available to both the vLLM
and llama.cpp CPU profiles. GPU profiles reject it.

On CRESCO8, `cresco8-hbm14` is configured in flat mode. Allocation identity is not trusted as the
only evidence: the benchmark still verifies the topology inside every job before inference starts.
That node has been unavailable since 2026-09-10, so `vllm_cpu_amx_hbm_flat.yaml` is validated by
configuration and preflight logic only; it has not yet produced a measured run.

All three providers can execute a compatible profile. Use the `bf16` model variant for the vLLM
CPU AMX profile. Only llama.cpp supports `q8_0` and `q4_k_m`; quantized-to-F16 results are
descriptive.

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

vLLM CPU profiles additionally set the CPU KV-cache allocation, OpenMP core binding, and reserved
front-end cores. The cache-mode profile uses two tensor-parallel ranks to match the two NUMA nodes
observed on `cresco8-hbm15`; preflight must confirm that topology again in the actual allocation.

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
  --workload configs/workloads/fixed_32_32.yaml
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

## Logging

Every `llm-bench` subcommand logs through the standard library `logging` module under the
`llm_bench` namespace, to stderr, so stdout stays reserved for the JSON/paths the CLI already
prints. INFO covers lifecycle milestones (resolved experiment, run directory, warm-up/repetition
progress, server health, completion); DEBUG adds the full resolved configuration, backend and
provider-wrapped argv, NUMA/ISA detail, and per-attempt health-check noise; WARNING/ERROR surface
the same conditions that already produce warnings or a non-zero exit.

The level is decided once, globally, for a whole job:

1. `llm-bench --log-level DEBUG run ...` — the flag must come **before** the subcommand.
2. the `LLM_BENCH_LOG_LEVEL` environment variable (`DEBUG`, `INFO`, `WARNING`, `ERROR`, or
   `CRITICAL`) — set it in a cluster config file (see
   `configs/cluster/sapienza.example.env`) to fix the level for an entire allocation without
   touching any script, or pass `--log-level` to `scripts/run_benchmark.sh` or
   `scripts/submit_cresco8.sh` to override it for one job.
3. `INFO` otherwise.

## Run locally or inside an allocation

```bash
llm-bench run "${COMMON[@]}" --results-root "$RESULTS_ROOT"
```

The Python lifecycle is:

```text
resolve -> preflight -> initialize result -> start backend -> readiness/model check
        -> warm-ups -> measured repetitions -> normalize -> signal-safe cleanup
```

All shipped literature workloads invoke `vllm bench throughput` or `llama-bench`. The requested prompt/generation lengths
and repetition count are mapped to their native flags, original JSON is retained, and llama-bench's
discarded internal warm-up is documented separately from configured outer warm-ups.

The codebase retains the OpenAI-compatible API runner for a future literature-backed trace such as
ShareGPT, but no generic serving or smoke configuration is part of the benchmark catalog.

## Literature-matched fixed-length workloads

The fixed-length presets reproduce the reported prompt/generation pairs from the directly relevant
systems papers. The shipped model manifests use the standard
[Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
[Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) native context limit of
131,072 tokens rather than the legacy 8,192-token deployment cap. Paper-reported token lengths are
exact unless marked approximate; harness controls remain standardized where a paper does not report
an equivalent backend-neutral value. This limit also applies to this benchmark's GGUF conversions
of those standard checkpoints; they are not Meta's separately released 8K-context quantized models.

| Workload | Input | Output | Source |
| --- | ---: | ---: | --- |
| `fixed_32_32.yaml` | 32 | 32 | [Shen et al. (2023), Table 3](https://arxiv.org/html/2311.00502) |
| `fixed_128_32.yaml` | 128 | 32 | [Na et al. (2024), default evaluation](https://seonjinna.github.io/assets/pdf/iiswc24_CPULLM.pdf) |
| `fixed_256_32.yaml` | 256 | 32 | Na et al. sequence-length sweep; [FlexGen Table 14](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf) |
| `fixed_512_32.yaml` | 512 | 32 | Na et al. sequence-length sweep; FlexGen main setup/Table 15 |
| `fixed_1024_32.yaml` | 1,024 | 32 | Na et al. sequence-length sweep; FlexGen Table 16 |
| `fixed_128_128.yaml` | 128 | 128 | FlexGen Table 17 |
| `fixed_512_8.yaml` | 512 | 8 | FlexGen Table 18 |
| `fixed_512_128.yaml` | 512 | 128 | [THInfer ablation setup](https://arxiv.org/html/2605.25655); [Kurt et al. llama.cpp `pp512`/`tg128`](https://arxiv.org/html/2601.14277) |
| `fixed_1024_128.yaml` | 1,024 | 128 | THInfer throughput comparison |
| `fixed_30000_10000.yaml` | ≈30,000 | 10,000 | [Fang et al. LongBench/NarrativeQA long-context workload](https://arxiv.org/html/2508.13231) |

These presets align token shape, not every experimental condition. In particular, Na et al.'s
static batch-size sweep (1--32), FlexGen's engine-specific effective batches, and THInfer's device
counts do not map to `number_of_prompts`. For vLLM offline runs that field becomes `--num-prompts`;
for llama.cpp it becomes the native `llama-bench` repetition count. Those native offline tools also
time different scopes, so their results remain descriptive across backends.

Papers that report only a context length or next-token latency without a generation length are not
encoded as fixed prompt/generation pairs. NoMAD's variable-length prompts are also not represented
as one fixed input length; doing so would falsely turn its “up to 16K” distribution into an exact
shape.

The published measurements, hardware and software conditions, and valid comparison method for
every preset are recorded in [06 - Literature result baselines](06_LITERATURE_RESULTS.md).

The shell wrapper is useful when the environment is stored in the private cluster config:

```bash
bash scripts/run_benchmark.sh \
  --config configs/cluster/sapienza.env \
  --model configs/models/llama32_1b.yaml \
  --workload configs/workloads/fixed_128_32.yaml \
  --profile configs/profiles/vllm_gpu.yaml \
  --provider native --variant f16
```

## Submit through Slurm

The generic entry point contains no site-specific partition, account, QoS, CPU, memory, GPU, or time
request. Supply verified values to `sbatch`:

```bash
sbatch <SITE_RESOURCE_OPTIONS> \
  --export=ALL,BENCH_CONFIG="$PWD/configs/cluster/sapienza.env",MODEL_CONFIG="$PWD/configs/models/llama32_1b.yaml",WORKLOAD_CONFIG="$PWD/configs/workloads/fixed_128_32.yaml",PROFILE_CONFIG="$PWD/configs/profiles/llamacpp_cuda.yaml",BENCH_PROVIDER=apptainer,MODEL_VARIANT=f16 \
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
named `raw_backend_output*.json`.

The initialized result is written before inference starts. Interruptions and benchmark failures
request process-group/container cleanup and retain a failed partial summary instead of deleting
evidence.

## Comparison rules

Use `llm-bench compare LEFT RIGHT --output-dir OUTPUT` for ordinary like-for-like comparisons.
The comparison checks source/tokenizer revisions, precision, provider, hardware, measurement method,
workload hash, generation controls, warm-ups, and repetitions before emitting ratios.

Native offline comparisons across backends, CPU-versus-GPU, provider-mismatched, and
quantized-versus-F16 pairs are descriptive or incompatible. A shared schema never implies a shared
measurement boundary.

Continue with [04 — Metrics and results](04_METRICS_AND_RESULTS.md).
