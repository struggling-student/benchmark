# Modular LLM inference benchmark

This repository benchmarks the same Llama models and workloads through either
[vLLM](https://docs.vllm.ai/) or [llama.cpp](https://github.com/ggml-org/llama.cpp).
It is designed for Linux/Slurm, but native and Docker execution also work on a local host.

The shipped models are:

- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

The benchmark measures inference-system behavior, not model quality. It is not an official
MLPerf implementation or submission.

## Architecture

Experiments are composed at the CLI instead of being copied into backend-specific files:

- `configs/models/` defines canonical Hugging Face identity, tokenizer, revision policy, and
  backend artifact variants.
- `configs/workloads/` defines smoke, offline, and serving request shapes and controls.
- `configs/profiles/` defines vLLM GPU and llama.cpp CPU/CUDA placement.
- `--provider` selects `native`, `docker`, or `apptainer` without changing the other three inputs.

`f16` is available to both backends. `q8_0` and `q4_k_m` are explicit llama.cpp-only variants.
Changing a model or a workload is therefore config-only; adding a backend or workload method is a
single registered Python implementation rather than a change to the CLI, Slurm entry point, result
writer, and dashboard.

Smoke and serving use the repository's asynchronous OpenAI-compatible streaming client. Offline
runs deliberately retain the native tools: `vllm bench throughput` and `llama-bench`. Their
measurement scopes differ, so cross-backend offline results are descriptive and never presented as
controlled ratios.

## Setup

Create a private cluster configuration and install the runner:

```bash
cp configs/cluster/sapienza.example.env configs/cluster/sapienza.env
# Replace every placeholder in configs/cluster/sapienza.env.
bash scripts/create_venv.sh --config configs/cluster/sapienza.env
source configs/cluster/sapienza.env
source "${VENV_PATH}/bin/activate"
```

Install a vLLM build compatible with the host CUDA stack when using vLLM. For native llama.cpp,
set `LLAMA_CPP_BIN_DIR`, `LLAMA_CPP_CONVERT_SCRIPT`, and optionally
`LLAMA_CPP_QUANTIZE_BIN`. For containers, configure immutable image references ending in
`@sha256:<digest>`.

Prepare GGUF files outside timed runs. This also resolves the Hugging Face snapshot to an immutable
commit, writes `artifact-manifest.json`, and makes subsequent vLLM and llama.cpp compositions use
that same source revision:

```bash
llm-bench prepare-model \
  --model configs/models/llama32_1b.yaml \
  --provider native \
  --variants f16,q8_0,q4_k_m \
  --artifact-root "$MODEL_ARTIFACT_ROOT" \
  --cache-root "$MODEL_CACHE_DIR"
```

Repeat for `configs/models/llama31_8b.yaml`. Model snapshots, converted files, container caches,
and results remain outside Git.

## Validate and run

Every execution uses the same stable interface:

```text
llm-bench validate --model MODEL --workload WORKLOAD --profile PROFILE --provider PROVIDER --variant VARIANT
llm-bench prepare-model --model MODEL --provider PROVIDER --variants f16,q8_0,q4_k_m
llm-bench preflight --model MODEL --workload WORKLOAD --profile PROFILE --provider PROVIDER --variant VARIANT
llm-bench run --model MODEL --workload WORKLOAD --profile PROFILE --provider PROVIDER --variant VARIANT
```

For example, validate a vLLM GPU smoke test and inspect the fully wrapped command without starting
inference:

```bash
llm-bench validate \
  --model configs/models/llama32_1b.yaml \
  --workload configs/workloads/smoke.yaml \
  --profile configs/profiles/vllm_gpu.yaml \
  --provider native --variant f16 \
  --artifact-root "$MODEL_ARTIFACT_ROOT"

llm-bench run --dry-run \
  --model configs/models/llama32_1b.yaml \
  --workload configs/workloads/smoke.yaml \
  --profile configs/profiles/llamacpp_cpu.yaml \
  --provider native --variant f16 \
  --artifact-root "$MODEL_ARTIFACT_ROOT"
```

Then remove `--dry-run` to execute. The generic shell wrapper loads the cluster environment first:

```bash
bash scripts/run_benchmark.sh \
  --config configs/cluster/sapienza.env \
  --model configs/models/llama32_1b.yaml \
  --workload configs/workloads/smoke.yaml \
  --profile configs/profiles/llamacpp_cpu.yaml \
  --provider native --variant f16
```

The same runner accepts `offline.yaml` or `serving.yaml`, either model, either compatible profile,
and any supported provider/variant. Invalid combinations fail before a result is timed.

For Slurm, pass site resources to `sbatch`; the repository does not guess them:

```bash
sbatch <SITE_RESOURCE_OPTIONS> \
  --export=ALL,BENCH_CONFIG="$PWD/configs/cluster/sapienza.env",MODEL_CONFIG="$PWD/configs/models/llama32_1b.yaml",WORKLOAD_CONFIG="$PWD/configs/workloads/smoke.yaml",PROFILE_CONFIG="$PWD/configs/profiles/llamacpp_cuda.yaml",BENCH_PROVIDER=apptainer,MODEL_VARIANT=f16 \
  slurm/benchmark.sbatch
```

## Results and comparison

Each run is self-contained and includes `resolved_experiment.yaml`, source-config hashes,
`model_artifact_provenance.json`, `metadata.json`, `summary.json` schema 2.0,
`measurements.json`, backend or shared-harness raw JSON, telemetry CSV, logs, and a deterministic
workload manifest/hash. API workloads include exact prompt text. Existing schema 1.0 and 1.1 result
directories remain readable.

The ordinary comparison command is strict and intended for like-for-like model comparisons:

```bash
llm-bench compare RUN_A RUN_B --output-dir COMPARISON_DIRECTORY
```

The Streamlit dashboard adds filters for backend, profile, provider, artifact variant, hardware,
and measurement method. Its backend-treatment lens only produces ratios for controlled F16 shared-
API runs whose source revision, tokenizer, provider, hardware, workload hash, actual token counts,
and generation controls align.

```bash
pip install -e ".[dashboard]"
llm-bench dashboard --results-root "$RESULTS_ROOT"
```

The dashboard is read-only. Running it without `--results-root` opens representative simulated
vLLM and llama.cpp CPU/GPU data.

## Guides

1. [Inspect the cluster](docs/01_CLUSTER_PREREQUISITES.md)
2. [Set up providers and prepare models](docs/02_ENVIRONMENT_AND_MODEL_SETUP.md)
3. [Run the benchmark matrix](docs/03_RUNNING_THE_BENCHMARKS.md)
4. [Interpret metrics, results, and compatibility](docs/04_METRICS_AND_RESULTS.md)
5. [Extend the CPU path toward HBM](docs/05_CPU_HBM_ROADMAP.md)
