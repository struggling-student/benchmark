# 02 — Environment, providers, and model preparation

Keep the repository, Python environment, Hugging Face cache, prepared GGUF artifacts, and results
as separate locations. In particular, weights and generated results should not be committed.

## Create the Python environment

Copy and edit the cluster example after completing the discovery steps in the previous guide:

```bash
cp configs/cluster/sapienza.example.env configs/cluster/sapienza.env
bash scripts/create_venv.sh --config configs/cluster/sapienza.env
source configs/cluster/sapienza.env
source "${VENV_PATH}/bin/activate"
```

The base install includes the CLI, Hugging Face snapshot support, tokenizer support, the shared
HTTP client, and portable process telemetry. Install `.[dashboard]` only where Streamlit is needed.
Install vLLM separately using the version compatible with the selected Python, CUDA runtime, and
driver.

## Configure an execution provider

Provider selection changes command execution, not the model or workload definition.

### Native

For vLLM, put `vllm` on `PATH` or set `VLLM_BIN` to its executable. For llama.cpp, provide a build
whose tools match each other:

```bash
export LLAMA_CPP_BIN_DIR=/absolute/path/to/llama.cpp/build/bin
export LLAMA_CPP_CONVERT_SCRIPT=/absolute/path/to/llama.cpp/convert_hf_to_gguf.py
export LLAMA_CPP_QUANTIZE_BIN=/absolute/path/to/llama.cpp/build/bin/llama-quantize
```

An explicit CPU ISA profile selects a separate build directory:

```bash
export LLAMA_CPP_AVX2_BIN_DIR=/absolute/path/to/llama.cpp/build-avx2/bin
export LLAMA_CPP_AVX512_BIN_DIR=/absolute/path/to/llama.cpp/build-avx512/bin
export LLAMA_CPP_AMX_BIN_DIR=/absolute/path/to/llama.cpp/build-amx/bin
```

Build these from the same pinned llama.cpp commit. The AVX-512 build must have AMX disabled; the
AMX build must enable the required AMX and AVX-512 features. llama.cpp controls these at build
time through `GGML_AVX2`, `GGML_AVX512*`, and `GGML_AMX_*`, so changing only a YAML label is not a
controlled ISA treatment. Its optional `GGML_CPU_HBM`/memkind build feature is relevant only when
using addressable HBM in flat mode; cache mode is transparent to the application. Preserve each
build's CMake cache or build log with the campaign records.

The runner invokes argv directly. YAML files cannot inject arbitrary shell commands.

### Docker

Set the image for each backend used. OCI references must be immutable digest references:

```bash
export LLAMA_CPP_DOCKER_IMAGE='ghcr.io/ggml-org/llama.cpp:full@sha256:<DIGEST>'
export VLLM_DOCKER_IMAGE='<REGISTRY>/<VLLM_IMAGE>@sha256:<DIGEST>'
```

For explicit CPU ISA profiles, use the corresponding immutable image variables:
`LLAMA_CPP_AVX2_DOCKER_IMAGE`, `LLAMA_CPP_AVX512_DOCKER_IMAGE`, and
`LLAMA_CPP_AMX_DOCKER_IMAGE`. Apptainer uses the same naming pattern with
`_APPTAINER_IMAGE`.

The provider adds explicit read-only cache/artifact mounts, a writable result mount, port mapping,
GPU selection for GPU/hybrid profiles, Docker-native `--cpuset-mems` placement when configured, a
small environment allowlist, a stable container name, ID capture, and forced cleanup.

### Apptainer or Singularity

Either a pinned `docker://...@sha256:<digest>` URI or an existing local SIF path is accepted:

```bash
export LLAMA_CPP_APPTAINER_IMAGE='docker://ghcr.io/ggml-org/llama.cpp:full@sha256:<DIGEST>'
export VLLM_APPTAINER_IMAGE='/absolute/path/to/vllm.sif'
```

Runs use a clean environment, explicit binds, and `--nv` only for GPU/hybrid profiles. Local image
SHA-256 is recorded. The `source-build` provider contract is reserved, but clone/CMake automation is
intentionally not implemented.

## Obtain access to the models

Both shipped Meta Llama repositories can require accepting their terms. Authenticate only in an
interactive shell on a network-enabled host allowed by site policy:

```bash
read -r -s -p 'Hugging Face token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

Do not write the token into the repository or cluster environment file. Preparation resolves the
configured revision, downloads the snapshot into `MODEL_CACHE_DIR`, and records the immutable commit.

## Prepare GGUF variants

Preparation is outside all timed runs:

```bash
llm-bench prepare-model \
  --model configs/models/llama32_1b.yaml \
  --provider native \
  --variants f16,q8_0,q4_k_m \
  --artifact-root "$MODEL_ARTIFACT_ROOT" \
  --cache-root "$MODEL_CACHE_DIR"

llm-bench prepare-model \
  --model configs/models/llama31_8b.yaml \
  --provider native \
  --variants f16,q8_0,q4_k_m \
  --artifact-root "$MODEL_ARTIFACT_ROOT" \
  --cache-root "$MODEL_CACHE_DIR"

unset HF_TOKEN
```

Use `--provider docker` or `--provider apptainer` to run conversion and quantization with the
configured pinned llama.cpp image. `--local-files-only` proves that preparation can be repeated
without network access.

The command writes one `artifact-manifest.json` per model. It records source commit, snapshot,
converter/quantizer identity, provider/image digest, arguments, artifact sizes, and SHA-256 values.
Temporary outputs are atomically moved into place. Existing files are reused only when their model,
revision, and checksum match the prior manifest; an unproven or modified artifact causes a failure.

Even though vLLM consumes the Hugging Face format rather than GGUF, composition reads the same
artifact manifest and pins vLLM to its resolved source commit. That is how F16 shared-API runs can
establish that backend/version is the intended treatment.

## Verify before entering a long allocation

```bash
llm-bench validate \
  --model configs/models/llama32_1b.yaml \
  --workload configs/workloads/smoke.yaml \
  --profile configs/profiles/llamacpp_cpu.yaml \
  --provider native --variant f16 \
  --artifact-root "$MODEL_ARTIFACT_ROOT"

llm-bench preflight \
  --model configs/models/llama32_1b.yaml \
  --workload configs/workloads/smoke.yaml \
  --profile configs/profiles/llamacpp_cpu.yaml \
  --provider native --variant f16 \
  --artifact-root "$MODEL_ARTIFACT_ROOT"
```

`validate` checks the composed schema. `preflight` additionally checks backend/provider capability,
the selected ISA-specific executable or pinned image, execution-node CPU flags, requested versus
detected Xeon Max HBM mode, flat-mode NUMA tier placement, GGUF provenance, optional telemetry
capabilities, port/host resolution, and the final provider-wrapped argv.

Continue with [03 — Running the benchmarks](03_RUNNING_THE_BENCHMARKS.md).
