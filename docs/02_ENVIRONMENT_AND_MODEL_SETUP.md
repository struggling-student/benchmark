# 02 — Environment and model setup

This guide creates a Python virtual environment, installs this small local package, and prepares both model targets. It deliberately does not prescribe a fixed PyTorch, CUDA, Python, or vLLM combination: compatibility changes, and the correct choice must follow the current official vLLM documentation and the environment actually available on the cluster.

## Step 0: load the discovered cluster settings

Complete [01 — Cluster prerequisites](01_CLUSTER_PREREQUISITES.md), then load the private configuration:

```bash
source configs/cluster/sapienza.env
```

Review the values before a setup run:

```bash
printf 'Repository: %s\nPython: %s\nVenv: %s\nHF cache: %s\nModel cache: %s\nResults: %s\n' \
  "$BENCH_REPO_ROOT" "$PYTHON_EXECUTABLE" "$VENV_PATH" "$HF_HOME" \
  "$MODEL_CACHE_DIR" "$RESULTS_ROOT"
```

Do not print `HF_TOKEN`. Run the configured module commands exactly as documented by the site. If `MODULE_COMMANDS` is used by the repository scripts, inspect its contents before execution because it is shell configuration supplied by you.

## Step 1: inspect compatibility inputs before installing

On a GPU node, after loading the selected modules, record:

```bash
"$PYTHON_EXECUTABLE" --version
"$PYTHON_EXECUTABLE" -c 'import platform, sys; print(sys.executable); print(platform.platform())'
command -v nvcc && nvcc --version
command -v nvidia-smi && nvidia-smi
```

Consult the current [official vLLM installation documentation](https://docs.vllm.ai/en/latest/getting_started/installation/) and release notes. Verify that the selected Python version, platform, NVIDIA driver, CUDA expectations, and intended PyTorch/vLLM distribution are mutually supported. Do not copy an old command merely because it worked for another cluster.

The setup scripts print detected versions, but they cannot make an unsupported driver/runtime combination compatible. A missing GPU on a login node is not conclusive; validate CUDA availability in a GPU allocation.

## Step 2: create and activate the virtual environment

The primary environment-management mechanism is the standard library's `venv`:

```bash
bash scripts/create_venv.sh --config configs/cluster/sapienza.env
source "${VENV_PATH}/bin/activate"
python --version
python -m pip --version
```

The script must use the configured `PYTHON_EXECUTABLE`; it must not silently choose a different Python or an arbitrary CUDA/PyTorch stack. If creation fails, retain the diagnostic output and revisit the module and Python selection.

## Step 3: install the repository and vLLM

Install the local package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install vLLM using the method recommended by its current official documentation for the environment discovered in Step 1. Some releases package the dependencies needed by `vllm bench` as a documented benchmark extra; install that extra when the selected release requires it, then verify the benchmark help in Step 7. No fixed vLLM/PyTorch/CUDA or benchmark-extra install command is embedded here because doing so would age quickly and could select an incompatible build.

Verify the resolved environment and record it:

```bash
python -c 'import torch; print("torch", torch.__version__); print("torch CUDA", torch.version.cuda); print("CUDA available", torch.cuda.is_available())'
python -c 'import vllm; print("vLLM", vllm.__version__)'
python -m pip check
python -m pip freeze > "${VENV_PATH}/installed-versions.txt"
```

`installed-versions.txt` is an environment record, not a cross-platform lock file. Every benchmark run also captures `pip freeze` in its `system/` metadata. If imports fail, `pip check` reports conflicts, or CUDA is unavailable on an allocated GPU node, stop and resolve compatibility before downloading or benchmarking models.

Development work can additionally install the project's test and lint dependencies as defined in `pyproject.toml`; they are not runtime requirements for vLLM inference.

## Step 4: obtain Hugging Face access safely

Both benchmark targets are hosted on Hugging Face:

- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

Model pages may require authentication and acceptance of the publisher's gated-model terms. In a browser, sign in to Hugging Face, request/accept access where required, and wait until the account is authorized. Repository code cannot accept license terms for you.

Create a Hugging Face access token with only the permissions needed to read the models. Export it into the current shell or provide it through the cluster's approved secret mechanism:

```bash
read -r -s -p 'HF token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

Never echo the token, pass it as a command-line argument, save it in `sapienza.env`, include it in Slurm logs, or commit it. Clear it when no longer needed:

```bash
unset HF_TOKEN
```

Authentication proves identity; it does not replace acceptance of gated-model terms. The Slurm examples later use `--export=ALL`, which would inherit every exported submission-shell variable, including `HF_TOKEN`. After pre-downloading, run `unset HF_TOKEN` before submitting those examples and use the shared cache offline. If a compute node genuinely needs authentication, do not use the examples verbatim: follow the site's approved secret-injection and environment-export procedure, and verify that it does not write the token to job arguments or logs.

## Step 5: place caches outside Git

Set `HF_HOME` and `MODEL_CACHE_DIR` in `configs/cluster/sapienza.env` to verified shared, project, or scratch paths visible on compute nodes. Follow site policy: scratch may be fast but periodically purged, while a shared project area may have stricter quotas. The benchmark shell helpers map `MODEL_CACHE_DIR` to Hugging Face Hub's `HF_HUB_CACHE`, keeping vLLM and the downloader on the same snapshot cache; `HF_HOME` remains the broader Hugging Face state/cache root.

Confirm access without revealing secrets:

```bash
mkdir -p "$HF_HOME" "$MODEL_CACHE_DIR" "$RESULTS_ROOT"
test -w "$HF_HOME" && test -w "$MODEL_CACHE_DIR" && test -w "$RESULTS_ROOT"
df -h "$HF_HOME" "$MODEL_CACHE_DIR" "$RESULTS_ROOT"
```

Do not place `HF_HOME`, `MODEL_CACHE_DIR`, or model weights beneath the repository. `RESULTS_ROOT` is different: it may be an external location or the checkout's ignored `results/` directory. The 8B snapshot and its runtime memory footprint are larger than those of the 1B model, but exact disk and GPU-memory needs vary by dtype, quantization, revision, format, context length, KV cache, and vLLM overhead. Check snapshot sizes and free space after download; treat an out-of-memory failure as measured resource information, not something to hide.

## Step 6: pre-download both model snapshots

With `HF_TOKEN` set and the cache paths loaded, download each configurable target using its experiment file:

```bash
python scripts/download_model.py --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_smoke.yaml
python scripts/download_model.py --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_smoke.yaml
```

These are the downloader's required `--config` and `--experiment` options. It reads only `HF_HOME`, `HF_HUB_CACHE`, and `MODEL_CACHE_DIR` assignments from the cluster file rather than executing that file; `--cache-dir <ABSOLUTE_PATH>` can override the selected snapshot cache. The model and any distinct tokenizer come from the experiment YAML, not from hard-coded IDs or a model-ID command-line option.

The example experiments deliberately leave `model_revision: null`, which remains `null` as the configured value. At run initialization, the code also attempts a network-free lookup of the selected local Hugging Face cache ref and records `resolved_model_revision` and `resolved_tokenizer_revision` when a valid commit hash is available. If a local ref cannot be resolved, those fields remain `null` with a warning; no identity is fabricated. This cache observation improves the audit trail but is not a replacement for intent. For a formal comparison, choose and put an explicit immutable revision in each paired experiment before downloading and running.

Verify that both snapshots are visible from a compute-node allocation:

```bash
test -d "$MODEL_CACHE_DIR"
du -sh "$MODEL_CACHE_DIR"
```

Do not list environment variables or use shell tracing while a token is present.

### Compute nodes without internet access

Download both models once from a network-enabled host or login node only if site policy allows it, targeting a cache visible to compute nodes. Then submit jobs with the same `HF_HOME`/`MODEL_CACHE_DIR` and enable the Hugging Face libraries' documented offline mode if needed:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Test offline loading before a long benchmark. Copying caches between unrelated machines can fail because of incomplete snapshots, permissions, symlinks, or different filesystem layouts; use the downloader's completed snapshot and preserve its metadata.

The same downloader can verify that each configured snapshot is already present without attempting network access:

```bash
python scripts/download_model.py --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_smoke.yaml --local-files-only
python scripts/download_model.py --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_smoke.yaml --local-files-only
```

## Step 7: inspect the installed vLLM CLI

Exact benchmark commands and flags differ across vLLM releases. Before choosing or changing an experiment, inspect what the installed version exposes:

```bash
vllm --help
vllm bench --help
vllm bench throughput --help
vllm bench serve --help
vllm serve --help
```

Some releases expose benchmark entry points under different command names. The repository launchers check the installed CLI and must fail with a useful message if the required offline throughput or serving benchmark is unavailable; do not blindly substitute flags from another release.

The serving launcher also checks both recorded help texts for every control needed by the configured workload. This includes disabling model-repository generation defaults on the server and applying temperature, top-p, and EOS behavior in the benchmark client. If the selected release does not advertise the required flags, the run is marked failed instead of silently benchmarking a different generation policy. Choose a compatible documented vLLM release; do not remove controls from only one model's experiment to work around the check.

## Step 8: run the preflight check

```bash
bash scripts/preflight_check.sh --config configs/cluster/sapienza.env
```

The preflight should report Python, package, CUDA/driver, GPU visibility, cache, and writable-result information without printing secrets. Run it inside the same kind of Slurm allocation used for benchmarks. Treat incompatible versions, an unreadable model cache, or no visible GPU as setup failures rather than benchmark results.

## Step 9: verify the models in increasing resource order

First validate the complete path with the lower-resource model:

```bash
bash scripts/run_smoke_test.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_smoke.yaml
```

Then verify the primary 8B target explicitly:

```bash
bash scripts/run_smoke_test.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_smoke.yaml
```

The 1B result does not substitute for the 8B result. Both are first-class targets and later use the same measured workloads and metrics. A smoke test establishes that loading and generation work; it is not a performance comparison.

Continue with [03 — Running the benchmarks](03_RUNNING_THE_BENCHMARKS.md).
