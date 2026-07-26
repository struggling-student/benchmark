# 03 — Running the benchmarks

The repository provides three workflows for each model: smoke validation, offline throughput, and online serving. Each launcher reads an experiment YAML, creates a unique `$RESULTS_ROOT/<date>/<run-id>/` directory, preserves raw vLLM output, writes normalized metadata, and keeps failed runs rather than silently discarding them.

Before every campaign, activate the same environment and run preflight in the same hardware context:

```bash
source configs/cluster/sapienza.env
source "${VENV_PATH}/bin/activate"
bash scripts/preflight_check.sh --config configs/cluster/sapienza.env
```

`BENCH_REPO_ROOT` in that cluster file must be the checkout's absolute path as visible from compute nodes. The Slurm entry points use it to locate repository scripts after Slurm has copied the submitted job file to a spool directory.

Do not describe any command in this guide as completed until its Slurm job finishes successfully and its `summary.json` says `"status": "completed"`.

## Step 0: understand the paired configurations

The six experiment files are intentionally paired:

| Workflow | Llama 3.2 1B | Llama 3.1 8B |
|---|---|---|
| Smoke | `llama32_1b_smoke.yaml` | `llama31_8b_smoke.yaml` |
| Offline | `llama32_1b_offline.yaml` | `llama31_8b_offline.yaml` |
| Serving | `llama32_1b_serving.yaml` | `llama31_8b_serving.yaml` |

Select a model by selecting its configuration file; source changes are unnecessary. For a fair pair, keep identical input length, requested output length, prompt/request count, request rate, maximum concurrency, seed, generation configuration, temperature, top-p, EOS policy, warm-up count, measured repetitions, dtype, quantization, tensor parallel size, `gpu_memory_utilization`, and telemetry interval wherever both models support them. Document any unavoidable model-specific difference.

The smoke workflow defaults to `llama32_1b_smoke.yaml` when no experiment is selected, because it is the lower-resource validation target. The commands below pass `--experiment` explicitly so the saved intent is unambiguous; selecting `llama31_8b_smoke.yaml` makes the 8B validation equally straightforward.

To change a workload, copy both paired files, give them unique `experiment_name` values, make the same workload edit in each copy, and keep the originals as a reference. Run configuration validation before submitting scarce GPU time:

```bash
python -m llm_bench.cli validate-config configs/experiments/<1B_EXPERIMENT>.yaml
python -m llm_bench.cli validate-config configs/experiments/<8B_EXPERIMENT>.yaml
```

Use `python -m llm_bench.cli --help` to confirm the installed local CLI. YAML fields such as `input_length`, `output_length`, `number_of_prompts`, `request_rate`, `maximum_concurrency`, `warmup_runs`, and `repetitions` are the source of truth; avoid adding undocumented positional overrides that would make the saved experiment incomplete.

The paired serving experiments additionally set `generation_config: vllm`, `temperature: 0.0`, `top_p: 1.0`, and `ignore_eos: true`. These controls are saved workload parameters, not hidden launcher defaults, and the comparison checks them.

All shipped pairs request `dtype: float16` so the first NVIDIA comparison has one explicit precision instead of two potentially different outcomes from `auto`. Verify that the allocated GPU and installed vLLM build support it. If another precision is appropriate, change both paired files together and rerun smoke validation before measuring.

## Step 1: run a local or interactive smoke test

Use a GPU compute node, not an ordinary login node. Within an interactive allocation obtained using site-approved resource values:

```bash
bash scripts/run_smoke_test.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_smoke.yaml
bash scripts/run_smoke_test.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_smoke.yaml
```

The first command is the lower-resource validation path; the second independently verifies that the 8B target fits and loads. Smoke configurations use one or a few prompts and short generation. Their priority is output-file and inference validation, not stable performance measurement.

Check the run rather than relying only on terminal output:

```bash
find "$RESULTS_ROOT" -maxdepth 3 -type f -name summary.json -print
python -m json.tool "$RESULTS_ROOT"/<date>/<run-id>/summary.json
```

## Step 2: submit a Slurm smoke test

The Slurm files contain no Sapienza partition, account, QoS, GPU type, or filesystem path. Replace placeholders below with values discovered in `01_CLUSTER_PREREQUISITES.md`; remove optional flags only when site policy says they are unnecessary:

```bash
# Required for these --export=ALL examples after authenticated pre-download.
unset HF_TOKEN

sbatch \
  --partition=<GPU_PARTITION> \
  --account=<ACCOUNT_IF_REQUIRED> \
  --qos=<QOS_IF_REQUIRED> \
  <SITE_APPROVED_ONE_GPU_OPTION> \
  --time=<TIME_LIMIT> \
  --export=ALL,BENCH_CONFIG=<ABSOLUTE_PATH_TO_SAPIENZA_ENV>,EXPERIMENT_CONFIG=<ABSOLUTE_PATH_TO_LLAMA32_1B_SMOKE_YAML> \
  slurm/smoke_test.sbatch
```

Replace `<SITE_APPROVED_ONE_GPU_OPTION>` with one complete option that the site documents as allocating exactly one GPU; do not assume a particular Slurm option family or GPU resource spelling. Repeat with `<ABSOLUTE_PATH_TO_LLAMA31_8B_SMOKE_YAML>`. Absolute configuration and experiment paths avoid dependence on the submission directory. `BENCH_CONFIG` selects the private cluster environment, `BENCH_REPO_ROOT` inside it locates the checkout, and `EXPERIMENT_CONFIG` selects one model/workload pair.

`--export=ALL` inherits all exported variables from the submission shell; merely omitting `HF_TOKEN` from the comma-separated additions does **not** exclude it. The commands therefore require `HF_TOKEN` to be unset after pre-downloading both snapshots. For an offline compute node, use the shared cache as described in the setup guide. If a compute node must authenticate, use only the site's approved secret-injection/export mechanism and do not place the token in `--export`, the cluster file, job arguments, or logs.

Monitor and inspect using the real job ID returned by `sbatch`:

```bash
squeue -j <JOB_ID>
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

## Step 3: run the offline throughput benchmark

The offline launcher invokes the throughput benchmark exposed by the installed vLLM version. It does not reimplement vLLM's scheduler or request generator.

Interactive examples:

```bash
bash scripts/run_offline_benchmark.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_offline.yaml
bash scripts/run_offline_benchmark.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_offline.yaml
```

Slurm examples:

The `unset HF_TOKEN` or site-approved secret-handling requirement from Step 2 applies to every `--export=ALL` command below.

```bash
sbatch \
  --partition=<GPU_PARTITION> \
  --account=<ACCOUNT_IF_REQUIRED> \
  --qos=<QOS_IF_REQUIRED> \
  <SITE_APPROVED_ONE_GPU_OPTION> \
  --time=<TIME_LIMIT> \
  --export=ALL,BENCH_CONFIG=<ABSOLUTE_PATH_TO_SAPIENZA_ENV>,EXPERIMENT_CONFIG=<ABSOLUTE_PATH_TO_LLAMA32_1B_OFFLINE_YAML> \
  slurm/offline_benchmark.sbatch

sbatch \
  --partition=<GPU_PARTITION> \
  --account=<ACCOUNT_IF_REQUIRED> \
  --qos=<QOS_IF_REQUIRED> \
  <SITE_APPROVED_ONE_GPU_OPTION> \
  --time=<TIME_LIMIT> \
  --export=ALL,BENCH_CONFIG=<ABSOLUTE_PATH_TO_SAPIENZA_ENV>,EXPERIMENT_CONFIG=<ABSOLUTE_PATH_TO_LLAMA31_8B_OFFLINE_YAML> \
  slurm/offline_benchmark.sbatch
```

The launcher performs configured warm-ups separately, then preserves every measured repetition. Warm-up observations are never merged into measured summaries. The deterministic seed is passed where the installed backend supports it. An unsuccessful or anomalous repetition remains recorded with status and warnings; it is not auto-deleted or silently replaced.

## Step 4: run the online serving benchmark

The serving workflow starts an OpenAI-compatible vLLM server and the installed vLLM serving benchmark client inside the same Slurm allocation:

```bash
bash scripts/run_serving_benchmark.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_serving.yaml
bash scripts/run_serving_benchmark.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_serving.yaml
```

Slurm examples use the same site placeholders:

The same `unset HF_TOKEN` or site-approved secret-handling requirement applies here.

```bash
sbatch \
  --partition=<GPU_PARTITION> \
  --account=<ACCOUNT_IF_REQUIRED> \
  --qos=<QOS_IF_REQUIRED> \
  <SITE_APPROVED_ONE_GPU_OPTION> \
  --time=<TIME_LIMIT> \
  --export=ALL,BENCH_CONFIG=<ABSOLUTE_PATH_TO_SAPIENZA_ENV>,EXPERIMENT_CONFIG=<ABSOLUTE_PATH_TO_LLAMA32_1B_SERVING_YAML> \
  slurm/serving_benchmark.sbatch

sbatch \
  --partition=<GPU_PARTITION> \
  --account=<ACCOUNT_IF_REQUIRED> \
  --qos=<QOS_IF_REQUIRED> \
  <SITE_APPROVED_ONE_GPU_OPTION> \
  --time=<TIME_LIMIT> \
  --export=ALL,BENCH_CONFIG=<ABSOLUTE_PATH_TO_SAPIENZA_ENV>,EXPERIMENT_CONFIG=<ABSOLUTE_PATH_TO_LLAMA31_8B_SERVING_YAML> \
  slurm/serving_benchmark.sbatch
```

The job writes `server.log`, polls a health endpoint with a bounded timeout, fails clearly if startup never completes, and terminates the background server through a shell trap on success, failure, or cancellation. Check the log if no client results appear. A manually interrupted interactive run should also be allowed to execute its cleanup trap; verify that no server process remains before starting another run.

Request rate and maximum concurrency describe different load controls. Preserve both in the experiment even if one represents an unbounded setting. Do not compare a saturated closed-loop run with a rate-limited run as though they were the same workload.

The serving generation controls make the requested decode work explicit:

- `generation_config: vllm` makes the server use vLLM's generation defaults instead of silently loading model-repository `generation_config.json` values that could differ between the targets.
- `temperature: 0.0` requests greedy token selection, while `top_p: 1.0` explicitly applies no nucleus-probability truncation. This reduces a hidden sampling difference but does not promise bitwise-identical execution across software or hardware versions.
- `ignore_eos: true` asks the client/server path not to end generation at an EOS token, making the requested output length a controlled amount of decode work. Actual output-token counts remain authoritative and are preserved because requests can still fail or terminate for another backend reason.

The launcher inspects the installed server's full-help interface (falling back to ordinary `vllm serve --help` for releases without it) and `vllm bench serve --help`, records those help texts, and requires the corresponding server and client flags. An older or otherwise incompatible vLLM release produces a clear failed run rather than silently omitting one of these controls.

## Step 5: change lengths, load, or repetitions safely

Create paired experiment copies:

```bash
cp configs/experiments/llama32_1b_serving.yaml configs/experiments/<1B_EXPERIMENT>.yaml
cp configs/experiments/llama31_8b_serving.yaml configs/experiments/<8B_EXPERIMENT>.yaml
```

Edit both copies together. Relevant fields include:

- `input_length` and `output_length`;
- `number_of_prompts` for offline work or the configured request count for serving;
- `request_rate` and `maximum_concurrency` for serving;
- `generation_config`, `temperature`, `top_p`, and `ignore_eos` for serving;
- `seed`;
- `warmup_runs` and `repetitions`;
- `dtype`, `quantization`, and `tensor_parallel_size`;
- `gpu_memory_utilization` and `telemetry_interval_ms`.

The initial examples request one GPU. Tensor parallelism is configurable for later single-node experiments, but multi-node inference is not implemented. A different precision or quantization changes both resource use and the comparison conditions; use it only when supported by both models and the installed backend, and record it accurately.

`gpu_memory_utilization` is also a comparison condition, not a measurement. vLLM can use that budget for runtime allocations and the KV cache, so the peak memory reported by `nvidia-smi` may largely reflect memory reserved under this policy. It is total sampled device memory, not a model-weight-only size or the minimum memory required to load the model. Keep the policy identical for a paired comparison and interpret it alongside input length, concurrency, and KV-cache demand.

## Step 6: inspect logs and result artifacts

Each run has a unique directory beneath the configured `RESULTS_ROOT`, normally:

```text
$RESULTS_ROOT/<date>/<run-id>/
├── experiment.yaml
├── metadata.json
├── summary.json
├── measurements.json           # stable per-repetition normalized observations
├── raw_vllm_output.json
├── stdout.log
├── stderr.log
├── server.log                 # serving only
├── vllm_serve_help.txt        # serving CLI contract used by this run
├── vllm_bench_serve_help.txt  # serving-client CLI contract used by this run
├── gpu_telemetry.csv          # when nvidia-smi telemetry works
└── system/
    ├── hostname.txt
    ├── lscpu.txt
    ├── numactl.txt
    ├── nvidia_smi.txt
    ├── nvidia_smi_query.csv
    ├── python_version.txt
    ├── pip_freeze.txt
    ├── environment_modules.txt
    └── slurm_environment.txt
```

`collect_system_info.sh` captures the reproducibility record, while optional `collect_gpu_telemetry.sh` samples only GPUs identified by an allocated-device visibility variable; it disables itself rather than sampling every GPU on an unscoped node. For smoke and offline runs, each measured telemetry window starts immediately before the corresponding vLLM throughput process and stops afterward, so it includes that process's initialization/model loading and inference but excludes warm-ups. For serving, the server is loaded first and telemetry covers each measured client workload while that server remains loaded; it excludes server/model startup and warm-ups. These different boundaries must be reported when interpreting power or energy, and neither path measures whole-node energy.

If `nvidia-smi` is absent, permission is denied, visibility is unscoped, or a device does not support a requested power or sensor field, telemetry fails gracefully. Optional files may be absent or contain a diagnostic; the summary uses a warning and `null` metrics, never a fabricated number. `raw_vllm_output.json` is retained because vLLM's output format can change; `summary.json` is the stable, hardware-neutral view.

When `repetitions` is greater than one, every explicitly supplied measured raw/telemetry file participates in normalization. Performance, latency, duration, utilization, and average-power fields are arithmetic means across available measured repetitions; `peak_gpu_memory_mib` is their maximum. `successful_requests`, actual input tokens, and actual output tokens are totals, and `energy_joules` is summed across the measured telemetry windows; energy-per-request and energy-per-output-token use those totals. Warm-up files are kept under `warmup/` but excluded. Missing, unreadable, or failed repetitions are retained, counted in `failed_repetitions`, warned about, and cause the run status to be `failed` rather than being silently dropped.

`measurements.json` keeps the corresponding normalized observation for every configured repetition,
including an explicit `unavailable` record when no readable measured raw output exists. Its file
references are relative to the run directory so copied result directories remain portable. The
aggregate `summary.json` remains authoritative for comparisons; repetition observations supply
spread and telemetry detail rather than changing aggregate semantics.

Useful checks are:

```bash
python -m json.tool "$RESULTS_ROOT"/<date>/<run-id>/metadata.json
python -m json.tool "$RESULTS_ROOT"/<date>/<run-id>/summary.json
tail -n 100 "$RESULTS_ROOT"/<date>/<run-id>/stderr.log
tail -n 100 "$RESULTS_ROOT"/<date>/<run-id>/server.log
```

Generated results under the checkout's `results/` directory are ignored and may be retained there, or `RESULTS_ROOT` may point to an external results area. In either case, do not commit generated runs. Model caches and weights must be outside the checkout. Preserve failed run directories; `status`, stderr, raw output, and metadata explain what happened.

## Step 7: complete two-model comparison workflow

Use this order for a first reproducible campaign:

1. Run `llama32_1b_smoke.yaml` and verify its expected files and completed status.
2. Run `llama31_8b_smoke.yaml` and verify its expected files and completed status.
3. Submit `llama32_1b_offline.yaml` and `llama31_8b_offline.yaml` as separate jobs.
4. Submit `llama32_1b_serving.yaml` and `llama31_8b_serving.yaml` as separate jobs.
5. Compare the saved `experiment.yaml` and `summary.json` files. Confirm matching lengths, count, rate, concurrency, seed, generation configuration, temperature, top-p, EOS policy, requested dtype, backend-reported precision when available, quantization, tensor parallelism, `gpu_memory_utilization`, warm-up policy, configured and measured repetitions, explicitly pinned model revisions, backend version, and relevant software versions. `dtype: auto` records a shared requested policy but is not itself a measured precision; without an observed `model_precision`, the comparison is partial and ratios are suppressed. For a formal comparison, select the same precision supported by both models and verify any observed `model_precision` reported by the backend.
6. Compare one explicit 1B summary with one explicit 8B summary:

   ```bash
   python scripts/compare_model_results.py \
     "$RESULTS_ROOT"/<date>/<1B-run-id>/summary.json \
     "$RESULTS_ROOT"/<date>/<8B-run-id>/summary.json \
     --output-dir "$RESULTS_ROOT"/<date>/<comparison-id>
   ```

   Result directories are also accepted instead of paths to `summary.json`:

   ```bash
   python scripts/compare_model_results.py \
     "$RESULTS_ROOT"/<date>/<1B-run-id> \
     "$RESULTS_ROOT"/<date>/<8B-run-id> \
     --output-dir "$RESULTS_ROOT"/<date>/<comparison-id>
   ```

7. Inspect both outputs:

   ```bash
   python -m json.tool "$RESULTS_ROOT"/<date>/<comparison-id>/comparison.json
   column -s, -t "$RESULTS_ROOT"/<date>/<comparison-id>/comparison.csv | less -S
   ```

The tool never discovers or chooses runs automatically, never selects a “best” repetition, and never drops failed runs. It records both identities and statuses. If compatibility fields differ, the output is `incompatible` or `partial`, lists the mismatches, and omits any headline winner or invalid ratios. A performance comparison says nothing about which model produces better answers.

Continue with [04 — Metrics and results](04_METRICS_AND_RESULTS.md).
