# 04 — Metrics and results

The normalized `summary.json` preserves a stable, hardware-independent vocabulary while `raw_vllm_output.json` preserves the installed vLLM version's original result. A missing observation is `null` with an explanation in `warnings`; it is never estimated or fabricated.

Always interpret a metric together with model identity, tokenizer, backend/software versions, hardware, precision, quantization, and workload. Similar-looking numbers from different conditions are not automatically comparable.

## Latency metrics

### Time to first token (TTFT)

TTFT is the elapsed time from submitting a serving request until the first generated token becomes available to the client. It includes queuing, prompt tokenization where measured by the client path, prompt processing (prefill), scheduling, and initial decode work. The summary can report mean, median, p95, and p99 TTFT in milliseconds.

TTFT is chiefly an online/streaming metric. An offline throughput CLI may not expose it; in that case the corresponding fields remain `null` with a warning.

### Time per output token (TPOT)

TPOT describes the average incremental time spent per generated token after the first token. A common per-request formulation divides the time from first token to final token by the number of intervals after the first token. Exact endpoint handling can vary by backend, so preserve the vLLM definition and version. Mean, median, p95, and p99 TPOT are reported when vLLM exposes them.

TPOT is not simply end-to-end latency divided by requested output length: that would fold prompt processing, queuing, and the first-token path into every token.

### Inter-token latency (ITL)

Some installed vLLM versions call the observed delay between successive streamed tokens inter-token latency (ITL). ITL describes individual gaps, while TPOT is usually an aggregate per request. They are related but may use different samples or weighting. Record the backend's original terminology and do not relabel an ITL percentile as a TPOT percentile unless the installed CLI explicitly defines it that way.

### End-to-end latency

End-to-end (E2E) latency is the client-observed time from request submission to completion. It includes queuing, prefill, all decode steps, and protocol/client overhead inside the measured boundary. The schema records mean and p95 E2E latency when available. A run's overall `duration_seconds` is campaign wall time and is not the same as per-request E2E latency.

Percentiles describe the distribution across requests, not measurement confidence. p95 means that roughly 95% of observed values are at or below that value; it does not mean “95% accurate.”

### Model loading time

`model_load_time_seconds` is the measured time spent initializing the model for the execution path when that boundary can be observed reliably. It can include reading weights, allocating device memory, and backend initialization. Filesystem cache state and prior process state can change it, so record warm/cold-cache policy and do not substitute total Slurm job duration. If the installed vLLM path does not expose a defensible boundary, store `null` and explain why.

## Throughput metrics

- **Request throughput** (`request_throughput_requests_per_second`) is completed requests divided by the relevant measured duration.
- **Input-token throughput** (`input_throughput_tokens_per_second`) is processed prompt tokens per second.
- **Output-token throughput** (`output_throughput_tokens_per_second`) is generated tokens per second.
- **Total-token throughput** (`total_throughput_tokens_per_second`) combines input and output tokens only when the backend supplies enough information and uses a consistent measured interval.

Requested token counts and actual token counts can differ because of tokenization, stop conditions, end-of-sequence policy, request failures, and length limits. The serving examples set `ignore_eos: true` to request the same amount of decode work from both models, but actual counts remain the measurement authority. Preserve the backend's counts and definitions. Do not derive output throughput from requested lengths when the actual count is unknown.

Throughput and latency interact: raising concurrency may improve aggregate throughput while worsening TTFT or E2E latency. A fair report retains both rather than reducing a run to one score.

## GPU memory, utilization, power, and energy

Optional `nvidia-smi` telemetry samples timestamp, GPU index/UUID, utilization, memory utilization, used and total memory, power draw, clocks, and temperature where the device and permissions support them.

- **Peak GPU memory** is the highest sampled used-memory value, in MiB, summed across the visible allocated GPUs at each sample. Sampling can miss a short-lived peak and includes other visible processes if the GPU is shared. It is not a model-weight-only footprint: it can include weights, vLLM runtime allocations, reserved memory, and a KV cache sized under `gpu_memory_utilization`. Consequently, similar utilization budgets can reserve substantial memory even for different model sizes; the number is neither the minimum memory required to load the model nor an isolated measure of parameter storage.
- **Average GPU utilization** is the mean of valid sampled utilization values over the chosen interval. It is a device-busy indicator, not an exact percentage of theoretical FLOP/s.
- **Power** is an instantaneous rate of energy use, measured in watts (joules per second). `average_gpu_power_watts` summarizes valid samples; it is not energy.
- **Energy** is power integrated over time, measured in joules. With sampled telemetry it is approximated by time integration between valid power observations. Unsupported or missing power samples make energy unavailable rather than zero.
- **Energy per request** divides measured run energy by successfully completed measured requests when the boundaries align.
- **Energy per generated token** divides measured run energy by actual generated tokens when both values are available.

The implemented telemetry boundaries exclude warm-ups. Smoke/offline telemetry surrounds each measured vLLM throughput process, including its initialization/model-loading and inference work. Serving telemetry begins only after the server is healthy and surrounds each measured client workload, so it excludes server/model startup while including the loaded server's activity during requests. Client overhead outside GPU power is not measured, and GPU-only telemetry is not whole-node energy. These boundaries make energy from the two workflows different quantities; do not compare them as though they cover the same lifecycle. DCGM or platform power instrumentation may be considered later, but neither is required in this repository.

## Why workload parameters matter

Every interpretation needs at least prompt/input length, requested output length, actual request or prompt count, concurrency, request rate, random seed, generation defaults, sampling/EOS controls, warm-up policy, configured and measured repetitions, tensor parallelism, precision, quantization, `gpu_memory_utilization`, and telemetry interval/boundary.

Prompt processing and token generation exercise hardware differently:

- **Prefill** processes many prompt tokens together. It often exposes parallel matrix-compute throughput and strongly influences TTFT, especially for long prompts.
- **Decode** produces tokens autoregressively while repeatedly reading model state and using the KV cache. It often has less parallel work per request and is sensitive to memory traffic, batch/concurrency, and cache footprint, strongly influencing TPOT.

Longer prompts therefore do not stress the system in the same way as longer outputs. Batch size/concurrency changes scheduling and memory demand. Request rate changes queueing. These parameters cannot be normalized away after a run; they must be controlled before measurement.

Warm-up runs populate caches and exercise one-time initialization but are not measured repetitions. Model loading time is recorded separately where the execution path exposes it. Repetition-level failures or anomalies remain visible; the repository does not automatically discard them.

For multiple measured repetitions, the normalizer uses every explicitly supplied readable raw/telemetry file. It takes arithmetic means for performance, latency, duration, utilization, and average-power metrics; takes the maximum sampled peak-memory value; sums successful requests and actual input/output token counts; and sums energy across repetition telemetry windows. Energy per request and per generated token are then computed from the summed energy and actual totals. Warm-ups do not enter these aggregates. Missing, unreadable, or failed repetitions remain recorded, increment `failed_repetitions`, and make the run status `failed` rather than being silently excluded.

## Result identity and compatibility

Each `summary.json` records model identity directly rather than relying on its directory name:

- `model_id`, configured `model_revision`, locally observed `resolved_model_revision`, `tokenizer_id`, `resolved_tokenizer_revision`, and explicitly configured `model_parameter_scale`;
- `model_precision` from an explicit dtype (or a backend observation when available), requested `dtype`, and `quantization`;
- backend and backend version;
- Git commit and relevant installed software;
- Slurm job, host, hardware type, accelerator or CPU identity, sockets, and NUMA nodes;
- workload—including explicit serving generation/sampling/EOS controls—and measured metrics;
- status and warnings.

A controlled comparison requires the intended model difference to be explicit while model revisions, tokenizer choice, precision, workload, and generation settings are controlled as far as the two models permit. The example configured `model_revision: null` remains `null`. At run initialization, the repository attempts to read a valid resolved commit from local Hugging Face cache refs into `resolved_model_revision` and `resolved_tokenizer_revision`; unavailable observations stay `null` with a warning, and no revision is fabricated. Pin explicit immutable revisions before a formal campaign rather than relying only on a mutable cache ref, cache path, or result-directory name.

The shipped pairs request `dtype: float16` as an explicit common NVIDIA baseline; because vLLM applies an explicit dtype to model weights and activations, it is also recorded as `model_precision`. Verify support on the allocated hardware and change both paired files together if another precision is required. By contrast, `dtype: auto` is only a requested backend policy, not a measurement of effective precision. `model_precision` remains `null` unless the installed backend's raw output reports it. Two matching `auto` strings with no observed precision therefore produce only a partial comparison and no ratios. For a formal fair comparison, explicitly select the same dtype supported by both targets; if the backend reports effective precision, verify that it is present and consistent in both runs.

Generation settings are part of the serving workload, not model-quality tuning in this benchmark. `generation_config: vllm` instructs the server not to import potentially different defaults from each model repository. The client explicitly passes `temperature: 0.0`, `top_p: 1.0`, and `ignore_eos: true` so token sampling and EOS stopping do not vary implicitly between the paired examples. These values reduce hidden workload differences; they do not make the models produce equivalent text or establish language quality. The launcher gates them against runtime CLI help and fails if the installed vLLM version cannot apply them.

`max_model_len` is also an explicit workload/runtime control. It bounds the engine's KV-cache context allocation, must be at least `input_length + output_length`, is passed to both offline and serving engines, and is recorded as a comparison-critical summary field. The paired examples use the same value so a model's repository-level maximum context cannot silently change memory allocation or make one side of a comparison unrunnable on the selected GPU.

Because the two targets ordinarily use their associated tokenizers, equal nominal token lengths do not guarantee identical source text or token content; preserve tokenizer identity and state what is controlled. A tokenizer difference must be reported, not hidden. Exact model revisions also matter.

The comparison command checks, at minimum:

- backend and backend version;
- benchmark type;
- input length and requested output length;
- prompt/request count;
- request rate and maximum concurrency;
- generation-configuration source, temperature, top-p, and EOS policy;
- GPU memory-utilization policy;
- seed and tensor parallel size;
- requested dtype and quantization (with backend-observed precision verified separately when available);
- warm-up policy, configured and successfully measured repetition counts, and telemetry interval;
- CPU model, socket/NUMA topology, and accelerator identity/count where applicable;
- relevant software versions.

It also reports failed status and missing metrics. A mismatch yields an incompatible or partial comparison with an explicit mismatch list. The tool does not silently normalize conditions, select a best run, discard a failure, or announce a headline winner.

## Comparing the two models

The initial comparison asks how `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct` behave as inference-system workloads. It is **not** a language-quality evaluation and cannot determine which model is more accurate, useful, safe, or capable. Quality evaluation would need separate datasets, scoring methodology, and controls outside this initial benchmark.

For equivalent smoke, offline, and serving configurations, examine:

- whether each model loads successfully;
- model loading time;
- peak GPU memory;
- TTFT;
- TPOT (and backend-defined ITL where applicable);
- end-to-end latency;
- request throughput;
- output-token throughput;
- average GPU utilization;
- average GPU power;
- energy per request;
- energy per generated token.

The 1B model is expected to require fewer resources than the 8B model, which makes it a useful infrastructure check and scale comparison. That expectation is not a result. Report measured values, unavailable metrics, failures, uncertainty, and workload conditions rather than embedding an expected performance conclusion.

A valid pair uses equivalent input lengths, requested output lengths, counts, concurrency, request rate, seed, generation-configuration source, temperature, top-p, EOS policy, precision, quantization, tensor parallelism, warm-ups, configured and measured repetitions, and software/hardware environment. Model and tokenizer identities are recorded explicitly. If a model requires a different setting, document the reason and treat the output as partial or incompatible rather than a fair head-to-head comparison.

`comparison.json` contains run/model identities, compatibility, mismatches, absolute values, valid model-to-model ratios, missing-metric warnings, and failed-run information. `comparison.csv` is a small tabular view of the same comparison. A ratio needs an explicit direction (for example, 8B value divided by 1B value); a larger ratio is not universally better because lower latency/energy and higher throughput have opposite desirability.

Continue with [05 — CPU-HBM roadmap](05_CPU_HBM_ROADMAP.md).
