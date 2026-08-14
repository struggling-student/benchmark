# 04 — Metrics, results, and comparison validity

`summary.json` schema 2.0 is the stable, backend-neutral view. Raw backend/harness JSON is always
retained as evidence. An unsupported observation is `null` with a warning; it is never guessed from
requested lengths or translated from an unrelated metric.

## Shared API measurement

Smoke and serving use the same repository-owned streaming completion client for both backends. A
deterministic `workload_manifest.json` stores exact prompt text, token counts, token-ID hashes,
tokenizer identity, seed, and a canonical manifest SHA-256.

For every request, the raw harness stores success/error status, HTTP status, requested and actual
token counts, time to first token (TTFT), time per output token (TPOT), defensible inter-token latency
(ITL), end-to-end latency, and response evidence. Aggregate fields include:

- successful/failed requests and request throughput;
- actual input, output, and total token throughput;
- mean/median/p95/p99 TTFT and TPOT;
- ITL only when stream chunks can be shown to map one-to-one to output tokens;
- mean and p95 client-observed end-to-end latency.

TTFT includes client-visible queuing, tokenization/server processing, prefill, scheduling, and first
decode work. TPOT uses the interval after the first observed content and actual output-token usage.
E2E covers request submission through stream completion. Campaign `duration_seconds` is not a
per-request latency.

If the server omits streaming usage, actual token counts and token throughput are null. If chunks do
not establish token boundaries, ITL is null. A different actual output length is retained and warned
about even though `ignore_eos` and `max_tokens` request fixed generation.

## Backend-native offline measurement

Offline results are intentionally method-specific:

- vLLM records `measurement_method: vllm_bench_throughput` and scope
  `vllm_bench_throughput_native`.
- llama.cpp records `measurement_method: llamacpp_bench` and scope
  `model_forward_only_excludes_tokenization_and_sampling`.

`llama-bench` performs its own discarded warm-up and excludes tokenization/sampling. Configured
outer warm-ups and measured repetitions are separate. Because native boundaries differ, offline
vLLM/llama.cpp ratios are prohibited; the results can still be shown side by side descriptively.

## Telemetry

Telemetry runs only around measured repetitions. Native and Apptainer paths sample the process tree's
CPU utilization and resident memory. Docker uses container statistics. GPU/hybrid profiles sample
only explicitly visible NVIDIA devices; unscoped node-wide devices are not silently attributed to a
job. CPU package power uses readable Linux RAPL package counters when available.

Summary fields include peak CPU/GPU memory, average CPU/GPU utilization and power, sampled energy,
and optional measured memory bandwidth. Missing tools, unsupported sensors, and denied permissions
leave the relevant fields null. Energy is integrated only across the instrumentation window and is
not whole-node energy unless the recorded scope explicitly establishes that boundary.

Warm-ups are excluded. API server/model initialization is also excluded from serving telemetry and
is recorded separately as `model_load_time_seconds`. Native offline telemetry surrounds the native
benchmark process, so its boundary is different and is recorded as such.

## Provenance and self-contained runs

Every run records:

- hashes of the source model/workload/profile YAML and the full resolved experiment;
- canonical and resolved model/tokenizer revisions;
- artifact format, variant, source revision, file SHA-256, and quantization;
- backend/profile/provider and backend/native binary or container image identity;
- Docker container ID or Apptainer image checksum where applicable;
- CPU/GPU/hybrid classification, topology, thread/NUMA/affinity/memory settings, and GPU layers;
- workload/generation controls and the prompt-manifest hash;
- measurement method/scope, telemetry/energy scope, status, failures, and warnings.

`measurements.json` provides one normalized record per configured repetition. Summary aggregation
uses the arithmetic mean for ordinary performance/latency/utilization/power metrics, maximum for
peak memory, and sums for token/request counts and energy. Missing or unreadable repetitions remain
visible and make the run fail rather than being silently discarded.

The catalog accepts schema 1.0, 1.1, and 2.0 without rewriting old result directories. Legacy
`raw_vllm_output*.json` discovery remains supported.

## Comparison lenses

Ordinary like-for-like comparison treats backend/version as controlled fields. It checks model and
tokenizer identity/revision, precision, quantization, workload shape, request controls, software,
hardware, provider, repetitions, and measurement scopes before emitting ratios.

The dashboard's backend-treatment lens instead permits backend/version to be the intended
difference. Ratios are emitted only for completed shared-API F16 runs when all of the following
evidence matches:

- canonical model and immutable source revision;
- tokenizer and tokenizer revision;
- provider type and hardware identity/topology;
- workload-manifest hash and requested workload controls;
- actual input/output token counts;
- sampling, EOS, seed, rate, concurrency, warm-up, and repetition settings;
- shared measurement method and client-observed scope.

CPU-versus-GPU, provider mismatches, quantized-versus-F16, and native offline pairs are descriptive
or incompatible. Quantized results remain scientifically useful, but they combine backend and
numeric-representation changes and cannot isolate the backend effect.

## Streamlit dashboard

```bash
pip install -e ".[dashboard]"
llm-bench dashboard --results-root "$RESULTS_ROOT"
```

The read-only dashboard filters by backend, execution profile, provider, artifact variant, hardware,
measurement method, model, workload, and status. Run detail places provenance and comparison-validity
evidence next to performance. The demo source includes representative vLLM and llama.cpp CPU/GPU
runs and is always labelled simulated.

Prefer copying results to a workstation. If cluster policy permits a login-node dashboard, bind only
to loopback and use SSH port forwarding; Streamlit never submits or controls jobs.

Continue with [05 — CPU-HBM roadmap](05_CPU_HBM_ROADMAP.md).
