# 05 — CPU-HBM roadmap

The thesis direction is CPU inference on systems equipped with High Bandwidth Memory (HBM). The Sapienza cluster in scope does not provide CPU HBM, so that phase will require access to suitable future hardware. This repository intentionally implements only the first, single-GPU phase now. The result schema and experiment vocabulary are hardware-independent so later CPU work can be added without replacing the baseline or building a generic orchestration platform.

## Phase 1: reproducible single-GPU baseline

- Establish the complete workflow on the Sapienza Slurm cluster: environment discovery, model access, smoke validation, measured repetitions, raw output, metadata, optional telemetry, and comparison.
- Run `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct` with equivalent offline and serving workloads.
- Record successful and failed loading, latency, throughput, memory, utilization, and available power/energy data.
- Keep tensor parallelism configurable but limit the initial examples to one GPU and no multi-node inference.

This phase provides the reference methodology and exposes infrastructure errors with the smaller model before interpreting the larger model's resource demands.

## Phase 2: ordinary CPU-DDR baseline

- Select an explicitly supported CPU inference path.
- Express the same prompt lengths, requested output lengths, request counts, rates/concurrency, seeds, warm-ups, and repetitions with the existing experiment vocabulary.
- Populate the existing hardware-neutral result fields with `hardware_type: cpu`; leave accelerator-only fields `null` with warnings.
- Measure CPU topology, placement, memory footprint, latency, throughput, bandwidth, and available node/package energy.

CPU-DDR results establish the control needed to attribute later changes to HBM rather than merely to a different processor, backend, precision, or workload.

## Phase 3: add a selected CPU backend

Add a supported vLLM CPU backend or another explicitly selected CPU inference backend after checking its current platform support. Keep the adapter thin: invoke the backend, preserve its raw output, and normalize only supported observations into the same `summary.json` schema.

Do not force GPU concepts onto CPU runs. Backend-specific settings can be recorded as metadata, while shared concepts—model identity, workload, latency, throughput, status, power, and energy—retain the same meaning. Any comparison across backends must report the backend/version difference and may be partial even when the workload matches.

## Phase 4: CPU-HBM experiments

Run the Phase 2 workload schema on CPU-HBM hardware and compare HBM with DDR. To isolate the memory-system effect, keep processor generation, backend, model revision, tokenizer, precision, thread/process placement, workload, and software versions controlled wherever the platform permits.

The CPU-HBM design should record or control:

- CPU model and ISA features;
- BF16, INT8, or other precision and kernel support;
- socket and NUMA-node counts;
- thread count;
- thread affinity;
- process placement;
- memory binding;
- HBM versus DDR allocation;
- HBM cache/flat modes when the platform provides them;
- prompt length, output length, batch size, and concurrency;
- separate prefill and decode behavior;
- model-weight and KV-cache memory footprint;
- measured memory bandwidth and the measurement tool/boundary;
- latency and token/request throughput;
- power and energy when valid instrumentation is available.

Placement is part of the experiment, not incidental metadata. On a multi-socket NUMA system, remote memory, thread migration, or an unintended HBM/DDR policy can dominate the result. Capture commands and effective placement, and fail or warn when the requested policy was not applied.

Prefill and decode should be analyzed separately where the backend exposes enough information. Prefill can exploit broad token-level parallelism; autoregressive decode may be especially sensitive to repeated model-weight reads, KV-cache traffic, and achievable memory bandwidth. Prompt length and batch/concurrency must therefore be swept deliberately rather than collapsed into one “CPU performance” number.

### Dashboard preview before CPU implementation

The dashboard includes a deterministic, in-memory CPU-DDR/HBM study so the presentation and
experimental controls can be reviewed before a CPU backend or HBM machine is available. It
exercises memory type and mode, batch size, thread/process placement, NUMA and memory binding,
instrumentation scope, CPU utilization and memory, package power, measured bandwidth, and separate
prefill/decode throughput. Every preview run is visibly marked simulated, is never written under
the results root, and must not be cited as benchmark evidence.

The CPU memory-study view treats DDR versus HBM as an explicit treatment comparison: model, CPU,
backend, workload, precision, placement, and instrumentation must match before ratios are shown.
The ordinary comparison lens continues to treat memory-system differences as incompatibilities.
This presentation contract does not replace the future result-schema and backend implementation.

## Reusing the schema across hardware

GPU, CPU-DDR, and CPU-HBM runs should use the same top-level experiment and result schema:

- explicit model/revision/tokenizer and precision identity;
- explicit input/output lengths and load controls;
- separate warm-ups and measured repetitions;
- backend and software versions;
- hardware-neutral latency, throughput, status, and warning fields;
- platform-specific metadata in applicable fields;
- `null` plus a warning for unavailable observations.

For GPUs, accelerator name/count and sampled GPU memory/utilization/power are populated. For CPUs, CPU model, sockets, NUMA nodes, placement, memory type, and CPU/platform instrumentation become central. Cross-hardware comparisons must state backend and instrumentation differences; schema compatibility does not by itself guarantee experimental compatibility.

## Optional MLCommons/MLPerf direction

MLCommons/MLPerf is a later option, not part of the current implementation. A future phase may:

- adopt LoadGen-style workload generation;
- compare the research workloads with standardized scenarios;
- improve benchmark discipline and auditability;
- prepare a compliant implementation as a separate, explicitly scoped effort if required.

The current repository does not vendor, clone, reimplement, or contain placeholder MLPerf code. Its results are not MLPerf compliant and must not be represented as official submissions. Compliance would require following the then-current rules, approved implementations, accuracy requirements, audit process, and submission procedures rather than applying MLPerf terminology to these baseline measurements.
