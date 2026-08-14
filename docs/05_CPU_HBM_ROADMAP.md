# 05 — CPU-HBM roadmap

The repository now has a real llama.cpp CPU path, including CPU execution profiles, native/container
providers, process-tree telemetry, RAPL package-power sampling where permitted, and CPU-aware result
and dashboard fields. That establishes an ordinary CPU-memory baseline. The remaining thesis step is
to run it on a suitable CPU-HBM platform and add platform-specific memory placement verification.

## What is implemented now

- The same two model manifests and smoke/offline/serving workloads compose with vLLM GPU,
  llama.cpp CPU, and llama.cpp CUDA/hybrid profiles.
- Shared API workloads use identical persisted prompts and generation controls.
- CPU profiles record threads, batch threads, affinity mask, NUMA policy, memory binding, batch and
  microbatch sizes, parallel slots, and GPU offload layers.
- CPU utilization/RSS and optional package power share schema 2.0 with GPU telemetry.
- The dashboard understands CPU/GPU/hybrid runs and has a simulated DDR/HBM presentation contract
  for fields that cannot yet be measured on the available cluster.

Native `llama-bench` offline and the shared API harness answer different questions. Use the shared
API method for a controlled vLLM-versus-llama.cpp backend treatment. Use llama-bench for detailed
backend-native CPU throughput, retaining its measurement scope and excluded tokenization/sampling.

## CPU-DDR baseline campaign

Before access to HBM hardware, establish CPU-DDR repetitions for both model scales and relevant
variants. Keep source revision, tokenizer, workload, thread placement, NUMA policy, software build,
and instrumentation boundary fixed. F16, Q8_0, and Q4_K_M are separate treatments; do not attribute
their differences solely to hardware or backend.

Record at least:

- CPU model/ISA, socket and NUMA topology;
- effective thread affinity and process placement;
- memory binding and page policy;
- model/artifact revision, precision, quantization, size, and checksum;
- prompt/output lengths, batch/microbatch, concurrency, warm-ups, and repetitions;
- prefill/decode or shared-API latency/throughput according to method;
- process RSS, CPU utilization, package power/energy, and measured memory bandwidth when available.

Placement is part of the experiment. Remote memory, thread migration, transparent page movement, or
an unintended socket can dominate inference results.

## CPU-HBM campaign

Run the same CPU profile/workload matrix on a platform where HBM and ordinary DDR placement can be
selected and verified. To interpret HBM as the treatment, hold constant wherever possible:

- processor generation and enabled ISA;
- llama.cpp commit/build flags and provider;
- model source revision, tokenizer, GGUF variant, and checksum;
- thread count, affinity, socket, NUMA, and process count;
- prompt/output lengths, batch/microbatch, slots, rate, and concurrency;
- warm-up/cache policy and instrumentation boundary.

Add platform adapters only for facts the machine can prove: HBM/DDR mode, flat/cache mode, effective
allocation nodes, capacity, achieved bandwidth, and memory-controller evidence. Missing permissions
remain null with warnings. A requested `numactl` policy is not proof that pages landed in HBM; formal
runs need an effective-placement check from the target platform.

Prefill and decode should be analyzed separately when supported. Prefill exposes broad matrix
parallelism, while autoregressive decode repeatedly reads weights and KV-cache state and can be more
sensitive to achievable memory bandwidth. Sweep prompt length and batch/concurrency deliberately.

## Dashboard and comparison extension

The current memory-study view treats DDR versus HBM as an explicit treatment and only shows ratios
when model, CPU, backend, workload, precision, placement, and instrumentation match. Its demo data is
simulated and cannot be cited as benchmark evidence.

When real HBM results arrive, populate the existing optional memory fields and add only the
platform-specific evidence needed to validate allocation. Schema compatibility alone does not make
cross-machine or cross-instrumentation results comparable.

## Optional MLCommons direction

MLPerf remains a separate possible later phase. A compliant effort would need the then-current rules,
approved implementation, accuracy methodology, LoadGen scenarios, audit process, and submission
requirements. These benchmark results must not be represented as official MLPerf results.
