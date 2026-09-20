# Documentation

The full documentation for the LLM Inference Benchmark. Follow the guides in order the
first time; each one builds on the previous.

| Guide | Purpose |
| --- | --- |
| [01 — Cluster prerequisites](01_CLUSTER_PREREQUISITES.md) | Discover a Slurm site from zero: partitions, accounts, GPUs, modules, Python/CUDA, storage, and network policy, then write the private cluster config. |
| [02 — Environment, providers, and model preparation](02_ENVIRONMENT_AND_MODEL_SETUP.md) | Create the Python environment, configure native/container providers, obtain the gated models, and prepare reproducible GGUF artifacts. |
| [03 — Running the benchmark matrix](03_RUNNING_THE_BENCHMARKS.md) | Compose models, workloads, profiles, providers, and variants; run locally or through Slurm; understand the result directory and comparison rules. |
| [04 — Metrics, results, and comparison validity](04_METRICS_AND_RESULTS.md) | Interpret the shared-API and backend-native measurements, telemetry, provenance, and comparison validity. |
| [05 — CPU-HBM implementation and roadmap](05_CPU_HBM_ROADMAP.md) | ISA/runtime selection, HBM mode verification, the CPU-DDR baseline, and the remaining CPU-HBM campaign work. |
| [06 - Literature result baselines](06_LITERATURE_RESULTS.md) | Published measurements for every shipped literature workload, experimental conditions, comparability limits, and the table to fill with cluster results. |
| [07 — KV-cache capacity-cliff experiments](07_KV_CACHE_CAPACITY_EXPERIMENTS.md) | Stress the 64 GiB/socket HBM budget with real vLLM serving (same model, no code change): does cache mode's cliff show up in real throughput, and does flat mode hit a hard capacity wall. |

## Getting started

For the shortest end-to-end path, start with the
[Quick start on a cluster](../README.md#quick-start-on-a-cluster) section in the README,
then return here for the reasoning behind each step.

## Writing style

- Commands are examples to run, not claims about any particular cluster.
- Never invent partition names, module names, GPU types, or filesystem paths.
- Values recorded in the private cluster config are the source of truth, never ambient
  shell state.
