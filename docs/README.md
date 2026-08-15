# Documentation

The full documentation for the LLM Inference Benchmark. Follow the guides in order the
first time; each one builds on the previous.

| Guide | Purpose |
| --- | --- |
| [01 — Cluster prerequisites](01_CLUSTER_PREREQUISITES.md) | Discover a Slurm site from zero: partitions, accounts, GPUs, modules, Python/CUDA, storage, and network policy, then write the private cluster config. |
| [02 — Environment, providers, and model preparation](02_ENVIRONMENT_AND_MODEL_SETUP.md) | Create the Python environment, configure native/container providers, obtain the gated models, and prepare reproducible GGUF artifacts. |
| [03 — Running the benchmark matrix](03_RUNNING_THE_BENCHMARKS.md) | Compose models, workloads, profiles, providers, and variants; run locally or through Slurm; understand the result directory and comparison rules. |
| [04 — Metrics, results, and comparison validity](04_METRICS_AND_RESULTS.md) | Interpret the shared-API and backend-native measurements, telemetry, provenance, comparison lenses, and the Streamlit dashboard. |
| [05 — CPU-HBM roadmap](05_CPU_HBM_ROADMAP.md) | The CPU-DDR baseline and the path toward CPU High Bandwidth Memory studies. |

## Getting started

For the shortest end-to-end path, start with the
[Quick start on a cluster](../README.md#quick-start-on-a-cluster) section in the README,
then return here for the reasoning behind each step.

## Writing style

- Commands are examples to run, not claims about any particular cluster.
- Never invent partition names, module names, GPU types, or filesystem paths.
- Values recorded in the private cluster config are the source of truth, never ambient
  shell state.