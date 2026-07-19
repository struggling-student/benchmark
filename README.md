# LLM inference benchmark for the Sapienza HPC cluster

This repository is a small, reproducible starting point for learning how to deploy and benchmark Large Language Model inference with Slurm and NVIDIA GPUs. The first backend is [vLLM](https://docs.vllm.ai/), and the initial study gives equal treatment to:

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`

The 1B model is useful both as a lower-resource environment check and as a genuine comparison target. Matching workloads for both models expose how model scale changes loading time, memory demand, latency, throughput, GPU utilization, and—where telemetry supports it—power and energy. These are inference-system measurements, not model-quality scores.

The longer-term MSc thesis objective is to compare ordinary CPU memory with CPU High Bandwidth Memory (HBM). Sapienza does not provide CPU HBM for this initial work, so the experiment and result formats are hardware-neutral even though this first phase establishes a single-GPU baseline.

> This repository is not MLPerf compliant, and its results must not be described as an official MLPerf submission. MLCommons/MLPerf integration is only a possible later phase.

## Repository map

- `configs/cluster/`: one documented example for site-specific settings; copy it locally and fill in values discovered on the cluster.
- `configs/experiments/`: matching smoke, offline, and serving configurations for both models.
- `scripts/`: environment checks, model download, benchmark launchers, telemetry, metadata collection, and comparison.
- `slurm/`: portable one-GPU Slurm entry points.
- `src/llm_bench/`: small configuration, metadata, and result-normalization package.
- `results/`: the permitted in-checkout location for ignored run output; only `.gitkeep` is tracked. Model caches and weights must still be outside the checkout.
- `docs/`: the complete workflow and future CPU-HBM roadmap.

## Shortest path to both smoke tests

From a clone, enter the repository, inspect the cluster, and replace every placeholder in a private, ignored copy of the example environment file. Set `BENCH_REPO_ROOT` to the checkout's absolute path; Slurm jobs require it because a submitted script may execute from Slurm's spool directory. Never put `HF_TOKEN` in that file.

```bash
git clone <REPOSITORY_URL> benchmark
cd benchmark
cp configs/cluster/sapienza.example.env configs/cluster/sapienza.env
# Edit configs/cluster/sapienza.env using values discovered from the cluster.
source configs/cluster/sapienza.env

bash scripts/create_venv.sh --config configs/cluster/sapienza.env
source "${VENV_PATH}/bin/activate"
# The script installs this repository with pip -e. Now install a vLLM build compatible
# with the detected Python/CUDA/driver environment, following the current official docs.

# On a network-enabled host permitted by site policy, first accept any gated-model
# terms in the browser, then enter a read token without echoing it.
read -r -s -p 'HF token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
python scripts/download_model.py --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_smoke.yaml
python scripts/download_model.py --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_smoke.yaml
unset HF_TOKEN

# Enter a one-GPU interactive allocation, reactivate the venv, then validate and run.
source configs/cluster/sapienza.env
source "${VENV_PATH}/bin/activate"
bash scripts/preflight_check.sh --config configs/cluster/sapienza.env
bash scripts/run_smoke_test.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_smoke.yaml
bash scripts/run_smoke_test.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_smoke.yaml
```

On the cluster, submit the same configurations through `slurm/smoke_test.sbatch` instead; site resource values are supplied with `sbatch`, not embedded in the job file. The detailed guide shows safe placeholder commands and offline-compute-node setup.

## Comparable two-model run

Use a matching pair of experiment files and do not change only one side:

```bash
bash scripts/run_offline_benchmark.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama32_1b_offline.yaml
bash scripts/run_offline_benchmark.sh --config configs/cluster/sapienza.env \
  --experiment configs/experiments/llama31_8b_offline.yaml

python scripts/compare_model_results.py \
  "$RESULTS_ROOT"/<date>/<1b-run-id>/summary.json \
  "$RESULTS_ROOT"/<date>/<8b-run-id>/summary.json \
  --output-dir "$RESULTS_ROOT"/<date>/<comparison-id>
```

The comparison writes `comparison.json` and `comparison.csv`. It reports configuration mismatches and will not present incompatible runs as a fair comparison. The shipped pairs explicitly request `float16` as a common NVIDIA baseline; confirm support on the allocated GPU, or change both sides together. Serving benchmarks use the two corresponding `*_serving.yaml` files in exactly the same way. Those paired configurations explicitly disable model-repository generation defaults and fix sampling/EOS controls so the two models receive the same requested decode workload; the launcher fails rather than silently dropping a control that the installed vLLM CLI does not support.

## Guides

1. [Inspect the cluster from step zero](docs/01_CLUSTER_PREREQUISITES.md)
2. [Create the environment and obtain both models](docs/02_ENVIRONMENT_AND_MODEL_SETUP.md)
3. [Run smoke, offline, and serving benchmarks](docs/03_RUNNING_THE_BENCHMARKS.md)
4. [Interpret metrics and compare the models](docs/04_METRICS_AND_RESULTS.md)
5. [Extend the same schema toward CPU HBM](docs/05_CPU_HBM_ROADMAP.md)
