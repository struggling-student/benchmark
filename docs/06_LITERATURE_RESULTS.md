# 06 - Literature result baselines

This document records the published results behind every workload in
`configs/workloads/`. It is the reference sheet for comparing future cluster runs with the
literature. Values are transcribed from the cited primary sources unless explicitly marked
`derived`, `approximately read from figure`, or `qualitative only`.

The token shape is only one part of an experiment. A paper value is a reproduction target only
when model, precision, batch or concurrency, backend, hardware count, and measurement boundary
also match. Otherwise it is a contextual baseline and must not be presented as a speedup or
slowdown of our system.

## Coverage and comparison status

| Workload | Published result available | Best use of the result |
| --- | --- | --- |
| [`fixed_32_32`](../configs/workloads/fixed_32_32.yaml) | Exact latency table from Shen et al. | CPU decode-latency sanity band |
| [`fixed_128_32`](../configs/workloads/fixed_128_32.yaml) | Exact normalized ranges and selected ratios from Na et al. | Xeon Max and CPU/GPU trend comparison |
| [`fixed_256_32`](../configs/workloads/fixed_256_32.yaml) | Exact FlexGen throughput table; Na et al. plot | Shape-matched throughput context |
| [`fixed_512_32`](../configs/workloads/fixed_512_32.yaml) | Exact FlexGen throughput table; Na et al. plot | Shape-matched throughput context |
| [`fixed_1024_32`](../configs/workloads/fixed_1024_32.yaml) | Exact FlexGen throughput table; Na et al. plot | Shape-matched throughput context |
| [`fixed_128_128`](../configs/workloads/fixed_128_128.yaml) | Exact FlexGen throughput table | Shape-matched throughput context |
| [`fixed_512_8`](../configs/workloads/fixed_512_8.yaml) | Exact FlexGen throughput table | Shape-matched throughput context |
| [`fixed_512_128`](../configs/workloads/fixed_512_128.yaml) | Exact Kurt and THInfer throughput tables | Closest llama.cpp quantization baseline; HPC system context |
| [`fixed_1024_128`](../configs/workloads/fixed_1024_128.yaml) | Exact THInfer throughput tables | HPC system context |
| [`fixed_30000_10000`](../configs/workloads/fixed_30000_10000.yaml) | Normalized simulator results from Fang et al. | HBM-placement headroom, not absolute throughput |

## 32 input / 32 output - Shen et al.

Source: [Efficient LLM Inference on CPUs, arXiv:2311.00502v2, Table 3](https://arxiv.org/html/2311.00502).

The paper uses a single socket of a 4th-generation Intel Xeon Scalable processor and a custom
CPU runtime. Its exact CPU SKU is not reported. Models use INT4 weight-only quantization. The
metric is average next-token generation latency under a proxy workload with 32 input and 32
output tokens. Group size refers to the INT4 weight-quantization group, not request batch size.

| Model | LLM Runtime INT4, group 32 (ms/token) | LLM Runtime INT4, group 128 (ms/token) | ggml INT4, group 32 (ms/token) |
| --- | ---: | ---: | ---: |
| GPT-J 6B | 22.99 | 19.98 | 31.62 |
| Llama-2 7B | 23.40 | 21.96 | 27.71 |
| LLaMA 7B | 23.88 | 22.04 | 27.20 |
| GPT-NeoX 20B | 80.16 | 61.21 | 92.36 |
| Falcon 7B | 31.23 | 22.26 | 36.22 |

For convenience, the following rates are derived as `1000 / milliseconds_per_token`. They are
single-stream equivalents, not measured aggregate throughput.

| Model | Runtime group 32 (derived token/s) | Runtime group 128 (derived token/s) | ggml group 32 (derived token/s) |
| --- | ---: | ---: | ---: |
| GPT-J 6B | 43.50 | 50.05 | 31.63 |
| Llama-2 7B | 42.74 | 45.54 | 36.09 |
| LLaMA 7B | 41.88 | 45.37 | 36.76 |
| GPT-NeoX 20B | 12.48 | 16.34 | 10.83 |
| Falcon 7B | 32.02 | 44.92 | 27.61 |

Comparison status: token-shape and CPU-generation-latency baseline only. Our Llama 3 models,
GGUF formats, llama.cpp implementation, and CPU SKU differ from the paper.

## 128 / 32 and the 128-1024 input sweep - Na et al.

Source: [Understanding Performance Implications of LLM Inference on CPUs, IISWC 2024](https://seonjinna.github.io/assets/pdf/iiswc24_CPULLM.pdf), especially Figures 8-10 and 17-21.

### Experimental conditions

| Dimension | Paper configuration |
| --- | --- |
| CPU | Xeon Max 9468, 48 cores on one socket for the main comparison |
| CPU memory placement | Quadrant + Flat, with HBM prioritized and DDR used after the socket's 64 GB HBM |
| CPU software / precision | Intel Extension for PyTorch 2.3, BF16 |
| Models | OPT 1.3B/6.7B/13B/30B/66B and LLaMA-2 7B/13B/70B |
| Default workload | Input 128, output 32, static batch sizes 1, 2, 4, 8, 16, and 32 |
| Length sensitivity | Input 128, 256, 512, and 1024; output 32; batch 1 and 16 |
| GPU comparison | FlexGen on one A100 40 GB or one H100 80 GB, including CPU offload when required |
| Metrics | End-to-end latency, TTFT, TPOT, and generated-output-token throughput |

### Exact reported results for `fixed_128_32`

Across all tested models, as batch size grows from 1 to 32, Xeon Max 9468 versus Xeon 8352Y is
reported as:

| Metric | Xeon Max result relative to Ice Lake |
| --- | ---: |
| End-to-end latency | 68.4%-84.1% lower |
| End-to-end output throughput | 3.2x-6.3x higher |
| TTFT | 84.1%-89.0% lower |
| Prefill throughput | 6.3x-9.1x higher |
| TPOT | 62.3%-81.7% lower |
| Decode throughput | 2.7x-5.5x higher |

Selected CPU/GPU points closest to our 1B and 8B model scales are below. Values are normalized to
the Xeon Max 9468 result for the same model and batch (`1.0`). The batch-1 throughput bars are read
approximately from the figure because the paper does not label them numerically.

| Model / batch | A100 latency | H100 latency | A100 throughput | H100 throughput |
| --- | ---: | ---: | ---: | ---: |
| OPT 1.3B / batch 1 | 0.7 | 0.7 | approx. 1.5x | approx. 1.4x |
| LLaMA-2 7B / batch 1 | 0.7 | 0.7 | approx. 1.4x | approx. 1.5x |
| OPT 1.3B / batch 16 | 0.2 | 0.2 | 4.7x | 4.9x |
| LLaMA-2 7B / batch 16 | 0.2 | 0.2 | 4.4x | 4.8x |

### Results for `fixed_256_32`, `fixed_512_32`, and `fixed_1024_32`

Na et al. publish these results only as curves in Figures 20 and 21, not as a numeric table or
artifact. Their defensible published conclusion is therefore trend-level:

- GPU latency and throughput remain comparatively stable as prompt length rises from 128 to 1024.
- Xeon Max throughput declines and latency rises more strongly with prompt length.
- At batch 1, Xeon Max remains better than the offloaded GPUs for LLaMA-2 70B at every tested
  length because PCIe movement dominates the GPU paths.
- At batch 16 and prompt lengths of at least 256, the H100 path becomes faster than Xeon Max for
  LLaMA-2 70B, while Xeon Max remains faster than the A100 offload path.

Comparison status: use the exact relative ranges or reproduce the paper's trend. Do not claim an
exact numerical reproduction from the plots. Our current offline workload also does not encode Na
et al.'s static batch-size sweep, so a direct ratio requires an explicitly matched batch experiment.

## FlexGen fixed-length throughput tables

Source: [FlexGen, ICML 2023](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf), Appendix Tables 14-18.

The metric is generated output tokens divided by prefill plus generation time. The numbers are the
maximum throughputs found using a source-selected policy for each engine and model, so batch and
placement policies are not held constant between table cells. The main platform is one NVIDIA T4
16 GB GPU, an Intel Xeon at 2.00 GHz with 208 GB CPU DRAM, and a 1.5 TB NVMe SSD. Models are OPT
in FP16. `FlexGen (c)` compresses both weights and KV cache to asymmetric group-wise 4-bit values
with group size 64; it is not GGUF `Q4_K_M`.

All values below are exact generation throughput in token/s. `OOM` is preserved from the paper.
Petals results are not copied because they use 1, 4, or 24 distributed GPUs depending on model size
and report per-GPU throughput under assumed network conditions, making them unsuitable cluster
baselines for this project.

| Workload | OPT size | Accelerate | DeepSpeed | FlexGen | FlexGen (c), 4-bit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_256_32` | 6.7B | 50.66 | 14.52 | 53.29 | 56.72 |
| `fixed_256_32` | 30B | 1.34 | 1.30 | 16.01 | 16.86 |
| `fixed_256_32` | 175B | 0.02 | 0.01 | 1.36 | 2.26 |
| `fixed_512_32` | 6.7B | 25.12 | 9.28 | 25.26 | 29.12 |
| `fixed_512_32` | 30B | 0.62 | 0.60 | 7.32 | 8.70 |
| `fixed_512_32` | 175B | 0.01 | 0.01 | 0.69 | 1.12 |
| `fixed_1024_32` | 6.7B | 13.01 | 4.59 | 13.72 | 13.18 |
| `fixed_1024_32` | 30B | 0.31 | 0.29 | 3.50 | 3.98 |
| `fixed_1024_32` | 175B | 0.01 | OOM | 0.35 | 0.42 |
| `fixed_128_128` | 6.7B | 73.411 | 19.193 | 106.404 | 92.568 |
| `fixed_128_128` | 30B | 1.547 | 1.717 | 24.634 | 39.141 |
| `fixed_128_128` | 175B | 0.021 | 0.024 | 2.409 | 4.264 |
| `fixed_512_8` | 6.7B | 17.290 | 9.055 | 16.425 | 14.244 |
| `fixed_512_8` | 30B | 0.628 | 0.872 | 3.938 | 4.019 |
| `fixed_512_8` | 175B | 0.009 | 0.007 | 0.451 | 0.559 |

For the closest paper scale, OPT 6.7B, these maxima use the following source-selected GPU batch
and block policies. Every listed 6.7B policy keeps weights, activations, and KV cache entirely on
the GPU. A FlexGen block is `GPU batch size x number of GPU batches`.

| Workload | Accelerate | DeepSpeed | FlexGen | FlexGen (c) |
| --- | ---: | ---: | ---: | ---: |
| `fixed_256_32` | 4x1 | 32x1 | 4x1 | 128x1 |
| `fixed_512_32` | 2x1 | 16x1 | 2x1 | 72x1 |
| `fixed_1024_32` | 1x1 | 8x1 | 1x1 | 28x1 |
| `fixed_128_128` | 5x1 | 36x1 | 7x1 | 196x1 |
| `fixed_512_8` | 2x1 | 18x1 | 2x1 | 76x1 |

The closest scale for our Llama 3.1 8B model is OPT 6.7B, but architecture, runtime, precision,
hardware, effective batch, and measurement implementation all differ. These are therefore
shape-matched contextual baselines, not expected values for the cluster.

## 512 input / 128 output - Kurt llama.cpp quantization study

Source: [Which Quantization Should I Use?, arXiv:2601.14277v1, Table 3](https://arxiv.org/html/2601.14277).

This is the closest baseline in the corpus because it uses Llama-3.1-8B-Instruct, GGUF, and
llama.cpp. The machine has two Xeon Platinum 8488C processors, 96 physical cores in total, with
AVX-512/BF16 enabled. The llama.cpp revision is `b7600`. `pp512` and `tg128` are separate
llama-bench phase measurements, not one end-to-end 512+128 request.

| GGUF scheme | pp512 (token/s) | tg128 (token/s) |
| --- | ---: | ---: |
| F16 | 79.57 +/- 1.00 | 2.83 +/- 0.01 |
| Q3_K_S | 57.39 +/- 3.01 | 9.91 +/- 0.03 |
| Q3_K_M | 68.86 +/- 1.36 | 6.52 +/- 0.23 |
| Q3_K_L | 58.34 +/- 1.93 | 7.93 +/- 0.06 |
| Q4_0 | 97.35 +/- 0.68 | 4.36 +/- 0.20 |
| Q4_1 | 55.20 +/- 1.61 | 6.80 +/- 0.15 |
| Q4_K_S | 92.52 +/- 1.53 | 4.65 +/- 0.15 |
| Q4_K_M | 87.70 +/- 0.70 | 5.12 +/- 0.37 |
| Q5_0 | 61.44 +/- 2.47 | 6.66 +/- 0.06 |
| Q5_1 | 45.98 +/- 2.17 | 6.33 +/- 0.26 |
| Q5_K_S | 55.31 +/- 1.67 | 5.98 +/- 0.02 |
| Q5_K_M | 58.24 +/- 2.05 | 6.85 +/- 0.13 |
| Q6_K | 59.81 +/- 1.07 | 6.33 +/- 0.13 |
| Q8_0 | 71.42 +/- 3.15 | 5.03 +/- 0.16 |

Our manifest directly supports F16, Q4_K_M, and Q8_0. However, the current offline adapter asks
llama-bench for a combined prompt-generation row (`-pg 512,128`) and normalizes only combined
throughput. A direct Kurt comparison requires separate prefill and generation measurements to be
retained. Until that runner gap is addressed, the token shape and model match but the metric does
not.

## 512 / 128 and 1024 / 128 - THInfer

Source: [Bandwidth-Aware LLM Inference on Heterogeneous Many-Core Supercomputers, arXiv:2605.25655v1, Tables V-VII](https://arxiv.org/html/2605.25655).

THInfer uses FP16 storage with FP32 accumulation and LLaMA-2 models. Prompts are synthetic and
padded to 512 or 1024 tokens; every prompt generates 128 tokens. The metric is end-to-end
generation throughput in token/s.

The peak-aligned comparison uses 8 MT-3000 devices against 2 V100S GPUs, or 10 MT-3000 devices
against 1 A800. The bandwidth-aligned configurations use 18 and 16 MT-3000 devices respectively.
Consequently, the two THInfer columns are different hardware allocations, not repeated results.

| Workload | Model | V100S Accelerate (2 GPUs) | V100S DeepSpeed (2 GPUs) | THInfer peak-aligned (8 devices) | THInfer BW-aligned (18 devices) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fixed_512_128` | 7B | 323 | 481 | 781 | 1755 |
| `fixed_512_128` | 13B | 168 | 273 | 241 | 543 |
| `fixed_512_128` | 30B | 1.9 | 7.69 | 81 | 186 |
| `fixed_512_128` | 70B | OOM | OOM | 64 | 129 |
| `fixed_1024_128` | 7B | 173 | 263 | 456 | 1026 |
| `fixed_1024_128` | 13B | 93 | 145 | 169 | 381 |
| `fixed_1024_128` | 30B | 0.82 | 4.07 | 46 | 105 |
| `fixed_1024_128` | 70B | OOM | OOM | 41 | 81 |

| Workload | Model | A800 DeepSpeed (1 GPU) | THInfer peak-aligned (10 devices) | THInfer BW-aligned (16 devices) |
| --- | ---: | ---: | ---: | ---: |
| `fixed_512_128` | 7B | 584 | 975 | 1560 |
| `fixed_512_128` | 13B | 345 | 301 | 483 |
| `fixed_512_128` | 30B | 116 | 105 | 162 |
| `fixed_512_128` | 70B | OOM | 64 | 129 |
| `fixed_1024_128` | 7B | 310 | 570 | 912 |
| `fixed_1024_128` | 13B | 184 | 211 | 339 |
| `fixed_1024_128` | 30B | 60 | 61 | 89 |
| `fixed_1024_128` | 70B | OOM | 41 | 81 |

For the 512/128 ablation on 12 MT-3000 clusters, the paper reports final A4 throughput of 321,
101, and 34 token/s for 7B, 13B, and 30B respectively. Its unoptimized A0 values are 1.02, 0.21,
and 0.07 token/s. These numbers measure the whole THInfer co-design, not memory placement alone.

Comparison status: exact token shape and metric, but different model generation, processor,
runtime, and device count. Use as HPC-system context rather than a direct performance target.

## Approximately 30K input / 10K output - Fang et al.

Source: [Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System, arXiv:2508.13231v2](https://arxiv.org/html/2508.13231).

This source uses LLaMA-3.1-8B with approximately 30K-token NarrativeQA prompts from LongBench and
records attention traces while decoding 10K tokens. It then evaluates those traces in a behavioral
memory-bandwidth simulator. The simulated system has 24 GB HBM at 4.9 TB/s and 480 GB off-package
DRAM at 500 GB/s over a 900 GB/s link. Approximately 16 GB of model weights remain in HBM, leaving
about 8 GB HBM for KV cache.

| Reported result | Value |
| --- | ---: |
| SA-guided placement versus static placement across studied scenarios | approximately 4x-5x |
| Maximum simulated upper-bound gap versus a simple baseline | up to 5.87x |
| Static placement normalization | 1.0x |

The result is an upper bound with future knowledge of token importance, not a deployable scheduler
and not a hardware measurement. Our cluster comparison should therefore be a within-system ratio:
the throughput of an explicitly placed HBM/tiered treatment divided by its static or DDR baseline.
Absolute token/s must not be compared to Fang et al.

## How to fill in our cluster results

For each completed run, add one row to the campaign table below or generate an equivalent table
from `summary.json`. Keep the literature value and our value in the same row only if their metric
definitions match.

| Workload | Our model / variant | Profile and hardware | Our metric | Literature target | Status / interpretation | Run ID |
| --- | --- | --- | ---: | ---: | --- | --- |
| `fixed_32_32` | pending | pending | pending | Shen: 19.98-92.36 ms/token, depending on model/runtime | Match TPOT only; not aggregate throughput | pending |
| `fixed_128_32` | pending | pending | pending | Na: relative ranges above | Requires matched static batch for a ratio | pending |
| `fixed_256_32` | pending | pending | pending | FlexGen table above | Shape-only unless engine policy is matched | pending |
| `fixed_512_32` | pending | pending | pending | FlexGen table above | Shape-only unless engine policy is matched | pending |
| `fixed_1024_32` | pending | pending | pending | FlexGen table above | Shape-only unless engine policy is matched | pending |
| `fixed_128_128` | pending | pending | pending | FlexGen table above | Shape-only unless engine policy is matched | pending |
| `fixed_512_8` | pending | pending | pending | FlexGen table above | Shape-only unless engine policy is matched | pending |
| `fixed_512_128` | pending | pending | pending | Kurt and THInfer tables above | Kurt requires separate pp/tg metrics | pending |
| `fixed_1024_128` | pending | pending | pending | THInfer tables above | Shape and metric match; systems differ | pending |
| `fixed_30000_10000` | pending | pending | pending | Fang: relative simulated upper bound | Compare only within-cluster placement ratios | pending |

## Audit notes

- Exact values were copied only from printed tables or explicitly stated prose.
- Approximate values are clearly labeled and are not suitable for precise speedup calculations.
- FlexGen's generated-token throughput includes prefill time. Kurt's `pp512` and `tg128` are
  separate phase rates. They are not interchangeable.
- Na et al. do not publish raw data, repetition counts, dispersion, full software versions, or
  exact absolute values for every plotted point. Their work supports trend reproduction more
  strongly than exact numerical reproduction.
- Fang et al. report simulation-normalized results. Their 5.87x result is not an empirical GH200
  or Xeon Max speedup.
- Source versions checked on 2026-09-13: Shen v2, FlexGen ICML/PMLR version, Na IISWC version,
  Kurt v1, THInfer v1, and Fang v2.
