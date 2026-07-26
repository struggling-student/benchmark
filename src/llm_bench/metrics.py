"""Presentation metadata for normalized benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Aggregation = Literal["mean", "maximum", "sum", "weighted"]
Direction = Literal["higher", "lower", "descriptive"]
Scope = Literal["shared", "gpu", "cpu", "memory"]


@dataclass(frozen=True)
class MetricSpec:
    """Human-facing semantics for one normalized metric."""

    field: str
    label: str
    unit: str
    category: str
    aggregation: Aggregation
    direction: Direction
    scope: Scope


def _metric(
    field: str,
    label: str,
    unit: str,
    category: str,
    aggregation: Aggregation = "mean",
    direction: Direction = "descriptive",
    scope: Scope = "shared",
) -> MetricSpec:
    return MetricSpec(field, label, unit, category, aggregation, direction, scope)


METRIC_SPECS = (
    _metric("duration_seconds", "Run duration", "s", "Execution"),
    _metric(
        "model_load_time_seconds",
        "Model load time",
        "s",
        "Execution",
        direction="lower",
    ),
    _metric(
        "request_throughput_requests_per_second",
        "Request throughput",
        "requests/s",
        "Throughput",
        direction="higher",
    ),
    _metric(
        "input_throughput_tokens_per_second",
        "Input-token throughput",
        "tokens/s",
        "Throughput",
        direction="higher",
    ),
    _metric(
        "output_throughput_tokens_per_second",
        "Output-token throughput",
        "tokens/s",
        "Throughput",
        direction="higher",
    ),
    _metric(
        "total_throughput_tokens_per_second",
        "Total-token throughput",
        "tokens/s",
        "Throughput",
        direction="higher",
    ),
    _metric(
        "prefill_throughput_tokens_per_second",
        "Prefill throughput",
        "tokens/s",
        "Throughput",
        direction="higher",
    ),
    _metric(
        "decode_throughput_tokens_per_second",
        "Decode throughput",
        "tokens/s",
        "Throughput",
        direction="higher",
    ),
    *(
        _metric(field, label, "ms", "Latency", direction="lower")
        for field, label in (
            ("mean_ttft_ms", "Mean TTFT"),
            ("median_ttft_ms", "Median TTFT"),
            ("p95_ttft_ms", "p95 TTFT"),
            ("p99_ttft_ms", "p99 TTFT"),
            ("mean_tpot_ms", "Mean TPOT"),
            ("median_tpot_ms", "Median TPOT"),
            ("p95_tpot_ms", "p95 TPOT"),
            ("p99_tpot_ms", "p99 TPOT"),
            ("mean_itl_ms", "Mean ITL"),
            ("median_itl_ms", "Median ITL"),
            ("p95_itl_ms", "p95 ITL"),
            ("p99_itl_ms", "p99 ITL"),
            ("mean_e2e_latency_ms", "Mean end-to-end latency"),
            ("p95_e2e_latency_ms", "p95 end-to-end latency"),
        )
    ),
    _metric(
        "peak_gpu_memory_mib",
        "Peak GPU memory",
        "MiB",
        "Resources",
        aggregation="maximum",
        direction="lower",
        scope="gpu",
    ),
    _metric(
        "average_gpu_utilization_percent",
        "Average GPU utilization",
        "%",
        "Resources",
        scope="gpu",
    ),
    _metric(
        "peak_cpu_memory_mib",
        "Peak CPU memory",
        "MiB",
        "Resources",
        aggregation="maximum",
        direction="lower",
        scope="cpu",
    ),
    _metric(
        "average_cpu_utilization_percent",
        "Average CPU utilization",
        "%",
        "Resources",
        scope="cpu",
    ),
    _metric(
        "measured_memory_bandwidth_gbps",
        "Measured memory bandwidth",
        "GB/s",
        "Memory",
        direction="higher",
        scope="memory",
    ),
    _metric(
        "memory_bandwidth_utilization_percent",
        "Memory-bandwidth utilization",
        "%",
        "Memory",
        scope="memory",
    ),
    _metric(
        "average_gpu_power_watts",
        "Average GPU power",
        "W",
        "Energy",
        scope="gpu",
    ),
    _metric(
        "average_cpu_power_watts",
        "Average CPU package power",
        "W",
        "Energy",
        scope="cpu",
    ),
    _metric(
        "energy_joules",
        "Device energy",
        "J",
        "Energy",
        aggregation="sum",
        direction="lower",
    ),
    _metric(
        "energy_per_request_joules",
        "Energy per request",
        "J/request",
        "Energy",
        aggregation="weighted",
        direction="lower",
    ),
    _metric(
        "energy_per_output_token_joules",
        "Energy per output token",
        "J/token",
        "Energy",
        aggregation="weighted",
        direction="lower",
    ),
)

METRICS_BY_FIELD = {spec.field: spec for spec in METRIC_SPECS}


def metric_spec(field: str) -> MetricSpec:
    """Return presentation metadata for a known normalized metric."""

    return METRICS_BY_FIELD[field]
