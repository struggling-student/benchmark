"""Deterministic, non-ranking interpretation of benchmark comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping
from statistics import stdev
from typing import Any

from .metrics import METRICS_BY_FIELD

_PRIMARY_METRICS = {
    "smoke": (
        "model_load_time_seconds",
        "output_throughput_tokens_per_second",
        "peak_gpu_memory_mib",
        "energy_joules",
    ),
    "offline": (
        "request_throughput_requests_per_second",
        "output_throughput_tokens_per_second",
        "total_throughput_tokens_per_second",
        "peak_gpu_memory_mib",
        "energy_per_output_token_joules",
    ),
    "serving": (
        "request_throughput_requests_per_second",
        "output_throughput_tokens_per_second",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
        "energy_per_request_joules",
    ),
}


def repetition_variability(measurements: Mapping[str, Any], field: str) -> dict[str, float] | None:
    """Return descriptive sample variability for mean-aggregated observations."""

    values = [
        record.get("metrics", {}).get(field)
        for record in measurements.get("records", [])
        if record.get("status") == "completed"
        and isinstance(record.get("metrics", {}).get(field), (int, float))
    ]
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    return {
        "count": float(len(finite)),
        "mean": mean,
        "standard_deviation": stdev(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def comparison_narrative(
    comparison: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_measurements: Mapping[str, Any] | None = None,
    right_measurements: Mapping[str, Any] | None = None,
) -> list[str]:
    """Describe observable differences while preserving compatibility constraints."""

    compatibility = comparison.get("compatibility", {})
    status = compatibility.get("status")
    statements = [f"Run status: left is {left.get('status')!r}; right is {right.get('status')!r}."]
    if status == "compatible":
        statements.append(
            "The recorded workload, software, and hardware evidence is compatible; "
            "directional ratios may be interpreted for this explicit pair."
        )
    elif status == "partial":
        statements.append(
            "Compatibility is only partial because required evidence is missing. Ratios and "
            "headline comparative claims are suppressed."
        )
    else:
        statements.append(
            "The runs are incompatible for a fair head-to-head comparison. Absolute values "
            "remain observations of their respective conditions."
        )

    benchmark_type = str(left.get("benchmark_type") or "")
    if status == "compatible":
        metrics = comparison.get("metrics", {})
        for field in _PRIMARY_METRICS.get(benchmark_type, ()):
            values = metrics.get(field, {})
            left_value = values.get("left")
            right_value = values.get("right")
            ratio = values.get("right_to_left_ratio")
            if not all(isinstance(item, (int, float)) for item in (left_value, right_value, ratio)):
                continue
            difference = (float(ratio) - 1.0) * 100.0
            relation = "higher" if difference >= 0 else "lower"
            spec = METRICS_BY_FIELD[field]
            preference = {
                "higher": " Higher throughput is generally preferable under controlled conditions.",
                "lower": (
                    " Lower resource use or latency is generally preferable under controlled "
                    "conditions."
                ),
                "descriptive": (
                    " This metric is descriptive and has no universal preferred direction."
                ),
            }[spec.direction]
            statements.append(
                f"{spec.label}: right is {abs(difference):.1f}% {relation} than left "
                f"({float(right_value):.4g} vs {float(left_value):.4g} {spec.unit}).{preference}"
            )

    if benchmark_type == "smoke":
        statements.append(
            "Smoke runs primarily establish that loading and generation work; their performance "
            "values are not a substitute for the measured offline or serving campaigns."
        )

    if left_measurements is not None and right_measurements is not None:
        variable_metrics = 0
        for field, spec in METRICS_BY_FIELD.items():
            if spec.aggregation != "mean":
                continue
            if repetition_variability(left_measurements, field) or repetition_variability(
                right_measurements, field
            ):
                variable_metrics += 1
        if variable_metrics:
            statements.append(
                f"Repetition spread is available for {variable_metrics} mean-aggregated metric(s). "
                "Error bars are descriptive sample standard deviations, not confidence intervals."
            )
    statements.append(
        "No statistical significance or model-answer quality is inferred, and the metrics are "
        "not combined into an overall score."
    )
    return statements
