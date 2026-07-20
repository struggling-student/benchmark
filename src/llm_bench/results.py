"""Stable result normalization, serialization, and two-run comparison helpers."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import ExperimentConfig
from .metadata import utc_timestamp

SCHEMA_VERSION = "1.1"

# Every summary written by this package contains every field. New fields may be added in
# later schema versions, while raw backend output remains the source of truth.
SUMMARY_FIELDS = (
    "schema_version",
    "run_id",
    "timestamp",
    "status",
    "error",
    "experiment_name",
    "benchmark_type",
    "backend",
    "backend_version",
    "model_id",
    "model_revision",
    "resolved_model_revision",
    "tokenizer_id",
    "resolved_tokenizer_revision",
    "model_parameter_scale",
    "model_precision",
    "git_commit",
    "git_dirty",
    "slurm_job_id",
    "hostname",
    "hardware_type",
    "accelerator_name",
    "accelerator_count",
    "accelerator_visibility",
    "cpu_model",
    "socket_count",
    "numa_node_count",
    "software_versions",
    "dtype",
    "quantization",
    "tensor_parallel_size",
    "seed",
    "input_length",
    "output_length",
    "max_model_len",
    "generation_config",
    "temperature",
    "top_p",
    "ignore_eos",
    "number_of_requests",
    "successful_requests",
    "failed_requests",
    "actual_input_tokens",
    "actual_output_tokens",
    "request_rate",
    "maximum_concurrency",
    "gpu_memory_utilization",
    "warmup_runs",
    "repetitions",
    "telemetry_interval_ms",
    "measured_repetitions",
    "failed_repetitions",
    "duration_seconds",
    "model_load_time_seconds",
    "request_throughput_requests_per_second",
    "input_throughput_tokens_per_second",
    "output_throughput_tokens_per_second",
    "total_throughput_tokens_per_second",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "peak_gpu_memory_mib",
    "average_gpu_utilization_percent",
    "average_gpu_power_watts",
    "energy_joules",
    "energy_per_request_joules",
    "energy_per_output_token_joules",
    "raw_output_files",
    "telemetry_files",
    "warnings",
)

NUMERIC_METRICS = (
    "duration_seconds",
    "model_load_time_seconds",
    "request_throughput_requests_per_second",
    "input_throughput_tokens_per_second",
    "output_throughput_tokens_per_second",
    "total_throughput_tokens_per_second",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "peak_gpu_memory_mib",
    "average_gpu_utilization_percent",
    "average_gpu_power_watts",
    "energy_joules",
    "energy_per_request_joules",
    "energy_per_output_token_joules",
)

COMPARISON_METRICS = NUMERIC_METRICS

COMPATIBILITY_FIELDS = (
    "backend",
    "backend_version",
    "benchmark_type",
    "input_length",
    "output_length",
    "max_model_len",
    "generation_config",
    "temperature",
    "top_p",
    "ignore_eos",
    "number_of_requests",
    "request_rate",
    "maximum_concurrency",
    "gpu_memory_utilization",
    "seed",
    "tensor_parallel_size",
    "dtype",
    "quantization",
    "warmup_runs",
    "repetitions",
    "measured_repetitions",
    "telemetry_interval_ms",
    "git_commit",
    "git_dirty",
    "hardware_type",
    "accelerator_name",
    "accelerator_count",
    "cpu_model",
    "socket_count",
    "numa_node_count",
    "software_versions.python",
    "software_versions.llm_bench",
    "software_versions.vllm",
    "software_versions.torch",
    "software_versions.cuda_runtime",
    "software_versions.nvidia_driver",
)


class ResultError(ValueError):
    """Raised for an unreadable or structurally invalid result."""


_ALIASES: dict[str, tuple[str, ...]] = {
    "successful_requests": (
        "successful_requests",
        "completed",
        "num_successful_requests",
        "num_requests",
    ),
    "failed_requests": ("failed_requests", "failed"),
    "actual_input_tokens": (
        "actual_input_tokens",
        "total_input_tokens",
        "num_input_tokens",
    ),
    "actual_output_tokens": (
        "actual_output_tokens",
        "total_output_tokens",
        "total_generated_tokens",
        "num_output_tokens",
    ),
    "duration_seconds": (
        "duration_seconds",
        "duration",
        "elapsed_time",
        "benchmark_duration_seconds",
    ),
    "model_load_time_seconds": (
        "model_load_time_seconds",
        "model_load_time",
        "model_loading_time_seconds",
    ),
    "request_throughput_requests_per_second": (
        "request_throughput_requests_per_second",
        "request_throughput",
        "requests_per_second",
    ),
    "input_throughput_tokens_per_second": (
        "input_throughput_tokens_per_second",
        "input_throughput",
        "input_token_throughput",
    ),
    "output_throughput_tokens_per_second": (
        "output_throughput_tokens_per_second",
        "output_throughput",
        "output_token_throughput",
    ),
    "total_throughput_tokens_per_second": (
        "total_throughput_tokens_per_second",
        "total_throughput",
        "total_token_throughput",
        "tokens_per_second",
        "token_throughput",
    ),
    "mean_ttft_ms": ("mean_ttft_ms", "avg_ttft_ms"),
    "median_ttft_ms": ("median_ttft_ms", "med_ttft_ms"),
    "p95_ttft_ms": ("p95_ttft_ms",),
    "p99_ttft_ms": ("p99_ttft_ms",),
    "mean_tpot_ms": ("mean_tpot_ms", "avg_tpot_ms"),
    "median_tpot_ms": ("median_tpot_ms", "med_tpot_ms"),
    "p95_tpot_ms": ("p95_tpot_ms",),
    "p99_tpot_ms": ("p99_tpot_ms",),
    "mean_itl_ms": ("mean_itl_ms", "avg_itl_ms"),
    "median_itl_ms": ("median_itl_ms", "med_itl_ms"),
    "p95_itl_ms": ("p95_itl_ms",),
    "p99_itl_ms": ("p99_itl_ms",),
    "mean_e2e_latency_ms": (
        "mean_e2e_latency_ms",
        "mean_e2el_ms",
        "avg_e2e_latency_ms",
    ),
    "p95_e2e_latency_ms": ("p95_e2e_latency_ms", "p95_e2el_ms"),
    "peak_gpu_memory_mib": ("peak_gpu_memory_mib", "max_gpu_memory_mib"),
    "average_gpu_utilization_percent": (
        "average_gpu_utilization_percent",
        "avg_gpu_utilization_percent",
    ),
    "average_gpu_power_watts": (
        "average_gpu_power_watts",
        "avg_gpu_power_watts",
    ),
    "energy_joules": ("energy_joules",),
    "energy_per_request_joules": ("energy_per_request_joules",),
    "energy_per_output_token_joules": ("energy_per_output_token_joules",),
}


def _empty_summary() -> dict[str, Any]:
    summary = {field: None for field in SUMMARY_FIELDS}
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "software_versions": {},
            "raw_output_files": [],
            "telemetry_files": [],
            "warnings": [],
        }
    )
    return summary


def create_summary(
    config: ExperimentConfig,
    metadata: Mapping[str, Any],
    *,
    status: str = "running",
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    """Create a complete hardware-independent summary with unavailable values null."""

    summary = _empty_summary()
    for field in (
        "run_id",
        "timestamp",
        "experiment_name",
        "benchmark_type",
        "backend",
        "backend_version",
        "model_id",
        "model_revision",
        "resolved_model_revision",
        "tokenizer_id",
        "resolved_tokenizer_revision",
        "model_parameter_scale",
        "git_commit",
        "git_dirty",
        "slurm_job_id",
        "hostname",
        "hardware_type",
        "accelerator_name",
        "accelerator_count",
        "accelerator_visibility",
        "cpu_model",
        "socket_count",
        "numa_node_count",
        "software_versions",
    ):
        if field in metadata:
            summary[field] = metadata[field]
    summary.update(
        {
            "status": status,
            "experiment_name": config.experiment_name,
            "benchmark_type": metadata.get("benchmark_type", config.resolved_benchmark_type),
            "backend": config.backend,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_id": config.tokenizer,
            "model_parameter_scale": config.model_parameter_scale,
            # An explicit vLLM dtype controls model weights and activations. ``auto``
            # is only a policy and remains unknown unless raw backend output resolves it.
            "model_precision": None if config.dtype == "auto" else config.dtype,
            "dtype": config.dtype,
            "quantization": config.quantization,
            "tensor_parallel_size": config.tensor_parallel_size,
            "seed": config.seed,
            "input_length": config.input_length,
            "output_length": config.output_length,
            "max_model_len": config.max_model_len,
            "generation_config": config.generation_config,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "ignore_eos": config.ignore_eos,
            "number_of_requests": config.number_of_prompts,
            "request_rate": config.request_rate,
            "maximum_concurrency": config.maximum_concurrency,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "warmup_runs": config.warmup_runs,
            "repetitions": config.repetitions,
            "telemetry_interval_ms": config.telemetry_interval_ms,
            "measured_repetitions": 0,
            "failed_repetitions": 0,
            "warnings": list(
                dict.fromkeys(str(item) for item in [*metadata.get("warnings", []), *warnings])
            ),
        }
    )
    if status == "running":
        summary["warnings"].append("Metrics are null until the run is finalized.")
    return summary


def write_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    """Write a summary with deterministic formatting and strict finite numbers."""

    missing = [field for field in SUMMARY_FIELDS if field not in summary]
    if missing:
        raise ResultError("summary is missing field(s): " + ", ".join(missing))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(dict(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ResultError(f"summary is not valid JSON: {exc}") from exc
    destination.write_text(encoded, encoding="utf-8")


def resolve_summary_path(path: str | Path) -> Path:
    """Resolve an explicit summary file or explicit result directory."""

    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "summary.json"
    if not candidate.is_file():
        raise ResultError(f"summary.json not found: {candidate}")
    return candidate


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load a summary from exactly the supplied file or result directory."""

    source = resolve_summary_path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"cannot read summary {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ResultError(f"summary must contain a JSON object: {source}")
    required = ("run_id", "status", "model_id", "warnings")
    missing = [field for field in required if field not in data]
    if missing:
        raise ResultError(f"summary {source} is missing field(s): " + ", ".join(missing))
    return data


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _flatten_values(value: Any) -> dict[str, list[Any]]:
    flattened: dict[str, list[Any]] = defaultdict(list)

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                flattened[_normalized_key(str(key))].append(child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return flattened


_NUMBER = re.compile(r"^[\s]*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        match = _NUMBER.match(value)
        if match is None:
            return None
        try:
            result = float(match.group(1))
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _metrics_from_raw(data: Any) -> dict[str, Any]:
    values = _flatten_values(data)
    metrics: dict[str, Any] = {}
    for field, aliases in _ALIASES.items():
        for alias in aliases:
            candidates = values.get(_normalized_key(alias), ())
            metric = next(
                (parsed for item in candidates if (parsed := _finite_float(item)) is not None),
                None,
            )
            if metric is not None:
                metrics[field] = metric
                break
    for alias in ("model_precision", "resolved_dtype", "model_dtype"):
        candidates = values.get(_normalized_key(alias), ())
        precision = next(
            (item.strip() for item in candidates if isinstance(item, str) and item.strip()),
            None,
        )
        if precision is not None:
            metrics["model_precision"] = precision
            break
    return metrics


def read_raw_metrics(path: str | Path) -> dict[str, Any]:
    """Extract known metrics from one preserved vLLM JSON output file."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"cannot parse raw vLLM JSON {source}: {exc}") from exc
    if not isinstance(data, (dict, list)):
        raise ResultError(f"raw vLLM output must be a JSON object or array: {source}")
    return _metrics_from_raw(data)


def _parse_timestamp(value: str) -> float | None:
    numeric = _finite_float(value)
    if numeric is not None and numeric > 1_000_000:
        return numeric
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        parsed = None
    if parsed is None:
        for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(cleaned, pattern).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _row_value(row: Mapping[str, str], aliases: Sequence[str]) -> str | None:
    normalized = {_normalized_key(key): value for key, value in row.items() if key}
    for alias in aliases:
        key = _normalized_key(alias)
        if key in normalized:
            return normalized[key]
    return None


def read_telemetry_metrics(
    path: str | Path, *, duration_seconds: float | None = None
) -> tuple[dict[str, float], list[str]]:
    """Reduce an optional nvidia-smi CSV without requiring every queried metric."""

    source = Path(path)
    warnings: list[str] = []
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise ResultError(f"cannot read GPU telemetry {source}: {exc}") from exc
    if not rows:
        return {}, [f"GPU telemetry file is empty: {source}"]

    # Group simultaneous per-GPU rows so power and memory represent the allocation,
    # not an arbitrary single device.
    samples: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    fallback_index = 0.0
    parsed_timestamps = True
    for row in rows:
        timestamp_text = _row_value(row, ("timestamp", "timestamp_iso", "time"))
        timestamp = _parse_timestamp(timestamp_text) if timestamp_text else None
        if timestamp is None:
            parsed_timestamps = False
            timestamp = fallback_index
            fallback_index += 1.0
        values = {
            "utilization": _row_value(
                row, ("utilization.gpu [%]", "utilization_gpu", "gpu_utilization_percent")
            ),
            "memory": _row_value(row, ("memory.used [MiB]", "memory_used_mib", "memory_used")),
            "power": _row_value(row, ("power.draw [W]", "power_draw_watts", "power_draw")),
        }
        for name, text in values.items():
            parsed = _finite_float(text)
            if parsed is not None:
                samples[timestamp][name].append(parsed)

    utilization_by_time: list[float] = []
    memory_by_time: list[float] = []
    power_by_time: list[tuple[float, float]] = []
    for timestamp in sorted(samples):
        sample = samples[timestamp]
        if sample["utilization"]:
            utilization_by_time.append(fmean(sample["utilization"]))
        if sample["memory"]:
            memory_by_time.append(sum(sample["memory"]))
        if sample["power"]:
            power_by_time.append((timestamp, sum(sample["power"])))

    metrics: dict[str, float] = {}
    if utilization_by_time:
        metrics["average_gpu_utilization_percent"] = fmean(utilization_by_time)
    if memory_by_time:
        metrics["peak_gpu_memory_mib"] = max(memory_by_time)
    if power_by_time:
        metrics["average_gpu_power_watts"] = fmean(value for _, value in power_by_time)
        energy: float | None = None
        if parsed_timestamps and len(power_by_time) >= 2:
            energy = sum(
                (right_time - left_time) * (left_power + right_power) / 2
                for (left_time, left_power), (right_time, right_power) in zip(
                    power_by_time, power_by_time[1:], strict=False
                )
                if right_time >= left_time
            )
        elif duration_seconds is not None and duration_seconds >= 0:
            energy = metrics["average_gpu_power_watts"] * duration_seconds
            warnings.append(
                f"Energy for {source.name} uses average sampled GPU power times run duration "
                "because telemetry timestamps were insufficient."
            )
        if energy is not None:
            metrics["energy_joules"] = energy
    return metrics, warnings


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    aggregate: dict[str, float | int] = {}
    for field in NUMERIC_METRICS:
        values = [record[field] for record in records if field in record]
        if not values:
            continue
        if field == "peak_gpu_memory_mib":
            aggregate[field] = max(values)
        elif field == "energy_joules":
            # Energy is additive across explicitly measured repetitions.
            aggregate[field] = sum(values)
        else:
            aggregate[field] = fmean(values)
    for field in (
        "successful_requests",
        "failed_requests",
        "actual_input_tokens",
        "actual_output_tokens",
    ):
        values = [record[field] for record in records if field in record]
        if values:
            total = sum(values)
            aggregate[field] = int(total) if float(total).is_integer() else total
    return aggregate


def normalize_summary(
    summary: Mapping[str, Any],
    *,
    raw_output_paths: Sequence[str | Path] = (),
    telemetry_paths: Sequence[str | Path] = (),
    status: str = "completed",
    duration_seconds: float | None = None,
    model_load_time_seconds: float | None = None,
    warnings: Iterable[str] = (),
    error: str | None = None,
) -> dict[str, Any]:
    """Normalize all explicitly supplied measured repetitions into one summary.

    Arithmetic means are used across measured repetitions. Peak memory is the maximum.
    Warm-up records must not be supplied to this function.
    """

    result = _empty_summary()
    result.update({key: value for key, value in summary.items() if key in result})
    combined_warnings = [
        str(item)
        for item in result.get("warnings", [])
        if str(item) != "Metrics are null until the run is finalized."
    ]
    combined_warnings.extend(str(item) for item in warnings)
    raw_records: list[dict[str, Any]] = []
    failed_repetitions = 0
    for path in raw_output_paths:
        try:
            raw_records.append(read_raw_metrics(path))
        except ResultError as exc:
            failed_repetitions += 1
            combined_warnings.append(str(exc))

    expected = result.get("repetitions")
    supplied = len(raw_output_paths)
    if isinstance(expected, int) and expected != supplied:
        combined_warnings.append(
            f"Expected {expected} measured repetition(s), but received {supplied} raw output "
            "file(s); no run was silently selected."
        )
        if supplied < expected:
            failed_repetitions += expected - supplied

    telemetry_records: list[dict[str, float]] = []
    if telemetry_paths and len(telemetry_paths) != supplied:
        combined_warnings.append(
            f"Received {len(telemetry_paths)} telemetry file(s) for {supplied} raw output "
            "file(s); partial telemetry metrics will not be normalized."
        )
    raw_durations = [record.get("duration_seconds") for record in raw_records]
    for index, path in enumerate(telemetry_paths):
        paired_duration = (
            raw_durations[index]
            if index < len(raw_durations) and raw_durations[index] is not None
            else duration_seconds
        )
        try:
            metrics, telemetry_warnings = read_telemetry_metrics(
                path, duration_seconds=paired_duration
            )
            telemetry_records.append(metrics)
            combined_warnings.extend(telemetry_warnings)
        except ResultError as exc:
            combined_warnings.append(str(exc))

    telemetry_complete = len(telemetry_paths) == supplied and len(telemetry_records) == len(
        telemetry_paths
    )
    if telemetry_paths and not telemetry_complete:
        telemetry_records = []
    elif telemetry_records:
        for field in (
            "peak_gpu_memory_mib",
            "average_gpu_utilization_percent",
            "average_gpu_power_watts",
            "energy_joules",
        ):
            present = sum(field in record for record in telemetry_records)
            if 0 < present < len(telemetry_records):
                for record in telemetry_records:
                    record.pop(field, None)
                combined_warnings.append(
                    f"Telemetry metric {field} was missing from one or more measured "
                    "repetitions and was left null rather than partially aggregated."
                )

    aggregate = _aggregate([*raw_records, *telemetry_records])
    if duration_seconds is not None and "duration_seconds" not in aggregate:
        aggregate["duration_seconds"] = duration_seconds
    if model_load_time_seconds is not None:
        aggregate["model_load_time_seconds"] = model_load_time_seconds
    result.update(aggregate)

    observed_precisions = {
        str(record["model_precision"])
        for record in raw_records
        if record.get("model_precision") is not None
    }
    if len(observed_precisions) == 1:
        result["model_precision"] = observed_precisions.pop()
    elif len(observed_precisions) > 1:
        result["model_precision"] = None
        combined_warnings.append(
            "Measured repetitions reported different model precisions: "
            + ", ".join(sorted(observed_precisions))
            + "."
        )

    energy = result.get("energy_joules")
    requests = result.get("successful_requests")
    output_tokens = result.get("actual_output_tokens")
    if isinstance(energy, (int, float)) and isinstance(requests, (int, float)) and requests > 0:
        result["energy_per_request_joules"] = energy / requests
    if (
        isinstance(energy, (int, float))
        and isinstance(output_tokens, (int, float))
        and output_tokens > 0
    ):
        result["energy_per_output_token_joules"] = energy / output_tokens

    result["status"] = status
    result["error"] = error
    result["measured_repetitions"] = len(raw_records)
    result["failed_repetitions"] = failed_repetitions
    result["raw_output_files"] = [str(Path(path)) for path in raw_output_paths]
    result["telemetry_files"] = [str(Path(path)) for path in telemetry_paths]
    failed_requests = result.get("failed_requests")
    if isinstance(failed_requests, (int, float)) and failed_requests > 0:
        if result["status"] == "completed":
            result["status"] = "failed"
        combined_warnings.append(
            f"The backend reported {failed_requests:g} failed request(s) across measured "
            "repetitions."
        )
    requested = result.get("number_of_requests")
    successful_requests = result.get("successful_requests")
    expected_requests = (
        requested * len(raw_records) if isinstance(requested, int) and raw_records else None
    )
    if (
        expected_requests is not None
        and isinstance(successful_requests, (int, float))
        and successful_requests < expected_requests
    ):
        if result["status"] == "completed":
            result["status"] = "failed"
        combined_warnings.append(
            f"The backend reported {successful_requests:g} successful request(s), fewer "
            f"than the {expected_requests} requested across measured repetitions."
        )
    if failed_repetitions and status == "completed":
        result["status"] = "failed"
        combined_warnings.append(
            "Run status changed to failed because one or more configured measured "
            "repetitions are missing or unreadable."
        )
    if result["status"] == "failed" and error:
        combined_warnings.append(f"Benchmark failed: {error}")

    if result.get("model_precision") is None:
        combined_warnings.append(
            "Actual model_precision was not reported by the backend; dtype records only "
            "the requested setting."
        )
    if raw_records and result.get("successful_requests") is None:
        combined_warnings.append(
            "The backend output did not report actual successful-request counts; request "
            "completion could not be independently verified."
        )
    if result.get("energy_joules") is not None and result.get("successful_requests") is None:
        combined_warnings.append(
            "energy_per_request_joules is unavailable because the backend did not report "
            "the actual number of successful requests."
        )
    if result.get("energy_joules") is not None and result.get("actual_output_tokens") is None:
        combined_warnings.append(
            "energy_per_output_token_joules is unavailable because the backend did not "
            "report the actual number of generated output tokens."
        )

    unavailable = [field for field in NUMERIC_METRICS if result.get(field) is None]
    if unavailable:
        combined_warnings.append(
            "Unavailable metrics (not reported by the backend or optional telemetry): "
            + ", ".join(unavailable)
            + "."
        )
    result["warnings"] = list(dict.fromkeys(combined_warnings))
    return result


_MISSING = object()
_NULL_IS_MEANINGFUL = {
    "request_rate",
    "maximum_concurrency",
    "generation_config",
    "temperature",
    "top_p",
    "ignore_eos",
    "quantization",
    "gpu_memory_utilization",
}


def _matching_null_is_meaningful(
    field: str, left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if field in _NULL_IS_MEANINGFUL:
        return True
    both_non_gpu = left.get("hardware_type") != "gpu" and right.get("hardware_type") != "gpu"
    return both_non_gpu and field in {
        "accelerator_name",
        "software_versions.cuda_runtime",
        "software_versions.nvidia_driver",
    }


def _get_dotted(data: Mapping[str, Any], field: str) -> Any:
    value: Any = data
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def check_compatibility(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Report every missing or differing field needed for a fair comparison."""

    mismatches: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for field in COMPATIBILITY_FIELDS:
        left_value = _get_dotted(left, field)
        right_value = _get_dotted(right, field)
        if left_value is _MISSING or right_value is _MISSING:
            missing.append(
                {
                    "field": field,
                    "left": None if left_value is _MISSING else left_value,
                    "right": None if right_value is _MISSING else right_value,
                    "left_present": left_value is not _MISSING,
                    "right_present": right_value is not _MISSING,
                }
            )
        elif (
            left_value is None
            and right_value is None
            and _matching_null_is_meaningful(field, left, right)
        ):
            # Explicit null is meaningful for fields such as no quantization or the
            # non-applicable request rate/concurrency of an offline workload.
            continue
        elif left_value is None or right_value is None:
            difference = {"field": field, "left": left_value, "right": right_value}
            if field in _NULL_IS_MEANINGFUL:
                mismatches.append(difference)
            else:
                missing.append(difference)
        elif left_value != right_value:
            mismatches.append({"field": field, "left": left_value, "right": right_value})

    left_precision = left.get("model_precision")
    right_precision = right.get("model_precision")
    if left_precision is not None and right_precision is not None:
        if left_precision != right_precision:
            mismatches.append(
                {
                    "field": "model_precision",
                    "left": left_precision,
                    "right": right_precision,
                }
            )
    elif left.get("dtype") == "auto" or right.get("dtype") == "auto":
        # Matching ``auto`` values are only matching backend policies. They do not
        # prove that two different models were loaded at the same effective
        # precision, so ratios would overstate comparison validity.
        missing.append(
            {
                "field": "model_precision",
                "left": left_precision,
                "right": right_precision,
            }
        )
    elif left_precision is not None or right_precision is not None:
        # An explicitly controlled, matching dtype can stand on its own, but if one
        # backend result additionally reports an observed precision, require the
        # other result to do so as well.
        missing.append(
            {
                "field": "model_precision",
                "left": left_precision,
                "right": right_precision,
            }
        )

    successful = {"completed", "success", "successful"}
    failed_runs = [
        {
            "side": side,
            "run_id": run.get("run_id"),
            "status": run.get("status"),
            "error": run.get("error"),
        }
        for side, run in (("left", left), ("right", right))
        if run.get("status") not in successful
    ]
    if mismatches or failed_runs:
        status = "incompatible"
    elif missing:
        status = "partial"
    else:
        status = "compatible"
    return {
        "status": status,
        "mismatched_fields": mismatches,
        "missing_fields": missing,
        "failed_runs": failed_runs,
    }


def _run_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "experiment_name": summary.get("experiment_name"),
        "model_id": summary.get("model_id"),
        "model_revision": summary.get("model_revision"),
        "resolved_model_revision": summary.get("resolved_model_revision"),
        "tokenizer_id": summary.get("tokenizer_id"),
        "resolved_tokenizer_revision": summary.get("resolved_tokenizer_revision"),
        "model_parameter_scale": summary.get("model_parameter_scale"),
        "model_precision": summary.get("model_precision"),
        "quantization": summary.get("quantization"),
    }


def compare_summaries(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two explicit summaries without selecting, ranking, or discarding runs."""

    compatibility = check_compatibility(left, right)
    ratio_allowed = compatibility["status"] == "compatible"
    metrics: dict[str, dict[str, float | None]] = {}
    missing_metrics: list[str] = []
    for field in COMPARISON_METRICS:
        left_value = _finite_float(left.get(field))
        right_value = _finite_float(right.get(field))
        ratio = None
        if ratio_allowed and left_value not in (None, 0.0) and right_value is not None:
            candidate = right_value / left_value
            if math.isfinite(candidate):
                ratio = candidate
        if left_value is None or right_value is None:
            missing_metrics.append(field)
        metrics[field] = {
            "left": left_value,
            "right": right_value,
            "right_to_left_ratio": ratio,
        }

    warnings = [str(item) for item in left.get("warnings", [])]
    warnings.extend(str(item) for item in right.get("warnings", []))
    if compatibility["status"] != "compatible":
        warnings.append(
            "Ratios are omitted because compatibility is not fully established; this is "
            "not a fair headline comparison."
        )
    if missing_metrics:
        warnings.append(
            "Metrics missing from one or both runs: " + ", ".join(missing_metrics) + "."
        )
    if left.get("model_id") == right.get("model_id"):
        warnings.append("Both explicit runs use the same model_id.")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "comparison_scope": (
            "Inference-system performance and resource use; model quality is not evaluated."
        ),
        "ratio_definition": "right value divided by left value",
        "left_run": _run_identity(left),
        "right_run": _run_identity(right),
        "compatibility": compatibility,
        "metrics": metrics,
        "warnings": list(dict.fromkeys(warnings)),
    }


def write_comparison(
    comparison: Mapping[str, Any], output_directory: str | Path
) -> tuple[Path, Path]:
    """Write comparison.json and a long-form comparison.csv."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "comparison.json"
    csv_path = output / "comparison.csv"
    try:
        json_text = json.dumps(dict(comparison), indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ResultError(f"comparison is not valid JSON: {exc}") from exc
    json_path.write_text(json_text, encoding="utf-8")

    left = comparison.get("left_run", {})
    right = comparison.get("right_run", {})
    compatibility = comparison.get("compatibility", {})
    metrics = comparison.get("metrics", {})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "metric",
            "left_run_id",
            "left_model_id",
            "left_value",
            "right_run_id",
            "right_model_id",
            "right_value",
            "right_to_left_ratio",
            "compatibility_status",
            "mismatched_fields",
            "missing_compatibility_fields",
            "left_status",
            "right_status",
            "failed_runs",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for metric in COMPARISON_METRICS:
            values = metrics.get(metric, {})
            writer.writerow(
                {
                    "metric": metric,
                    "left_run_id": left.get("run_id"),
                    "left_model_id": left.get("model_id"),
                    "left_value": values.get("left"),
                    "right_run_id": right.get("run_id"),
                    "right_model_id": right.get("model_id"),
                    "right_value": values.get("right"),
                    "right_to_left_ratio": values.get("right_to_left_ratio"),
                    "compatibility_status": compatibility.get("status"),
                    "mismatched_fields": json.dumps(
                        compatibility.get("mismatched_fields", []), sort_keys=True
                    ),
                    "missing_compatibility_fields": json.dumps(
                        compatibility.get("missing_fields", []), sort_keys=True
                    ),
                    "left_status": left.get("status"),
                    "right_status": right.get("status"),
                    "failed_runs": json.dumps(compatibility.get("failed_runs", []), sort_keys=True),
                }
            )
    return json_path, csv_path


def compare_paths(
    left_path: str | Path,
    right_path: str | Path,
    output_directory: str | Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Load exactly two explicit runs, compare them, and write both output formats."""

    left = load_summary(left_path)
    right = load_summary(right_path)
    comparison = compare_summaries(left, right)
    json_path, csv_path = write_comparison(comparison, output_directory)
    return comparison, json_path, csv_path
