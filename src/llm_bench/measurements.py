"""Stable repetition-level artifacts and telemetry series for result presentation."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from .results import NUMERIC_METRICS, ResultError, read_raw_metrics, read_telemetry_metrics

MEASUREMENTS_SCHEMA_VERSION = "1.0"
MEASUREMENT_VALUE_FIELDS = (
    *NUMERIC_METRICS,
    "successful_requests",
    "failed_requests",
    "actual_input_tokens",
    "actual_output_tokens",
    "model_precision",
)


def _empty_values() -> dict[str, Any]:
    return {field: None for field in MEASUREMENT_VALUE_FIELDS}


def _portable_path(path: str | Path, run_directory: Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(run_directory.resolve()))
    except (OSError, ValueError):
        return candidate.name


def resolve_recorded_path(path: str | Path | None, run_directory: str | Path) -> Path | None:
    """Resolve a recorded path after a result directory may have been copied."""

    if path is None or not str(path).strip():
        return None
    recorded = Path(path)
    if recorded.is_file():
        return recorded
    root = Path(run_directory)
    relative = root / recorded
    if relative.is_file():
        return relative
    by_name = root / recorded.name
    return by_name if by_name.is_file() else recorded


def create_measurements(
    summary: Mapping[str, Any],
    *,
    run_directory: str | Path,
    raw_output_paths: Sequence[str | Path] = (),
    telemetry_paths: Sequence[str | Path] = (),
    source: str = "normalized",
) -> dict[str, Any]:
    """Normalize each configured measured repetition without changing run aggregates."""

    root = Path(run_directory)
    configured = summary.get("repetitions")
    expected = (
        configured if isinstance(configured, int) and configured >= 0 else len(raw_output_paths)
    )
    expected = max(expected, len(raw_output_paths))
    telemetry_aligned = not telemetry_paths or len(telemetry_paths) == len(raw_output_paths)
    document_warnings: list[str] = []
    if telemetry_paths and not telemetry_aligned:
        document_warnings.append(
            "Telemetry files could not be paired one-to-one with measured repetitions; "
            "repetition telemetry values were left null."
        )

    records: list[dict[str, Any]] = []
    for index in range(expected):
        raw_path = Path(raw_output_paths[index]) if index < len(raw_output_paths) else None
        telemetry_path = (
            Path(telemetry_paths[index])
            if telemetry_aligned and index < len(telemetry_paths)
            else None
        )
        values = _empty_values()
        warnings: list[str] = []
        status = "unavailable"
        raw_metrics: dict[str, Any] = {}
        if raw_path is not None:
            try:
                raw_metrics = read_raw_metrics(
                    raw_path,
                    backend=str(summary.get("backend") or "") or None,
                    measurement_method=(
                        str(summary["measurement_method"])
                        if summary.get("measurement_method")
                        else None
                    ),
                )
                values.update({key: value for key, value in raw_metrics.items() if key in values})
                status = "completed"
            except ResultError as exc:
                warnings.append(str(exc))

        if telemetry_path is not None:
            try:
                telemetry_metrics, telemetry_warnings = read_telemetry_metrics(
                    telemetry_path,
                    duration_seconds=raw_metrics.get("duration_seconds"),
                )
                values.update(telemetry_metrics)
                warnings.extend(telemetry_warnings)
            except ResultError as exc:
                warnings.append(str(exc))

        energy = values.get("energy_joules")
        requests = values.get("successful_requests")
        output_tokens = values.get("actual_output_tokens")
        if isinstance(energy, (int, float)) and isinstance(requests, (int, float)) and requests > 0:
            values["energy_per_request_joules"] = energy / requests
        if (
            isinstance(energy, (int, float))
            and isinstance(output_tokens, (int, float))
            and output_tokens > 0
        ):
            values["energy_per_output_token_joules"] = energy / output_tokens

        records.append(
            {
                "repetition": index + 1,
                "status": status,
                "raw_output_file": (
                    _portable_path(raw_path, root) if raw_path is not None else None
                ),
                "telemetry_file": (
                    _portable_path(telemetry_path, root) if telemetry_path is not None else None
                ),
                "metrics": values,
                "warnings": list(dict.fromkeys(warnings)),
            }
        )

    return {
        "schema_version": MEASUREMENTS_SCHEMA_VERSION,
        "run_id": summary.get("run_id"),
        "expected_repetitions": expected,
        "source": source,
        "records": records,
        "warnings": document_warnings,
    }


def write_measurements(path: str | Path, measurements: Mapping[str, Any]) -> None:
    """Write a deterministic, finite repetition-level artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(dict(measurements), indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ResultError(f"measurements are not valid JSON: {exc}") from exc
    destination.write_text(encoded, encoding="utf-8")


def load_measurements(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one measurements artifact."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultError(f"cannot read measurements {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ResultError(f"measurements must contain a JSON object: {source}")
    if data.get("schema_version") != MEASUREMENTS_SCHEMA_VERSION:
        raise ResultError(
            f"unsupported measurements schema {data.get('schema_version')!r}: {source}"
        )
    if not isinstance(data.get("records"), list):
        raise ResultError(f"measurements records must be a list: {source}")
    return data


def measurements_for_run(
    run_directory: str | Path, summary: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Load stable measurements or derive a read-only legacy view from preserved files."""

    root = Path(run_directory)
    artifact = root / "measurements.json"
    if artifact.is_file():
        return load_measurements(artifact), False

    raw_paths = [
        resolved
        for item in summary.get("raw_output_files", [])
        if (resolved := resolve_recorded_path(item, root)) is not None and resolved.is_file()
    ]
    telemetry_paths = [
        resolved
        for item in summary.get("telemetry_files", [])
        if (resolved := resolve_recorded_path(item, root)) is not None and resolved.is_file()
    ]
    derived = create_measurements(
        summary,
        run_directory=root,
        raw_output_paths=raw_paths,
        telemetry_paths=telemetry_paths,
        source="legacy-derived",
    )
    derived["warnings"] = [
        "measurements.json was absent; repetition values were derived in memory from "
        "preserved legacy files.",
        *derived["warnings"],
    ]
    return derived, True


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    match = re.match(r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", text)
    if match is None:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: str | None, fallback: float) -> float:
    numeric = _finite(value)
    if numeric is not None and numeric > 1_000_000:
        return numeric
    if value:
        cleaned = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            pass
    return fallback


def _find_row_value(normalized: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        if (key := _normalized_key(alias)) in normalized:
            return normalized[key]
    return None


def read_telemetry_series(path: str | Path) -> list[dict[str, float | None]]:
    """Return GPU or CPU allocation telemetry grouped by timestamp for plotting."""

    source = Path(path)
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise ResultError(f"cannot read resource telemetry {source}: {exc}") from exc

    samples: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        normalized = {_normalized_key(key): value for key, value in row.items() if key}
        timestamp = _timestamp(
            _find_row_value(normalized, ("timestamp", "timestamp_iso", "time")),
            float(index),
        )
        for name, aliases in (
            (
                "average_gpu_utilization_percent",
                ("utilization.gpu [%]", "utilization_gpu", "gpu_utilization_percent"),
            ),
            ("total_gpu_memory_mib", ("memory.used [MiB]", "memory_used_mib", "memory_used")),
            ("total_gpu_power_watts", ("power.draw [W]", "power_draw_watts", "power_draw")),
            (
                "average_cpu_utilization_percent",
                (
                    "cpu.utilization [%]",
                    "cpu_utilization_percent",
                    "cpu_usage_percent",
                    "utilization_cpu",
                ),
            ),
            (
                "total_cpu_memory_mib",
                (
                    "rss_mib",
                    "resident_memory_mib",
                    "cpu_memory_used_mib",
                    "host_memory_used_mib",
                ),
            ),
            (
                "total_cpu_power_watts",
                (
                    "package_power_watts",
                    "cpu_package_power_watts",
                    "total_cpu_power_watts",
                    "package_power_w",
                ),
            ),
            (
                "memory_bandwidth_gbps",
                (
                    "memory_bandwidth_gbps",
                    "memory_bandwidth_gb_s",
                    "bandwidth_gbps",
                    "total_memory_bandwidth_gbps",
                ),
            ),
        ):
            value = _finite(_find_row_value(normalized, aliases))
            if value is not None:
                samples[timestamp][name].append(value)

    if not samples:
        return []
    first = min(samples)
    series: list[dict[str, float | None]] = []
    for timestamp in sorted(samples):
        sample = samples[timestamp]
        utilization = sample["average_gpu_utilization_percent"]
        memory = sample["total_gpu_memory_mib"]
        power = sample["total_gpu_power_watts"]
        cpu_utilization = sample["average_cpu_utilization_percent"]
        cpu_memory = sample["total_cpu_memory_mib"]
        cpu_power = sample["total_cpu_power_watts"]
        memory_bandwidth = sample["memory_bandwidth_gbps"]
        record = {
            "elapsed_seconds": timestamp - first,
            "average_gpu_utilization_percent": fmean(utilization) if utilization else None,
            "total_gpu_memory_mib": sum(memory) if memory else None,
            "total_gpu_power_watts": sum(power) if power else None,
        }
        if cpu_utilization:
            record["average_cpu_utilization_percent"] = fmean(cpu_utilization)
        if cpu_memory:
            record["total_cpu_memory_mib"] = sum(cpu_memory)
        if cpu_power:
            record["total_cpu_power_watts"] = sum(cpu_power)
        if memory_bandwidth:
            record["memory_bandwidth_gbps"] = sum(memory_bandwidth)
        series.append(record)
    return series
