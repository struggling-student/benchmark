"""Stable repetition-level benchmark artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
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
