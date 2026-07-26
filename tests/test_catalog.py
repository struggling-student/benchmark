from __future__ import annotations

import json
from pathlib import Path

from llm_bench.catalog import catalog_row, discover_results
from llm_bench.results import SCHEMA_VERSION


def _summary(run_id: str, *, benchmark_type: str = "smoke") -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": "2026-07-20T12:00:00+00:00",
        "status": "completed",
        "warnings": [],
        "model_id": "test-model",
        "model_precision": "float16",
        "dtype": "float16",
        "benchmark_type": benchmark_type,
        "hardware_type": "gpu",
        "accelerator_name": "test-gpu",
        "input_length": 16,
        "output_length": 8,
        "number_of_requests": 2,
        "request_rate": None,
        "maximum_concurrency": None,
    }


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_catalog_discovers_mixed_runs_and_keeps_duplicate_paths(tmp_path: Path) -> None:
    _write(tmp_path / "2026-07-20" / "a" / "summary.json", _summary("duplicate"))
    _write(
        tmp_path / "2026-07-20" / "b" / "summary.json",
        _summary("duplicate", benchmark_type="serving"),
    )
    _write(tmp_path / "bad" / "summary.json", {"schema_version": "99", "run_id": "bad"})
    malformed = tmp_path / "malformed" / "summary.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")

    catalog = discover_results(tmp_path)

    assert len(catalog.entries) == 2
    assert len({entry.key for entry in catalog.entries}) == 2
    messages = [item.message for item in catalog.diagnostics]
    assert sum("Duplicate run_id" in message for message in messages) == 2
    assert any(
        "unsupported" in message.lower() or "missing" in message.lower() for message in messages
    )
    assert any("cannot read summary" in message for message in messages)
    row = catalog_row(catalog.entries[0])
    assert row["date"] == "2026-07-20"
    assert "input=16" in row["workload"]


def test_missing_results_root_is_a_diagnostic(tmp_path: Path) -> None:
    catalog = discover_results(tmp_path / "missing")

    assert catalog.entries == ()
    assert "does not exist" in catalog.diagnostics[0].message


def test_catalog_accepts_legacy_schema_with_diagnostic(tmp_path: Path) -> None:
    legacy = _summary("legacy")
    legacy["schema_version"] = "1.0"
    _write(tmp_path / "legacy" / "summary.json", legacy)

    catalog = discover_results(tmp_path)

    assert len(catalog.entries) == 1
    assert "Legacy summary schema" in catalog.diagnostics[0].message
