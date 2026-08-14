from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import pytest

from llm_bench.catalog import CatalogEntry, ResultCatalog, catalog_row
from llm_bench.dashboard import _memory_treatment_compatibility
from llm_bench.dashboard_data import (
    combine_datasets,
    dataset_from_catalog,
    hardware_platform,
    measurements_for_entry,
    simulated_dataset,
)
from llm_bench.metrics import METRIC_SPECS
from llm_bench.results import SUMMARY_FIELDS


def test_simulated_study_is_deterministic_finite_and_in_memory(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    first = simulated_dataset()
    second = simulated_dataset()

    assert len(first.catalog.entries) == 18
    assert [entry.summary for entry in first.catalog.entries] == [
        entry.summary for entry in second.catalog.entries
    ]
    assert first.measurements == second.measurements
    assert first.telemetry == second.telemetry
    assert len({entry.key for entry in first.catalog.entries}) == 18
    assert list(tmp_path.iterdir()) == before

    json.dumps(
        {
            "summaries": [entry.summary for entry in first.catalog.entries],
            "measurements": first.measurements,
            "telemetry": first.telemetry,
        },
        allow_nan=False,
    )


def test_simulated_study_covers_gpu_cpu_ddr_hbm_and_failure_states() -> None:
    dataset = simulated_dataset()
    platforms = {hardware_platform(entry.summary) for entry in dataset.catalog.entries}
    statuses = {entry.summary["status"] for entry in dataset.catalog.entries}

    assert platforms == {"GPU", "CPU · DDR", "CPU · HBM"}
    assert statuses == {"completed", "failed"}
    assert {entry.summary["backend"] for entry in dataset.catalog.entries} == {
        "vllm",
        "llamacpp",
    }
    assert all(
        set(SUMMARY_FIELDS).issubset(entry.summary) for entry in dataset.catalog.entries
    )
    assert all(entry.summary["is_simulated"] for entry in dataset.catalog.entries)

    cpu = [
        entry
        for entry in dataset.catalog.entries
        if entry.summary["hardware_type"] == "cpu"
    ]
    gpu = [
        entry
        for entry in dataset.catalog.entries
        if entry.summary["hardware_type"] == "gpu"
    ]
    assert all(entry.summary["peak_gpu_memory_mib"] is None for entry in cpu)
    assert all(entry.summary.get("peak_cpu_memory_mib") is None for entry in gpu)
    assert all(entry.summary.get("measured_memory_bandwidth_gbps") for entry in cpu)


def test_simulated_repetitions_respect_metric_aggregation_semantics() -> None:
    dataset = simulated_dataset()
    entry = next(
        entry
        for entry in dataset.catalog.entries
        if entry.summary["status"] == "completed"
    )
    measurements, legacy = measurements_for_entry(dataset, entry)

    assert not legacy
    assert len(measurements["records"]) == 5
    for spec in METRIC_SPECS:
        summary_value = entry.summary.get(spec.field)
        if not isinstance(summary_value, (int, float)):
            continue
        values = [
            record["metrics"][spec.field] for record in measurements["records"]
        ]
        if spec.aggregation == "maximum":
            assert max(values) == pytest.approx(summary_value)
        elif spec.aggregation == "sum":
            assert sum(values) == pytest.approx(summary_value)
        else:
            assert fmean(values) == pytest.approx(summary_value)


def test_memory_treatment_requires_matching_controls_and_distinct_memory() -> None:
    dataset = simulated_dataset()
    ddr = next(
        entry
        for entry in dataset.catalog.entries
        if entry.summary["run_id"] == "demo-cpu-ddr5-8b-offline"
    )
    hbm = next(
        entry
        for entry in dataset.catalog.entries
        if entry.summary["run_id"] == "demo-cpu-hbm2e-8b-offline"
    )

    compatible = _memory_treatment_compatibility(ddr.summary, hbm.summary)
    assert compatible == {"status": "compatible", "evidence": []}

    changed = dict(hbm.summary)
    changed["thread_count"] = 56
    incompatible = _memory_treatment_compatibility(ddr.summary, changed)
    assert incompatible["status"] == "incompatible"
    assert any(item["field"] == "thread_count" for item in incompatible["evidence"])


def test_combining_datasets_preserves_measured_provenance_and_sorting(
    tmp_path: Path,
) -> None:
    summary = {
        "run_id": "measured",
        "timestamp": "2026-07-26T12:00:00+00:00",
        "status": "completed",
        "model_id": "measured-model",
        "warnings": [],
        "hardware_type": "gpu",
        "accelerator_name": "measured-gpu",
    }
    path = tmp_path / "measured" / "summary.json"
    entry = CatalogEntry(path, path.parent, summary)
    measured = dataset_from_catalog(ResultCatalog(tmp_path, (entry,), ()))

    combined = combine_datasets(measured, simulated_dataset())

    assert len(combined.catalog.entries) == 19
    assert combined.catalog.entries[0].key == entry.key
    assert not combined.is_simulated(entry)
    assert catalog_row(entry)["source"] == "Measured"
