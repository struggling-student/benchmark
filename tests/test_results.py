from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_bench.config import config_from_mapping
from llm_bench.results import (
    SUMMARY_FIELDS,
    check_compatibility,
    compare_paths,
    compare_summaries,
    create_summary,
    load_summary,
    normalize_summary,
    write_summary,
)


def experiment(*, model_id: str = "model-a", repetitions: int = 1):
    return config_from_mapping(
        {
            "experiment_name": "offline-comparison",
            "benchmark_type": "offline",
            "backend": "vllm",
            "model_id": model_id,
            "model_parameter_scale": "1B" if model_id == "model-a" else "8B",
            "dtype": "bfloat16",
            "quantization": None,
            "tensor_parallel_size": 1,
            "seed": 42,
            "number_of_prompts": 8,
            "input_length": 128,
            "output_length": 32,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.9,
            "repetitions": repetitions,
            "warmup_runs": 1,
        }
    )


def metadata(*, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "timestamp": "2026-07-19T12:00:00+00:00",
        "benchmark_type": "offline",
        "backend_version": "0.10.0",
        "git_commit": "abc123",
        "git_dirty": False,
        "hostname": "node1",
        "hardware_type": "gpu",
        "accelerator_name": "test-gpu",
        "accelerator_count": 1,
        "cpu_model": "test-cpu",
        "socket_count": 1,
        "numa_node_count": 1,
        "software_versions": {
            "python": "3.11.9",
            "llm_bench": "0.1.0",
            "vllm": "0.10.0",
            "torch": "2.7.0",
            "cuda_runtime": "12.8",
            "nvidia_driver": "575.0",
        },
    }


def completed_summary(*, model_id: str, run_id: str, throughput: float) -> dict[str, object]:
    summary = create_summary(experiment(model_id=model_id), metadata(run_id=run_id))
    summary.update(
        {
            "status": "completed",
            "measured_repetitions": 1,
            "model_precision": "bfloat16",
            "request_throughput_requests_per_second": throughput,
            "output_throughput_tokens_per_second": throughput * 32,
            "duration_seconds": 4.0,
            "warnings": [],
        }
    )
    return summary


def test_summary_serialization_keeps_schema_and_nulls(tmp_path: Path) -> None:
    summary = create_summary(experiment(), metadata(run_id="run-a"))
    destination = tmp_path / "summary.json"

    write_summary(destination, summary)
    loaded = load_summary(destination)

    assert set(SUMMARY_FIELDS).issubset(loaded)
    assert loaded["mean_ttft_ms"] is None
    assert loaded["model_id"] == "model-a"
    assert loaded["dtype"] == "bfloat16"
    assert loaded["model_precision"] == "bfloat16"
    assert loaded["max_model_len"] == 8192
    assert loaded["warnings"] == ["Metrics are null until the run is finalized."]


def test_normalization_averages_all_repetitions_and_never_selects_best(
    tmp_path: Path,
) -> None:
    first = tmp_path / "raw-1.json"
    second = tmp_path / "raw-2.json"
    first.write_text(
        json.dumps(
            {
                "elapsed_time": 10.0,
                "requests_per_second": 1.0,
                "tokens_per_second": 100.0,
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "elapsed_time": 20.0,
                "requests_per_second": 3.0,
                "tokens_per_second": 300.0,
            }
        ),
        encoding="utf-8",
    )
    initial = create_summary(experiment(repetitions=2), metadata(run_id="repeated-run"))

    normalized = normalize_summary(initial, raw_output_paths=[first, second], status="completed")

    assert normalized["status"] == "completed"
    assert normalized["measured_repetitions"] == 2
    assert normalized["duration_seconds"] == pytest.approx(15.0)
    assert normalized["request_throughput_requests_per_second"] == pytest.approx(2.0)
    assert normalized["total_throughput_tokens_per_second"] == pytest.approx(200.0)
    assert normalized["raw_output_files"] == [str(first), str(second)]


def test_missing_repetition_marks_completed_request_failed(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text('{"elapsed_time": 1.0}', encoding="utf-8")
    initial = create_summary(experiment(repetitions=2), metadata(run_id="partial-run"))

    normalized = normalize_summary(initial, raw_output_paths=[raw], status="completed")

    assert normalized["status"] == "failed"
    assert normalized["failed_repetitions"] == 1
    assert any("Expected 2 measured" in item for item in normalized["warnings"])


def test_energy_uses_actual_counts_reported_by_backend(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "elapsed_time": 1.0,
                "completed": 8,
                "total_output_tokens": 200,
            }
        ),
        encoding="utf-8",
    )
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "timestamp,index,power.draw [W]\n1700000000,0,100\n1700000001,0,100\n",
        encoding="utf-8",
    )
    initial = create_summary(experiment(), metadata(run_id="energy-run"))

    normalized = normalize_summary(
        initial,
        raw_output_paths=[raw],
        telemetry_paths=[telemetry],
        status="completed",
    )

    assert normalized["successful_requests"] == 8
    assert normalized["actual_output_tokens"] == 200
    assert normalized["energy_joules"] == pytest.approx(100.0)
    assert normalized["energy_per_request_joules"] == pytest.approx(12.5)
    assert normalized["energy_per_output_token_joules"] == pytest.approx(0.5)


def test_backend_request_failures_are_preserved_in_status(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps({"completed": 7, "failed": 1, "elapsed_time": 1.0}),
        encoding="utf-8",
    )
    initial = create_summary(experiment(), metadata(run_id="request-failure"))

    normalized = normalize_summary(initial, raw_output_paths=[raw], status="completed")

    assert normalized["status"] == "failed"
    assert normalized["successful_requests"] == 7
    assert normalized["failed_requests"] == 1
    assert any("failed request" in item for item in normalized["warnings"])


def test_compatible_comparison_includes_absolute_values_and_ratio() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)

    comparison = compare_summaries(left, right)

    assert comparison["compatibility"]["status"] == "compatible"
    metric = comparison["metrics"]["request_throughput_requests_per_second"]
    assert metric == {"left": 2.0, "right": 3.0, "right_to_left_ratio": 1.5}
    assert "quality is not evaluated" in comparison["comparison_scope"]


def test_unreported_resolved_precision_makes_auto_policy_partial() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    left["dtype"] = right["dtype"] = "auto"
    left["model_precision"] = right["model_precision"] = None

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "partial"
    assert compatibility["missing_fields"] == [
        {"field": "model_precision", "left": None, "right": None}
    ]
    assert (
        compare_summaries(left, right)["metrics"][
            "request_throughput_requests_per_second"
        ]["right_to_left_ratio"]
        is None
    )


def test_actual_measured_repetition_difference_is_incompatible() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    right["measured_repetitions"] = 2

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "incompatible"
    assert compatibility["mismatched_fields"] == [
        {"field": "measured_repetitions", "left": 1, "right": 2}
    ]


def test_hardware_difference_is_incompatible() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    right["accelerator_name"] = "different-test-gpu"

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "incompatible"
    assert compatibility["mismatched_fields"] == [
        {
            "field": "accelerator_name",
            "left": "test-gpu",
            "right": "different-test-gpu",
        }
    ]


def test_cpu_pairs_require_memory_and_placement_controls() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    cpu_controls = {
        "hardware_type": "cpu",
        "accelerator_name": None,
        "accelerator_count": 0,
        "gpu_memory_utilization": None,
        "cpu_model": "test-cpu",
        "memory_type": "HBM2e",
        "memory_mode": "flat",
        "thread_count": 64,
        "thread_affinity": "compact",
        "process_count": 2,
        "numa_policy": "local",
        "memory_binding": "hbm",
        "cpu_isa": "AVX-512",
    }
    left.update(cpu_controls)
    right.update(cpu_controls)
    for summary in (left, right):
        summary["software_versions"]["cuda_runtime"] = None
        summary["software_versions"]["nvidia_driver"] = None

    assert check_compatibility(left, right)["status"] == "compatible"

    right["memory_type"] = "DDR5"
    compatibility = check_compatibility(left, right)
    assert compatibility["status"] == "incompatible"
    assert {"field": "memory_type", "left": "HBM2e", "right": "DDR5"} in compatibility[
        "mismatched_fields"
    ]


def test_missing_cpu_placement_evidence_makes_comparison_partial() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    for summary in (left, right):
        summary.update(
            {
                "hardware_type": "cpu",
                "accelerator_name": None,
                "accelerator_count": 0,
                "gpu_memory_utilization": None,
                "memory_type": "HBM2e",
                "memory_mode": "flat",
                "thread_count": 64,
                "thread_affinity": None,
                "process_count": 2,
                "numa_policy": "local",
                "memory_binding": "hbm",
                "cpu_isa": "AVX-512",
            }
        )
        summary["software_versions"]["cuda_runtime"] = None
        summary["software_versions"]["nvidia_driver"] = None

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "partial"
    assert {"field": "thread_affinity", "left": None, "right": None} in compatibility[
        "missing_fields"
    ]


def test_incompatible_workload_reports_mismatch_and_omits_ratios() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    right["input_length"] = 256

    compatibility = check_compatibility(left, right)
    comparison = compare_summaries(left, right)

    assert compatibility["status"] == "incompatible"
    assert compatibility["mismatched_fields"] == [
        {"field": "input_length", "left": 128, "right": 256}
    ]
    assert all(values["right_to_left_ratio"] is None for values in comparison["metrics"].values())


def test_generation_control_difference_is_incompatible() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    left["temperature"] = 0.0
    right["temperature"] = 0.5

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "incompatible"
    assert compatibility["mismatched_fields"] == [
        {"field": "temperature", "left": 0.0, "right": 0.5}
    ]


def test_max_model_len_difference_is_incompatible() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    right["max_model_len"] = 16384

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "incompatible"
    assert compatibility["mismatched_fields"] == [
        {"field": "max_model_len", "left": 8192, "right": 16384}
    ]


def test_null_versus_enabled_quantization_is_a_mismatch() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    right["quantization"] = "awq"

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "incompatible"
    assert {item["field"] for item in compatibility["mismatched_fields"]} == {"quantization"}


def test_failed_run_is_not_discarded() -> None:
    left = completed_summary(model_id="model-a", run_id="left", throughput=2.0)
    right = completed_summary(model_id="model-b", run_id="right", throughput=3.0)
    right.update({"status": "failed", "error": "server startup timed out"})

    compatibility = check_compatibility(left, right)

    assert compatibility["status"] == "incompatible"
    assert compatibility["failed_runs"][0]["run_id"] == "right"
    assert compatibility["failed_runs"][0]["error"] == "server startup timed out"


def test_compare_paths_accepts_explicit_result_directories(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    write_summary(
        left_dir / "summary.json",
        completed_summary(model_id="model-a", run_id="left", throughput=2.0),
    )
    write_summary(
        right_dir / "summary.json",
        completed_summary(model_id="model-b", run_id="right", throughput=3.0),
    )

    comparison, json_path, csv_path = compare_paths(left_dir, right_dir, tmp_path / "comparison")

    assert comparison["compatibility"]["status"] == "compatible"
    assert json_path.name == "comparison.json" and json_path.is_file()
    assert csv_path.name == "comparison.csv" and csv_path.is_file()
    assert "right_to_left_ratio" in csv_path.read_text(encoding="utf-8")
