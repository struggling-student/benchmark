from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_bench.config import config_from_mapping
from llm_bench.measurements import (
    create_measurements,
    load_measurements,
    measurements_for_run,
    read_telemetry_series,
    write_measurements,
)
from llm_bench.results import create_summary, normalize_summary, write_summary


def _initial_summary(*, repetitions: int = 2) -> dict[str, object]:
    config = config_from_mapping(
        {
            "experiment_name": "measurement-test",
            "benchmark_type": "offline",
            "backend": "vllm",
            "model_id": "test-model",
            "number_of_prompts": 2,
            "input_length": 16,
            "output_length": 8,
            "repetitions": repetitions,
            "warmup_runs": 0,
        }
    )
    return create_summary(config, {"run_id": "test-run", "warnings": []})


def _telemetry(path: Path, *, power: int, memory: int) -> None:
    path.write_text(
        "timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]\n"
        f"1700000000,0,50,{memory},{power}\n"
        f"1700000000,1,70,{memory + 10},{power}\n"
        f"1700000001,0,60,{memory + 20},{power}\n"
        f"1700000001,1,80,{memory + 30},{power}\n",
        encoding="utf-8",
    )


def test_measurements_preserve_repetition_aggregation_semantics(tmp_path: Path) -> None:
    raw_paths = [tmp_path / "raw_vllm_output.json", tmp_path / "raw-2.json"]
    raw_paths[0].write_text(
        json.dumps(
            {
                "duration": 2,
                "requests_per_second": 4,
                "completed": 2,
                "total_output_tokens": 20,
            }
        ),
        encoding="utf-8",
    )
    raw_paths[1].write_text(
        json.dumps(
            {
                "duration": 4,
                "requests_per_second": 6,
                "completed": 2,
                "total_output_tokens": 30,
            }
        ),
        encoding="utf-8",
    )
    telemetry_paths = [tmp_path / "gpu_telemetry.csv", tmp_path / "gpu-2.csv"]
    _telemetry(telemetry_paths[0], power=100, memory=100)
    _telemetry(telemetry_paths[1], power=150, memory=200)

    normalized = normalize_summary(
        _initial_summary(),
        raw_output_paths=raw_paths,
        telemetry_paths=telemetry_paths,
    )
    measurements = create_measurements(
        normalized,
        run_directory=tmp_path,
        raw_output_paths=raw_paths,
        telemetry_paths=telemetry_paths,
    )
    records = measurements["records"]

    assert [record["status"] for record in records] == ["completed", "completed"]
    assert records[0]["raw_output_file"] == "raw_vllm_output.json"
    assert normalized["request_throughput_requests_per_second"] == pytest.approx(
        sum(record["metrics"]["request_throughput_requests_per_second"] for record in records) / 2
    )
    assert normalized["peak_gpu_memory_mib"] == max(
        record["metrics"]["peak_gpu_memory_mib"] for record in records
    )
    assert normalized["energy_joules"] == sum(
        record["metrics"]["energy_joules"] for record in records
    )
    assert normalized["energy_per_request_joules"] == pytest.approx(125.0)
    assert normalized["energy_per_output_token_joules"] == pytest.approx(10.0)

    artifact = tmp_path / "measurements.json"
    write_measurements(artifact, measurements)
    assert load_measurements(artifact) == measurements


def test_missing_repetition_and_unaligned_telemetry_are_explicit(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text('{"duration": 1}', encoding="utf-8")
    telemetry = tmp_path / "telemetry.csv"
    _telemetry(telemetry, power=100, memory=100)
    extra = tmp_path / "extra.csv"
    _telemetry(extra, power=100, memory=100)

    measurements = create_measurements(
        _initial_summary(),
        run_directory=tmp_path,
        raw_output_paths=[raw],
        telemetry_paths=[telemetry, extra],
    )

    assert [record["status"] for record in measurements["records"]] == [
        "completed",
        "unavailable",
    ]
    assert all(record["telemetry_file"] is None for record in measurements["records"])
    assert "could not be paired" in measurements["warnings"][0]


def test_legacy_measurements_resolve_stale_absolute_paths(tmp_path: Path) -> None:
    raw = tmp_path / "raw_vllm_output.json"
    raw.write_text('{"requests_per_second": 7}', encoding="utf-8")
    summary = _initial_summary(repetitions=1)
    summary["raw_output_files"] = ["/old/cluster/path/raw_vllm_output.json"]

    measurements, legacy = measurements_for_run(tmp_path, summary)

    assert legacy is True
    assert measurements["source"] == "legacy-derived"
    assert measurements["records"][0]["metrics"][
        "request_throughput_requests_per_second"
    ] == pytest.approx(7.0)


def test_telemetry_series_aggregates_multiple_gpus(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    _telemetry(telemetry, power=100, memory=100)

    series = read_telemetry_series(telemetry)

    assert series == [
        {
            "elapsed_seconds": 0.0,
            "average_gpu_utilization_percent": 60.0,
            "total_gpu_memory_mib": 210.0,
            "total_gpu_power_watts": 200.0,
        },
        {
            "elapsed_seconds": 1.0,
            "average_gpu_utilization_percent": 70.0,
            "total_gpu_memory_mib": 250.0,
            "total_gpu_power_watts": 200.0,
        },
    ]


def test_telemetry_series_reads_cpu_memory_and_power_samples(tmp_path: Path) -> None:
    telemetry = tmp_path / "cpu-telemetry.csv"
    telemetry.write_text(
        (
            "timestamp,cpu_utilization_percent,rss_mib,package_power_watts,"
            "memory_bandwidth_gbps\n"
            "1700000000,80,12000,300,410\n"
            "1700000001,90,12500,320,430\n"
        ),
        encoding="utf-8",
    )

    series = read_telemetry_series(telemetry)

    assert series[0]["average_cpu_utilization_percent"] == pytest.approx(80)
    assert series[0]["total_cpu_memory_mib"] == pytest.approx(12000)
    assert series[0]["total_cpu_power_watts"] == pytest.approx(300)
    assert series[0]["memory_bandwidth_gbps"] == pytest.approx(410)
    assert series[1]["elapsed_seconds"] == pytest.approx(1)


def test_measurement_writer_handles_generic_raw_output(tmp_path: Path) -> None:
    config = config_from_mapping(
        {
            "experiment_name": "cli-test",
            "benchmark_type": "offline",
            "backend": "vllm",
            "model_id": "test-model",
            "number_of_prompts": 2,
            "input_length": 16,
            "output_length": 8,
            "repetitions": 1,
            "warmup_runs": 0,
        }
    )
    run_dir = tmp_path / "run"
    write_summary(run_dir / "summary.json", create_summary(config, {"run_id": "cli-run"}))
    raw = run_dir / "raw_vllm_output.json"
    raw.write_text('{"requests_per_second": 2}', encoding="utf-8")

    normalized = normalize_summary(
        create_summary(config, {"run_id": "cli-run"}), raw_output_paths=[raw]
    )
    measurements = create_measurements(
        normalized, run_directory=run_dir, raw_output_paths=[raw]
    )
    write_measurements(run_dir / "measurements.json", measurements)

    assert load_measurements(run_dir / "measurements.json")["run_id"] == "cli-run"
