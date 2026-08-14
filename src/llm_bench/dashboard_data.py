"""Dashboard-only data adapters and a deterministic simulated benchmark study.

The simulated records deliberately stay in memory.  They exercise the same catalog and
presentation paths as measured results without writing fake ``summary.json`` files into a
research results directory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .catalog import CatalogEntry, ResultCatalog
from .measurements import measurements_for_run
from .metrics import METRIC_SPECS
from .results import SCHEMA_VERSION, SUMMARY_FIELDS

SIMULATED_SOURCE = "Simulated"
MEASURED_SOURCE = "Measured"
DEMO_REPETITIONS = 5


@dataclass(frozen=True)
class DashboardDataset:
    """A catalog plus optional in-memory sidecars used by the dashboard."""

    catalog: ResultCatalog
    measurements: Mapping[str, Mapping[str, Any]]
    telemetry: Mapping[str, Sequence[Mapping[str, float | None]]]
    simulated_keys: frozenset[str]

    def is_simulated(self, entry: CatalogEntry) -> bool:
        return entry.key in self.simulated_keys

    def source_label(self, entry: CatalogEntry) -> str:
        return SIMULATED_SOURCE if self.is_simulated(entry) else MEASURED_SOURCE


def dataset_from_catalog(catalog: ResultCatalog) -> DashboardDataset:
    """Wrap a filesystem catalog in the dashboard data abstraction."""

    return DashboardDataset(catalog, {}, {}, frozenset())


def combine_datasets(
    measured: DashboardDataset, simulated: DashboardDataset
) -> DashboardDataset:
    """Combine measured and simulated entries while retaining measured diagnostics."""

    entries = [*measured.catalog.entries, *simulated.catalog.entries]
    entries.sort(key=lambda entry: str(entry.summary.get("timestamp") or ""), reverse=True)
    return DashboardDataset(
        ResultCatalog(
            measured.catalog.root,
            tuple(entries),
            measured.catalog.diagnostics,
        ),
        {**measured.measurements, **simulated.measurements},
        {**measured.telemetry, **simulated.telemetry},
        measured.simulated_keys | simulated.simulated_keys,
    )


def measurements_for_entry(
    dataset: DashboardDataset, entry: CatalogEntry
) -> tuple[dict[str, Any], bool]:
    """Load repetition data from a sidecar or the measured result directory."""

    if entry.key in dataset.measurements:
        return dict(dataset.measurements[entry.key]), False
    return measurements_for_run(entry.run_directory, entry.summary)


def telemetry_for_entry(
    dataset: DashboardDataset, entry: CatalogEntry
) -> list[dict[str, float | None]] | None:
    """Return an in-memory telemetry timeline when one is registered."""

    series = dataset.telemetry.get(entry.key)
    if series is None:
        return None
    return [dict(sample) for sample in series]


def memory_system(summary: Mapping[str, Any]) -> str:
    """Return a compact memory-system label without forcing GPU fields onto CPUs."""

    value = summary.get("memory_type") or summary.get("memory_system")
    if value:
        return str(value)
    return "Unspecified"


def hardware_platform(summary: Mapping[str, Any]) -> str:
    """Classify a run into the platform groups used throughout the dashboard."""

    hardware_type = str(summary.get("hardware_type") or "unknown").lower()
    memory = memory_system(summary).lower()
    if hardware_type == "gpu":
        return "GPU"
    if hardware_type == "cpu":
        if "hbm" in memory:
            return "CPU · HBM"
        if "ddr" in memory:
            return "CPU · DDR"
        return "CPU"
    return hardware_type.upper()


def hardware_name(summary: Mapping[str, Any]) -> str:
    """Return the applicable processor/accelerator name for a summary."""

    if summary.get("hardware_type") == "gpu":
        return str(summary.get("accelerator_name") or "Unknown GPU")
    return str(summary.get("cpu_model") or "Unknown CPU")


def _base_summary() -> dict[str, Any]:
    summary = {field: None for field in SUMMARY_FIELDS}
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "warnings": [
                "Simulated preview data: values are illustrative and must not be cited as "
                "measured benchmark results."
            ],
            "raw_output_files": [],
            "telemetry_files": [],
            "software_versions": {},
            "data_source": "simulated",
            "is_simulated": True,
        }
    )
    return summary


def _shared_identity(
    *,
    run_id: str,
    timestamp: datetime,
    benchmark_type: str,
    model_id: str,
    model_scale: str,
    backend: str,
    input_length: int,
    output_length: int,
    number_of_requests: int,
    request_rate: float | None,
    maximum_concurrency: int | None,
) -> dict[str, Any]:
    summary = _base_summary()
    summary.update(
        {
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            "experiment_name": run_id,
            "benchmark_type": benchmark_type,
            "backend": backend,
            "backend_version": "preview-1",
            "backend_profile": "vllm_gpu" if backend == "vllm" else "llamacpp_cpu",
            "execution_provider": "native",
            "measurement_method": (
                "backend_native" if benchmark_type == "offline" else "shared_openai_streaming"
            ),
            "measurement_scope": (
                "backend_native_no_common_boundary"
                if benchmark_type == "offline"
                else "client_observed_end_to_end"
            ),
            "workload_manifest_sha256": (
                None if benchmark_type == "offline" else f"demo-{model_scale}-{input_length}"
            ),
            "model_id": model_id,
            "model_revision": "demo-revision",
            "resolved_model_revision": "demo-revision",
            "tokenizer_id": model_id,
            "resolved_tokenizer_revision": "demo-revision",
            "model_parameter_scale": model_scale,
            "model_precision": "float16",
            "model_artifact_format": "huggingface" if backend == "vllm" else "gguf",
            "model_artifact_variant": "f16",
            "model_artifact_sha256": f"demo-artifact-{model_scale}",
            "model_artifact_source_revision": "demo-revision",
            "git_commit": "demo000",
            "git_dirty": False,
            "slurm_job_id": f"demo-{run_id}",
            "hostname": "preview-node",
            "dtype": "bfloat16",
            "quantization": None,
            "tensor_parallel_size": 1,
            "seed": 42,
            "input_length": input_length,
            "output_length": output_length,
            "max_model_len": 8192,
            "generation_config": "controlled",
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "number_of_requests": number_of_requests,
            "successful_requests": number_of_requests * DEMO_REPETITIONS,
            "failed_requests": 0,
            "actual_input_tokens": input_length * number_of_requests * DEMO_REPETITIONS,
            "actual_output_tokens": output_length * number_of_requests * DEMO_REPETITIONS,
            "request_rate": request_rate,
            "maximum_concurrency": maximum_concurrency,
            "warmup_runs": 1,
            "repetitions": DEMO_REPETITIONS,
            "telemetry_interval_ms": 500,
            "measured_repetitions": DEMO_REPETITIONS,
            "failed_repetitions": 0,
            "batch_size": maximum_concurrency or 8,
        }
    )
    return summary


def _performance_metrics(
    *,
    benchmark_type: str,
    output_throughput: float,
    input_length: int,
    output_length: int,
    request_count: int,
    energy_per_token: float,
) -> dict[str, float]:
    request_throughput = output_throughput / output_length
    duration = request_count / request_throughput
    serving_factor = 1.0 if benchmark_type == "serving" else 0.72
    return {
        "duration_seconds": duration,
        "model_load_time_seconds": 4.8 + input_length / 900,
        "request_throughput_requests_per_second": request_throughput,
        "input_throughput_tokens_per_second": request_throughput * input_length,
        "output_throughput_tokens_per_second": output_throughput,
        "total_throughput_tokens_per_second": request_throughput
        * (input_length + output_length),
        "prefill_throughput_tokens_per_second": output_throughput * 2.35,
        "decode_throughput_tokens_per_second": output_throughput,
        "mean_ttft_ms": (34 + input_length * 0.095) / serving_factor,
        "median_ttft_ms": (31 + input_length * 0.088) / serving_factor,
        "p95_ttft_ms": (52 + input_length * 0.14) / serving_factor,
        "p99_ttft_ms": (67 + input_length * 0.18) / serving_factor,
        "mean_tpot_ms": 1000 / max(output_throughput / 12, 1),
        "median_tpot_ms": 930 / max(output_throughput / 12, 1),
        "p95_tpot_ms": 1280 / max(output_throughput / 12, 1),
        "p99_tpot_ms": 1460 / max(output_throughput / 12, 1),
        "mean_itl_ms": 970 / max(output_throughput / 12, 1),
        "median_itl_ms": 910 / max(output_throughput / 12, 1),
        "p95_itl_ms": 1240 / max(output_throughput / 12, 1),
        "p99_itl_ms": 1420 / max(output_throughput / 12, 1),
        "mean_e2e_latency_ms": (
            (34 + input_length * 0.095) / serving_factor
            + output_length * 1000 / max(output_throughput / 12, 1)
        ),
        "p95_e2e_latency_ms": (
            (52 + input_length * 0.14) / serving_factor
            + output_length * 1280 / max(output_throughput / 12, 1)
        ),
        "energy_per_output_token_joules": energy_per_token,
        "energy_per_request_joules": energy_per_token * output_length,
        "energy_joules": energy_per_token
        * output_length
        * request_count
        * DEMO_REPETITIONS,
    }


def _gpu_summary(
    *,
    run_id: str,
    timestamp: datetime,
    benchmark_type: str,
    model_id: str,
    model_scale: str,
    scale_factor: float,
    input_length: int,
    output_length: int,
    backend: str = "vllm",
) -> dict[str, Any]:
    request_count = 96 if benchmark_type == "offline" else 64
    concurrency = None if benchmark_type == "offline" else 16
    request_rate = None if benchmark_type == "offline" else 12.0
    summary = _shared_identity(
        run_id=run_id,
        timestamp=timestamp,
        benchmark_type=benchmark_type,
        model_id=model_id,
        model_scale=model_scale,
        backend=backend,
        input_length=input_length,
        output_length=output_length,
        number_of_requests=request_count,
        request_rate=request_rate,
        maximum_concurrency=concurrency,
    )
    output_throughput = 4200 * scale_factor * (256 / input_length) ** 0.09
    summary.update(
        _performance_metrics(
            benchmark_type=benchmark_type,
            output_throughput=output_throughput,
            input_length=input_length,
            output_length=output_length,
            request_count=request_count,
            energy_per_token=0.061 / max(scale_factor, 0.25),
        )
    )
    summary.update(
        {
            "hardware_type": "gpu",
            "accelerator_name": "NVIDIA H100 80GB HBM3",
            "accelerator_count": 1,
            "accelerator_visibility": "0",
            "cpu_model": "AMD EPYC 9654",
            "socket_count": 2,
            "numa_node_count": 8,
            "memory_type": "HBM3",
            "memory_mode": "device",
            "memory_capacity_gib": 80.0,
            "memory_bandwidth_instrument": "NVIDIA DCGM preview",
            "memory_bandwidth_scope": "single accelerator",
            "power_instrument": "nvidia-smi sampled power",
            "power_scope": "single accelerator",
            "gpu_memory_utilization": 0.9,
            "peak_gpu_memory_mib": 10_800 + 31_500 * (1 - scale_factor),
            "average_gpu_utilization_percent": 91.5 - input_length / 512,
            "average_gpu_power_watts": 418.0 - 24 * scale_factor,
            "measured_memory_bandwidth_gbps": 2_180.0 * (0.92 + 0.08 * scale_factor),
            "memory_bandwidth_utilization_percent": 65.0 + 12 * (1 - scale_factor),
            "software_versions": {
                "python": "3.12.4",
                "llm_bench": "preview",
                "vllm": "0.11.0" if backend == "vllm" else None,
                "llamacpp": "b6200" if backend == "llamacpp" else None,
                "torch": "2.8.0",
                "cuda_runtime": "12.9",
                "nvidia_driver": "580.0",
            },
        }
    )
    return summary


def _cpu_summary(
    *,
    run_id: str,
    timestamp: datetime,
    benchmark_type: str,
    model_id: str,
    model_scale: str,
    scale_factor: float,
    input_length: int,
    output_length: int,
    memory_type: str,
) -> dict[str, Any]:
    request_count = 48 if benchmark_type == "offline" else 32
    concurrency = None if benchmark_type == "offline" else 8
    request_rate = None if benchmark_type == "offline" else 4.0
    summary = _shared_identity(
        run_id=run_id,
        timestamp=timestamp,
        benchmark_type=benchmark_type,
        model_id=model_id,
        model_scale=model_scale,
        backend="llamacpp",
        input_length=input_length,
        output_length=output_length,
        number_of_requests=request_count,
        request_rate=request_rate,
        maximum_concurrency=concurrency,
    )
    uses_hbm = "HBM" in memory_type
    memory_factor = (
        1.36
        + 0.24 * (1 - scale_factor)
        + (0.10 if benchmark_type == "serving" else 0.0)
        if uses_hbm
        else 1.0
    )
    output_throughput = 620 * scale_factor * memory_factor * (256 / input_length) ** 0.13
    summary.update(
        _performance_metrics(
            benchmark_type=benchmark_type,
            output_throughput=output_throughput,
            input_length=input_length,
            output_length=output_length,
            request_count=request_count,
            energy_per_token=(0.19 if uses_hbm else 0.26) / max(scale_factor, 0.2),
        )
    )
    bandwidth = 810.0 if uses_hbm else 225.0
    summary.update(
        {
            "hardware_type": "cpu",
            "accelerator_name": None,
            "accelerator_count": 0,
            "accelerator_visibility": None,
            "cpu_model": "Intel Xeon CPU Max 9480",
            "socket_count": 2,
            "numa_node_count": 8,
            "memory_type": memory_type,
            "memory_mode": "flat" if uses_hbm else "DDR-only",
            "memory_capacity_gib": 128.0 if uses_hbm else 512.0,
            "thread_count": 112,
            "process_count": 2,
            "thread_affinity": "compact, 56 threads/socket",
            "numa_policy": "local",
            "memory_binding": "hbm" if uses_hbm else "ddr",
            "cpu_isa": "AVX-512 + AMX-BF16",
            "memory_bandwidth_instrument": "Intel PCM preview",
            "memory_bandwidth_scope": "two-socket node",
            "power_instrument": "RAPL package + DRAM preview",
            "power_scope": "two CPU packages",
            "gpu_memory_utilization": None,
            "peak_cpu_memory_mib": 9_600 + 28_800 * (1 - scale_factor),
            "average_cpu_utilization_percent": 88.0 + 4 * (1 - scale_factor),
            "average_cpu_power_watts": 620.0 if uses_hbm else 575.0,
            "measured_memory_bandwidth_gbps": bandwidth * (0.91 - 0.05 * scale_factor),
            "memory_bandwidth_utilization_percent": 72.0 if uses_hbm else 84.0,
            "software_versions": {
                "python": "3.12.4",
                "llm_bench": "preview",
                "vllm": "not-applicable",
                "torch": "2.8.0",
                "cuda_runtime": "not-applicable",
                "nvidia_driver": "not-applicable",
                "llama_cpp": "b6200",
            },
        }
    )
    return summary


def _measurement_document(summary: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        spec.field: float(value)
        for spec in METRIC_SPECS
        if (value := _finite_metric(summary.get(spec.field))) is not None
    }
    specs = {spec.field: spec for spec in METRIC_SPECS}
    records = []
    mean_factors = (0.975, 1.012, 0.992, 1.018, 1.003)
    maximum_factors = (0.94, 0.97, 0.955, 1.0, 0.985)
    for repetition, (mean_factor, maximum_factor) in enumerate(
        zip(mean_factors, maximum_factors, strict=True), start=1
    ):
        metrics = {}
        for field, value in values.items():
            aggregation = specs[field].aggregation
            if aggregation == "sum":
                metrics[field] = value / DEMO_REPETITIONS * mean_factor
            elif aggregation == "maximum":
                metrics[field] = value * maximum_factor
            else:
                metrics[field] = value * mean_factor
        metrics.update(
            {
                "successful_requests": summary.get("number_of_requests"),
                "failed_requests": 0,
                "actual_input_tokens": (
                    int(summary.get("number_of_requests") or 0)
                    * int(summary.get("input_length") or 0)
                ),
                "actual_output_tokens": (
                    int(summary.get("number_of_requests") or 0)
                    * int(summary.get("output_length") or 0)
                ),
                "model_precision": summary.get("model_precision"),
            }
        )
        records.append(
            {
                "repetition": repetition,
                "status": "completed",
                "raw_output_file": None,
                "telemetry_file": f"in-memory-telemetry-{repetition}",
                "metrics": metrics,
                "warnings": [],
            }
        )
    return {
        "schema_version": "1.0",
        "run_id": summary.get("run_id"),
        "expected_repetitions": DEMO_REPETITIONS,
        "source": "simulated-preview",
        "records": records,
        "warnings": [
            "Repetition values are deterministic simulated preview data, not measurements."
        ],
    }


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _telemetry_series(summary: Mapping[str, Any]) -> list[dict[str, float | None]]:
    samples: list[dict[str, float | None]] = []
    gpu = summary.get("hardware_type") == "gpu"
    for index in range(25):
        wave = math.sin(index / 3.2)
        ramp = min(index / 5, 1.0)
        sample: dict[str, float | None] = {
            "elapsed_seconds": float(index * 2),
            "average_gpu_utilization_percent": None,
            "total_gpu_memory_mib": None,
            "total_gpu_power_watts": None,
            "average_cpu_utilization_percent": None,
            "total_cpu_memory_mib": None,
            "total_cpu_power_watts": None,
            "memory_bandwidth_gbps": None,
        }
        if gpu:
            sample.update(
                {
                    "average_gpu_utilization_percent": max(
                        0.0,
                        float(summary["average_gpu_utilization_percent"]) * ramp + wave * 2.0,
                    ),
                    "total_gpu_memory_mib": float(summary["peak_gpu_memory_mib"])
                    * (0.65 + 0.35 * ramp),
                    "total_gpu_power_watts": float(summary["average_gpu_power_watts"])
                    * (0.72 + 0.28 * ramp)
                    + wave * 6,
                    "memory_bandwidth_gbps": float(summary["measured_memory_bandwidth_gbps"])
                    * (0.7 + 0.3 * ramp)
                    + wave * 22,
                }
            )
        else:
            sample.update(
                {
                    "average_cpu_utilization_percent": max(
                        0.0,
                        float(summary["average_cpu_utilization_percent"]) * ramp + wave * 1.5,
                    ),
                    "total_cpu_memory_mib": float(summary["peak_cpu_memory_mib"])
                    * (0.58 + 0.42 * ramp),
                    "total_cpu_power_watts": float(summary["average_cpu_power_watts"])
                    * (0.76 + 0.24 * ramp)
                    + wave * 9,
                    "memory_bandwidth_gbps": float(summary["measured_memory_bandwidth_gbps"])
                    * (0.68 + 0.32 * ramp)
                    + wave * 8,
                }
            )
        samples.append(sample)
    return samples


def simulated_dataset() -> DashboardDataset:
    """Build a deterministic GPU plus CPU-DDR/HBM preview campaign."""

    models = (
        ("meta-llama/Llama-3.2-1B-Instruct", "1B", 1.0),
        ("meta-llama/Llama-3.1-8B-Instruct", "8B", 0.31),
    )
    start = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)
    summaries: list[dict[str, Any]] = []
    counter = 0

    for model_id, model_scale, scale_factor in models:
        for benchmark_type in ("offline", "serving"):
            for input_length, output_length in ((256, 128), (1024, 256)):
                counter += 1
                summaries.append(
                    _gpu_summary(
                        run_id=f"demo-gpu-{model_scale.lower()}-{benchmark_type}-{input_length}",
                        timestamp=start + timedelta(hours=counter),
                        benchmark_type=benchmark_type,
                        model_id=model_id,
                        model_scale=model_scale,
                        scale_factor=scale_factor,
                        input_length=input_length,
                        output_length=output_length,
                    )
                )

    counter += 1
    llama_gpu = _gpu_summary(
        run_id="demo-gpu-llamacpp-1b-serving-256",
        timestamp=start + timedelta(hours=counter),
        benchmark_type="serving",
        model_id=models[0][0],
        model_scale=models[0][1],
        scale_factor=models[0][2] * 0.78,
        input_length=256,
        output_length=128,
        backend="llamacpp",
    )
    llama_gpu.update(
        {
            "backend_profile": "llamacpp_cuda",
            "gpu_layers": "all",
            "thread_count": 8,
            "parallel_slots": 16,
        }
    )
    summaries.append(llama_gpu)

    for model_id, model_scale, scale_factor in models:
        for benchmark_type, input_length, output_length in (
            ("offline", 256, 128),
            ("serving", 512, 128),
        ):
            for memory_type in ("DDR5", "HBM2e"):
                counter += 1
                summaries.append(
                    _cpu_summary(
                        run_id=(
                            f"demo-cpu-{memory_type.lower()}-{model_scale.lower()}-"
                            f"{benchmark_type}"
                        ),
                        timestamp=start + timedelta(hours=counter),
                        benchmark_type=benchmark_type,
                        model_id=model_id,
                        model_scale=model_scale,
                        scale_factor=scale_factor,
                        input_length=input_length,
                        output_length=output_length,
                        memory_type=memory_type,
                    )
                )

    failed = _cpu_summary(
        run_id="demo-cpu-hbm-placement-validation",
        timestamp=start + timedelta(minutes=30),
        benchmark_type="smoke",
        model_id=models[1][0],
        model_scale=models[1][1],
        scale_factor=models[1][2],
        input_length=64,
        output_length=16,
        memory_type="HBM2e",
    )
    failed.update(
        {
            "status": "failed",
            "error": "Requested HBM binding was not observed.",
            "measured_repetitions": 0,
            "failed_repetitions": 1,
            "warnings": [
                "Simulated preview data: this failure demonstrates placement validation.",
                "The run failed because effective memory placement did not match the request.",
            ],
        }
    )
    summaries.append(failed)

    entries: list[CatalogEntry] = []
    measurements: dict[str, Mapping[str, Any]] = {}
    telemetry: dict[str, Sequence[Mapping[str, float | None]]] = {}
    for summary in summaries:
        run_id = str(summary["run_id"])
        run_directory = Path("/__llm_bench_simulated__") / run_id
        entry = CatalogEntry(run_directory / "summary.json", run_directory, summary)
        entries.append(entry)
        if summary["status"] == "completed":
            measurements[entry.key] = _measurement_document(summary)
            telemetry[entry.key] = _telemetry_series(summary)

    entries.sort(key=lambda entry: str(entry.summary.get("timestamp") or ""), reverse=True)
    catalog = ResultCatalog(Path("/__llm_bench_simulated__"), tuple(entries), ())
    return DashboardDataset(
        catalog,
        measurements,
        telemetry,
        frozenset(entry.key for entry in entries),
    )
