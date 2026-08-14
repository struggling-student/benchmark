"""Read-only discovery and filtering primitives for benchmark result roots."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .results import SCHEMA_VERSION, ResultError, load_summary

SUPPORTED_SUMMARY_SCHEMAS = {"1.0", "1.1", SCHEMA_VERSION}


@dataclass(frozen=True)
class CatalogEntry:
    """One valid summary, uniquely identified by its filesystem location."""

    summary_path: Path
    run_directory: Path
    summary: dict[str, Any]

    @property
    def key(self) -> str:
        return str(self.summary_path.resolve())


@dataclass(frozen=True)
class CatalogDiagnostic:
    path: Path
    message: str


@dataclass(frozen=True)
class ResultCatalog:
    root: Path
    entries: tuple[CatalogEntry, ...]
    diagnostics: tuple[CatalogDiagnostic, ...]


def discover_results(results_root: str | Path) -> ResultCatalog:
    """Recursively discover supported summaries without failing the entire catalog."""

    root = Path(results_root).expanduser().resolve()
    diagnostics: list[CatalogDiagnostic] = []
    entries: list[CatalogEntry] = []
    if not root.is_dir():
        return ResultCatalog(
            root,
            (),
            (CatalogDiagnostic(root, "Results root does not exist or is not a directory."),),
        )

    for summary_path in sorted(root.rglob("summary.json")):
        try:
            summary = load_summary(summary_path)
        except ResultError as exc:
            diagnostics.append(CatalogDiagnostic(summary_path, str(exc)))
            continue
        version = summary.get("schema_version")
        if version not in SUPPORTED_SUMMARY_SCHEMAS:
            diagnostics.append(
                CatalogDiagnostic(
                    summary_path,
                    f"Unsupported summary schema {version!r}; expected {SCHEMA_VERSION!r}.",
                )
            )
            continue
        if version != SCHEMA_VERSION:
            diagnostics.append(
                CatalogDiagnostic(
                    summary_path,
                    f"Legacy summary schema {version!r} was accepted; fields introduced in "
                    f"{SCHEMA_VERSION!r} may be unavailable and comparisons may be partial.",
                )
            )
        entries.append(CatalogEntry(summary_path, summary_path.parent, summary))

    counts = Counter(str(entry.summary.get("run_id")) for entry in entries)
    for entry in entries:
        run_id = str(entry.summary.get("run_id"))
        if counts[run_id] > 1:
            diagnostics.append(
                CatalogDiagnostic(
                    entry.summary_path,
                    f"Duplicate run_id {run_id!r}; filesystem paths remain distinct identities.",
                )
            )
    entries.sort(key=lambda entry: str(entry.summary.get("timestamp") or ""), reverse=True)
    return ResultCatalog(root, tuple(entries), tuple(diagnostics))


def workload_label(summary: dict[str, Any]) -> str:
    """Build a compact searchable workload description."""

    parts = [
        f"input={summary.get('input_length')}",
        f"output={summary.get('output_length')}",
        f"requests={summary.get('number_of_requests')}",
    ]
    if summary.get("request_rate") is not None:
        parts.append(f"rate={summary['request_rate']}")
    if summary.get("maximum_concurrency") is not None:
        parts.append(f"concurrency={summary['maximum_concurrency']}")
    if summary.get("batch_size") is not None:
        parts.append(f"batch={summary['batch_size']}")
    return ", ".join(parts)


def catalog_row(entry: CatalogEntry) -> dict[str, Any]:
    """Flatten the catalog dimensions used by tables and filters."""

    summary = entry.summary
    timestamp = str(summary.get("timestamp") or "")
    memory_type = summary.get("memory_type") or summary.get("memory_system")
    hardware_type = str(summary.get("hardware_type") or "unknown").lower()
    hardware = (
        summary.get("accelerator_name")
        if hardware_type == "gpu"
        else summary.get("cpu_model")
    )
    if hardware_type == "gpu":
        platform = "GPU"
    elif hardware_type == "hybrid":
        platform = "CPU + GPU"
    elif hardware_type == "cpu" and "hbm" in str(memory_type or "").lower():
        platform = "CPU · HBM"
    elif hardware_type == "cpu" and "ddr" in str(memory_type or "").lower():
        platform = "CPU · DDR"
    elif hardware_type == "cpu":
        platform = "CPU"
    else:
        platform = hardware_type.upper()
    return {
        "path": entry.key,
        "run_directory": str(entry.run_directory.resolve()),
        "run_id": summary.get("run_id"),
        "timestamp": timestamp,
        "date": timestamp[:10] if len(timestamp) >= 10 else None,
        "status": summary.get("status"),
        "benchmark_type": summary.get("benchmark_type"),
        "backend": summary.get("backend"),
        "backend_profile": summary.get("backend_profile"),
        "provider": summary.get("execution_provider"),
        "artifact_variant": summary.get("model_artifact_variant"),
        "measurement_method": summary.get("measurement_method"),
        "model_id": summary.get("model_id"),
        "model_scale": summary.get("model_parameter_scale"),
        "hardware_type": summary.get("hardware_type"),
        "platform": platform,
        "hardware": hardware,
        "memory_type": memory_type,
        "memory_mode": summary.get("memory_mode"),
        "precision": summary.get("model_precision") or summary.get("dtype"),
        "source": (
            "Simulated"
            if summary.get("is_simulated") or summary.get("data_source") == "simulated"
            else "Measured"
        ),
        "workload": workload_label(summary),
        "input_length": summary.get("input_length"),
        "output_length": summary.get("output_length"),
        "number_of_requests": summary.get("number_of_requests"),
        "request_rate": summary.get("request_rate"),
        "maximum_concurrency": summary.get("maximum_concurrency"),
        "batch_size": summary.get("batch_size"),
        "thread_count": summary.get("thread_count"),
        "numa_policy": summary.get("numa_policy"),
        "memory_binding": summary.get("memory_binding"),
    }
