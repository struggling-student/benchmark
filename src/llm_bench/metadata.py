"""Collect portable run metadata without assuming that an accelerator exists."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import __version__
from .config import ExperimentConfig


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSON records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(experiment_name: str) -> str:
    """Create a readable ID that remains unique for concurrent Slurm jobs."""

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", experiment_name).strip("-_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{slug or 'run'}-{uuid4().hex[:8]}"


def _run(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _lscpu_values() -> dict[str, str]:
    output = _run(("lscpu", "--json"))
    if output:
        try:
            rows = json.loads(output).get("lscpu", [])
            return {
                str(row["field"]).rstrip(":"): str(row["data"]).strip()
                for row in rows
                if row.get("field") and row.get("data") is not None
            }
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            pass
    plain = _run(("lscpu",))
    if not plain:
        return {}
    values: dict[str, str] = {}
    for line in plain.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _visible_devices(
    devices: list[dict[str, Any]], visibility: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    if visibility is None or visibility.strip().lower() == "all":
        return devices, None
    if visibility.strip().lower() in {"", "-1", "none", "void"}:
        return [], None
    selectors = [item.strip() for item in visibility.split(",") if item.strip()]
    selected = [
        device
        for device in devices
        if any(
            selector == device["index"]
            or device["uuid"] == selector
            or device["uuid"].startswith(selector)
            for selector in selectors
        )
    ]
    warning = None
    if selectors and not selected:
        warning = (
            "GPU visibility was configured, but its indices/UUIDs could not be matched to "
            "nvidia-smi output; accelerator_count may be unavailable."
        )
    return selected, warning


def _accelerators() -> tuple[list[dict[str, Any]], str | None, str | None, str | None]:
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visibility is None:
        visibility = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    query = "index,uuid,name,driver_version"
    output = _run(
        (
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        )
    )
    if not output:
        return [], None, visibility, None
    devices: list[dict[str, Any]] = []
    driver: str | None = None
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",", 3)]
        if len(values) != 4:
            continue
        index, uuid, name, device_driver = values
        devices.append({"index": index, "uuid": uuid, "name": name})
        driver = driver or device_driver
    visible, warning = _visible_devices(devices, visibility)
    return visible, driver, visibility, warning


def _torch_cuda() -> tuple[list[dict[str, Any]], str | None]:
    """Discover allocated CUDA devices when nvidia-smi is absent or restricted."""

    try:
        torch = importlib.import_module("torch")
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        if not torch.cuda.is_available():
            return [], cuda_runtime
        devices = [
            {
                "index": str(index),
                "uuid": None,
                "name": str(torch.cuda.get_device_name(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
        return devices, cuda_runtime
    except (ImportError, OSError, RuntimeError, AttributeError):
        return [], None


def git_commit(repository: str | Path | None = None) -> str | None:
    command = ["git"]
    if repository is not None:
        command.extend(("-C", str(repository)))
    command.extend(("rev-parse", "HEAD"))
    return _run(command)


def git_dirty(repository: str | Path | None = None) -> bool | None:
    """Return whether tracked or untracked files differ from the recorded commit."""

    command = ["git"]
    if repository is not None:
        command.extend(("-C", str(repository)))
    command.extend(("status", "--porcelain"))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout)


_COMMIT_HASH = re.compile(r"^[0-9a-fA-F]{40,64}$")


def _huggingface_cache_root() -> Path:
    try:
        constants = importlib.import_module("huggingface_hub.constants")
        configured = getattr(constants, "HF_HUB_CACHE", None)
        if configured:
            return Path(str(configured)).expanduser()
    except (ImportError, AttributeError):
        pass
    if configured := os.environ.get("HF_HUB_CACHE"):
        return Path(configured).expanduser()
    if configured := os.environ.get("HF_HOME"):
        return Path(configured).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_revision(model_id: str, revision: str | None) -> str | None:
    """Resolve a local HF cache ref without network access or authentication."""

    selected = revision or "main"
    if _COMMIT_HASH.fullmatch(selected):
        return selected.lower()
    repository = _huggingface_cache_root() / f"models--{model_id.replace('/', '--')}"
    refs = (repository / "refs").resolve()
    candidate = (refs / selected).resolve()
    try:
        candidate.relative_to(refs)
    except ValueError:
        return None
    try:
        resolved = candidate.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return resolved.lower() if _COMMIT_HASH.fullmatch(resolved) else None


def collect_metadata(
    config: ExperimentConfig,
    *,
    benchmark_type: str | None = None,
    run_id: str | None = None,
    repository: str | Path | None = None,
) -> dict[str, Any]:
    """Collect identity, software, scheduler, and hardware metadata for one run."""

    lscpu = _lscpu_values()
    accelerators, driver_version, accelerator_visibility, visibility_warning = _accelerators()
    nvidia_devices_unscoped = bool(accelerators) and (
        accelerator_visibility is None or accelerator_visibility.strip().lower() == "all"
    )
    torch_accelerators, cuda_runtime = _torch_cuda()
    if torch_accelerators:
        accelerators = torch_accelerators
        visibility_warning = None
        nvidia_devices_unscoped = False
    elif nvidia_devices_unscoped:
        accelerators = []
        visibility_warning = (
            "nvidia-smi reported node GPUs, but CUDA visibility was unset/unscoped and "
            "PyTorch did not confirm an allocated CUDA device; GPU identity/count are "
            "left unavailable rather than recording every node GPU."
        )
    backend_version = _package_version(config.backend)
    # Distribution spelling is fixed for these common packages, independent of imports.
    software_versions = {
        "python": platform.python_version(),
        "llm_bench": __version__,
        "vllm": _package_version("vllm"),
        "torch": _package_version("torch"),
        "pyyaml": _package_version("PyYAML"),
        "nvidia_driver": driver_version,
        "cuda_runtime": cuda_runtime,
    }
    accelerator_names = sorted({device["name"] for device in accelerators})
    selected_type = benchmark_type or config.resolved_benchmark_type
    resolved_model_revision = _cached_revision(config.model_id, config.model_revision)
    if config.tokenizer == config.model_id:
        resolved_tokenizer_revision = resolved_model_revision
    else:
        resolved_tokenizer_revision = _cached_revision(config.tokenizer, None)
    metadata_warnings = [visibility_warning] if visibility_warning else []
    if resolved_model_revision is None:
        metadata_warnings.append(
            "Resolved model snapshot revision was not available from the local Hugging "
            "Face cache; pin model_revision before a formal comparison."
        )
    if resolved_tokenizer_revision is None:
        metadata_warnings.append(
            "Resolved tokenizer snapshot revision was not available from the local "
            "Hugging Face cache; pin/cache its revision before a formal comparison."
        )
    return {
        "schema_version": "1.0",
        "run_id": run_id or make_run_id(config.experiment_name),
        "timestamp": utc_timestamp(),
        "experiment_name": config.experiment_name,
        "benchmark_type": selected_type,
        "backend": config.backend,
        "backend_version": backend_version,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "resolved_model_revision": resolved_model_revision,
        "tokenizer_id": config.tokenizer,
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        "model_parameter_scale": config.model_parameter_scale,
        "git_commit": git_commit(repository),
        "git_dirty": git_dirty(repository),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "hardware_type": (
            "gpu" if accelerators else "unknown" if nvidia_devices_unscoped else "cpu"
        ),
        "accelerator_name": ", ".join(accelerator_names) if accelerator_names else None,
        "accelerator_count": None if nvidia_devices_unscoped else len(accelerators),
        "accelerator_visibility": accelerator_visibility,
        "accelerators": accelerators,
        "cpu_model": lscpu.get("Model name") or platform.processor() or None,
        "socket_count": _optional_int(lscpu.get("Socket(s)")),
        "numa_node_count": _optional_int(lscpu.get("NUMA node(s)")),
        "software_versions": software_versions,
        "warnings": metadata_warnings,
    }


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Serialize metadata as deterministic, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
