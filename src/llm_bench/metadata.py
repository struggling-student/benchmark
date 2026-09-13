"""Collect portable run metadata without assuming that an accelerator exists."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import __version__
from .config import ExperimentConfig
from .hardware import inspect_hardware, lscpu_values, resolve_memory_binding
from .providers import llama_cpp_runtime_variable, vllm_runtime_variable


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
    return completed.stdout.strip() or completed.stderr.strip() or None


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _memory_capacity_gib() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except (ImportError, OSError):
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

    lscpu = lscpu_values()
    hardware = inspect_hardware(config)
    memory_binding_resolved = resolve_memory_binding(config, hardware)
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
    selected_type = benchmark_type or config.resolved_benchmark_type
    runtime_environment_variable: str | None = None
    native_binary_path: str | None = None
    if config.backend == "vllm":
        backend_version = _package_version("vllm")
        binary_variable = vllm_runtime_variable(config, "bin")
        vllm_executable = os.environ.get(binary_variable) or "vllm"
        runtime_environment_variable = binary_variable
        if config.provider == "native":
            native_binary_path = shutil.which(vllm_executable)
        native_binary_version = (
            _run((vllm_executable, "--version")) if config.provider == "native" else None
        )
    else:
        llama_executable = "llama-bench" if selected_type == "offline" else "llama-server"
        binary_variable = llama_cpp_runtime_variable(config, "bin_dir")
        runtime_environment_variable = binary_variable
        if config.provider == "native" and os.environ.get(binary_variable):
            llama_executable = str(
                Path(os.environ[binary_variable]).expanduser() / llama_executable
            )
        if config.provider == "native":
            native_binary_path = shutil.which(llama_executable)
        native_binary_version = (
            _run((llama_executable, "--version")) if config.provider == "native" else None
        )
        backend_version = native_binary_version
    # Distribution spelling is fixed for these common packages, independent of imports.
    software_versions = {
        "python": platform.python_version(),
        "llm_bench": __version__,
        "vllm": _package_version("vllm"),
        "llamacpp": backend_version if config.backend == "llamacpp" else None,
        "torch": _package_version("torch"),
        "pyyaml": _package_version("PyYAML"),
        "nvidia_driver": driver_version,
        "cuda_runtime": cuda_runtime,
    }
    accelerator_names = sorted({device["name"] for device in accelerators})
    if config.hardware_type == "cpu":
        accelerators = []
        accelerator_names = []
        visibility_warning = None
        nvidia_devices_unscoped = False
        accelerator_visibility = None
    container_image = None
    if config.provider == "docker":
        variable = (
            vllm_runtime_variable(config, "docker_image")
            if config.backend == "vllm"
            else llama_cpp_runtime_variable(config, "docker_image")
        )
        runtime_environment_variable = variable
        container_image = os.environ.get(variable)
    elif config.provider == "apptainer":
        variable = (
            vllm_runtime_variable(config, "apptainer_image")
            if config.backend == "vllm"
            else llama_cpp_runtime_variable(config, "apptainer_image")
        )
        runtime_environment_variable = variable
        container_image = os.environ.get(variable)
    container_digest = (
        "sha256:" + container_image.split("@sha256:", 1)[1]
        if container_image and "@sha256:" in container_image
        else _file_sha256(Path(container_image).expanduser())
        if container_image and Path(container_image).expanduser().is_file()
        else None
    )
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
    rapl_paths = list(Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"))
    rapl_available = any(os.access(path, os.R_OK) for path in rapl_paths)
    gpu_power_available = config.hardware_type in {"gpu", "hybrid"} and bool(accelerators)
    power_instruments = [
        name
        for enabled, name in (
            (rapl_available, "linux_rapl"),
            (gpu_power_available, "nvidia_smi"),
        )
        if enabled
    ]
    return {
        "schema_version": "1.0",
        "run_id": run_id or make_run_id(config.experiment_name),
        "timestamp": utc_timestamp(),
        "experiment_name": config.experiment_name,
        "model_key": config.model_key,
        "workload_key": config.workload_key,
        "benchmark_type": selected_type,
        "backend": config.backend,
        "backend_version": backend_version,
        "backend_profile": config.profile_name,
        "execution_provider": config.provider,
        "container_image": container_image,
        "container_image_digest": container_digest,
        "container_id": None,
        "native_binary_version": native_binary_version,
        "native_binary_path": native_binary_path,
        "runtime_environment_variable": runtime_environment_variable,
        "measurement_method": (
            "backend_native" if selected_type == "offline" else "shared_openai_streaming"
        ),
        "measurement_scope": (
            "backend_native_no_common_boundary"
            if selected_type == "offline"
            else "client_observed_end_to_end"
        ),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_revision_policy": config.model_revision_policy,
        "resolved_model_revision": resolved_model_revision,
        "tokenizer_id": config.tokenizer,
        "resolved_tokenizer_revision": resolved_tokenizer_revision,
        "model_parameter_scale": config.model_parameter_scale,
        "git_commit": git_commit(repository),
        "git_dirty": git_dirty(repository),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "hardware_type": config.hardware_type,
        "accelerator_name": ", ".join(accelerator_names) if accelerator_names else None,
        "accelerator_count": None if nvidia_devices_unscoped else len(accelerators),
        "accelerator_visibility": accelerator_visibility,
        "accelerators": accelerators,
        "cpu_model": lscpu.get("Model name") or platform.processor() or None,
        "socket_count": _optional_int(lscpu.get("Socket(s)")),
        "numa_node_count": _optional_int(lscpu.get("NUMA node(s)")),
        "thread_count": config.thread_count,
        "thread_count_batch": config.thread_count_batch,
        "cpu_mask": config.cpu_mask,
        "numa_policy": config.numa_policy,
        "memory_binding": config.memory_binding,
        "memory_binding_resolved": memory_binding_resolved,
        "memory_policy": config.memory_policy,
        "load_mode": config.load_mode,
        "vllm_cpu_kvcache_space_gib": config.vllm_cpu_kvcache_space_gib,
        "vllm_cpu_omp_threads_bind": config.vllm_cpu_omp_threads_bind,
        "vllm_cpu_num_reserved_cpu": config.vllm_cpu_num_reserved_cpu,
        "memory_type": config.memory_type,
        "memory_mode": config.memory_mode,
        "memory_mode_requested": hardware["memory_mode_requested"],
        "memory_mode_detected": hardware["memory_mode_detected"],
        "memory_mode_detection_method": hardware["memory_mode_detection_method"],
        "memory_mode_verified": hardware["memory_mode_verified"],
        "memory_capacity_gib": _memory_capacity_gib(),
        "numa_nodes": hardware["numa_nodes"],
        "hbm_numa_nodes": hardware["hbm_numa_nodes"],
        "ddr_numa_nodes": hardware["ddr_numa_nodes"],
        "thread_affinity": (
            config.vllm_cpu_omp_threads_bind
            if config.backend == "vllm" and config.hardware_type == "cpu"
            else config.cpu_mask
        ),
        "process_count": (
            config.tensor_parallel_size
            if config.backend == "vllm" and config.hardware_type == "cpu"
            else 1
        ),
        "cpu_isa": hardware["cpu_isa"],
        "cpu_isa_target": hardware["cpu_isa_target"],
        "cpu_features_required": hardware["cpu_features_required"],
        "cpu_features_detected": hardware["cpu_features_detected"],
        "cpu_features_missing": hardware["cpu_features_missing"],
        "cpu_isa_verified": hardware["cpu_isa_verified"],
        "gpu_layers": config.gpu_layers,
        "batch_size": config.batch_size,
        "ubatch_size": config.ubatch_size,
        "parallel_slots": config.parallel_slots,
        "telemetry_scope": "workload_process_and_visible_devices",
        "energy_scope": "available_cpu_package_plus_visible_gpu",
        "instrumentation_boundary": "measured_repetitions_only",
        "memory_bandwidth_instrument": None,
        "memory_bandwidth_scope": None,
        "power_instrument": ",".join(power_instruments) if power_instruments else None,
        "power_scope": (
            "available_cpu_packages_and_visible_gpus" if power_instruments else None
        ),
        "configuration_hashes": {},
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
