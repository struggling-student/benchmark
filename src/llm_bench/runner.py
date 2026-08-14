"""Backend-neutral benchmark lifecycle and benchmark-runner registry."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from .api_client import run_api_benchmark
from .backends import BackendAdapter, endpoint, get_backend
from .config import ConfigurationError, ExperimentConfig
from .measurements import create_measurements, write_measurements
from .metadata import collect_metadata, write_metadata
from .preparation import load_artifact_manifest, sha256_file
from .providers import CommandProvider, ProviderContext, get_provider
from .results import create_summary, normalize_summary, write_summary
from .telemetry import TelemetrySession, telemetry_capabilities
from .workloads import (
    build_native_workload_manifest,
    build_workload_manifest,
    write_workload_manifest,
)


def _file_hash(path: Path) -> str:
    return sha256_file(path)


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _configure_cache_environment() -> None:
    cache = os.environ.get("MODEL_CACHE_DIR")
    if cache and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(Path(cache).expanduser().resolve())


def _artifact_provenance(config: ExperimentConfig) -> dict[str, Any] | None:
    manifest = load_artifact_manifest(config.artifact_path)
    if manifest is not None:
        return manifest
    if config.artifact_manifest_path:
        manifest_path = Path(config.artifact_manifest_path).expanduser()
    else:
        root = os.environ.get("MODEL_ARTIFACT_ROOT")
        if not root or not config.model_key:
            return None
        manifest_path = Path(root).expanduser() / config.model_key / "artifact-manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _new_run_directory(config: ExperimentConfig, results_root: Path) -> Path:
    now = datetime.now(timezone.utc)
    identifier = (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}-{config.experiment_name}-"
        f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
    )
    return results_root / now.strftime("%Y-%m-%d") / identifier


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _remove_container(provider_name: str, name: str) -> None:
    if provider_name != "docker":
        return
    try:
        subprocess.run(
            ["docker", "rm", "--force", name],
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _container_id(provider_name: str, name: str | None) -> str | None:
    if provider_name != "docker" or not name:
        return None
    for attempt in range(5):
        try:
            completed = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
        if attempt < 4:
            time.sleep(0.05)
    return None


@dataclass(slots=True)
class RunContext:
    config: ExperimentConfig
    run_directory: Path
    backend: BackendAdapter
    provider: CommandProvider
    provider_context: ProviderContext
    summary: dict[str, Any]


class BenchmarkRunner(Protocol):
    measurement_method: str
    measurement_scope: str

    def run(self, context: RunContext) -> tuple[list[Path], list[Path], list[str], str | None]: ...


class OfflineBenchmarkRunner:
    measurement_method = "backend_native"
    measurement_scope = "backend_native_no_common_boundary"

    def _execute(self, context: RunContext, output: Path, *, measured: bool) -> int:
        config = context.config
        command = context.backend.offline_command(config, output)
        wrapped = context.provider.wrap(command, config, context.provider_context)
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".log")
        stdout_target = output if config.backend == "llamacpp" else log_path
        diagnostic_path = (
            log_path if config.backend == "llamacpp" else output.with_suffix(".stderr.log")
        )
        with stdout_target.open("w", encoding="utf-8") as stdout, diagnostic_path.open(
            "w", encoding="utf-8"
        ) as log:
            process = subprocess.Popen(
                wrapped,
                stdout=stdout,
                stderr=log if config.backend == "llamacpp" else subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            context.summary["container_id"] = _container_id(
                config.provider, context.provider_context.container_name
            )
            telemetry = None
            if measured:
                telemetry = TelemetrySession(
                    output.with_name(output.stem + ".telemetry.csv"),
                    pid=process.pid,
                    interval_ms=config.telemetry_interval_ms,
                    provider=config.provider,
                    container_name=context.provider_context.container_name,
                    collect_gpu=config.hardware_type in {"gpu", "hybrid"},
                )
                telemetry.start()
            try:
                return process.wait()
            except KeyboardInterrupt:
                _stop_process(process)
                _remove_container(
                    config.provider, context.provider_context.container_name or ""
                )
                raise
            finally:
                if telemetry:
                    telemetry.stop()

    def run(self, context: RunContext) -> tuple[list[Path], list[Path], list[str], str | None]:
        config = context.config
        workload = build_native_workload_manifest(config)
        write_workload_manifest(context.run_directory / "workload_manifest.json", workload)
        context.summary["workload_manifest_sha256"] = workload["sha256"]
        warnings = [
            "Offline results use backend-native measurement boundaries; cross-backend ratios "
            "are intentionally disabled."
        ]
        if config.backend == "llamacpp":
            warnings.append(
                "llama-bench excludes tokenization and sampling and performs an internal "
                "discarded warm-up before its measured samples."
            )
            context.summary["measurement_method"] = "llamacpp_bench"
            context.summary["measurement_scope"] = (
                "model_forward_only_excludes_tokenization_and_sampling"
            )
            context.summary["backend_internal_warmup"] = (
                "llama-bench default warm-up before each native test"
            )
        else:
            context.summary["measurement_method"] = "vllm_bench_throughput"
            context.summary["measurement_scope"] = "vllm_bench_throughput_native"
        warmup_root = context.run_directory / "warmup"
        for index in range(config.warmup_runs):
            output = warmup_root / f"raw_backend_output.warmup-{index + 1:03d}.json"
            if self._execute(context, output, measured=False) != 0:
                return [], [], warnings, f"Offline warm-up {index + 1} failed"

        raw_paths: list[Path] = []
        telemetry_paths: list[Path] = []
        for index in range(config.repetitions):
            suffix = "" if index == 0 else f".repetition-{index + 1:03d}"
            output = context.run_directory / f"raw_backend_output{suffix}.json"
            return_code = self._execute(context, output, measured=True)
            telemetry = output.with_name(output.stem + ".telemetry.csv")
            if return_code != 0 or not output.is_file() or output.stat().st_size == 0:
                return raw_paths, telemetry_paths, warnings, (
                    f"Offline measured repetition {index + 1} failed with exit {return_code}"
                )
            raw_paths.append(output)
            if telemetry.is_file() and telemetry.stat().st_size > 0:
                telemetry_paths.append(telemetry)
        return raw_paths, telemetry_paths, warnings, None


class ApiBenchmarkRunner:
    measurement_method = "shared_openai_streaming"
    measurement_scope = "client_observed_end_to_end"

    def _wait_ready(self, context: RunContext, process: subprocess.Popen[Any]) -> float:
        host, port = endpoint(context.config)
        url = context.backend.health_url(host, port)
        started = time.monotonic()
        timeout = float(os.environ.get("LLM_BENCH_STARTUP_TIMEOUT_SECONDS", "600"))
        while time.monotonic() - started < timeout:
            if process.poll() is not None:
                raise ConfigurationError("backend server exited before becoming healthy")
            try:
                with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                    if 200 <= response.status < 300:
                        return time.monotonic() - started
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        raise ConfigurationError(f"backend server did not become healthy within {timeout:g}s")

    def _check_model_identity(self, config: ExperimentConfig) -> None:
        host, port = endpoint(config)
        url = f"http://{host}:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ConfigurationError("cannot verify backend model identity") from exc
        identifiers = {
            str(item.get("id"))
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }
        if config.model_id not in identifiers:
            raise ConfigurationError(
                f"backend reported models {sorted(identifiers)!r}, expected {config.model_id!r}"
            )

    def run(self, context: RunContext) -> tuple[list[Path], list[Path], list[str], str | None]:
        config = context.config
        host, port = endpoint(config)
        server_command = context.backend.server_command(config)
        wrapped = context.provider.wrap(
            server_command,
            config,
            context.provider_context,
            server=True,
        )
        server_log = (context.run_directory / "server.log").open("w", encoding="utf-8")
        server: subprocess.Popen[Any] | None = None
        try:
            server = subprocess.Popen(
                wrapped,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            context.summary["model_load_time_seconds"] = self._wait_ready(context, server)
            context.summary["container_id"] = _container_id(
                config.provider, context.provider_context.container_name
            )
            self._check_model_identity(config)
            workload = build_workload_manifest(config, local_files_only=True)
            write_workload_manifest(context.run_directory / "workload_manifest.json", workload)
            context.summary["workload_manifest_sha256"] = workload["sha256"]
            base_url = f"http://{host}:{port}"
            warmup_root = context.run_directory / "warmup"
            for index in range(config.warmup_runs):
                raw = asyncio.run(run_api_benchmark(config, workload, base_url=base_url))
                _json_write(
                    warmup_root / f"raw_harness_output.warmup-{index + 1:03d}.json",
                    raw,
                )
                if raw["failed_requests"]:
                    return [], [], list(raw.get("warnings", [])), (
                        f"API warm-up {index + 1} had failed requests"
                    )

            raw_paths: list[Path] = []
            telemetry_paths: list[Path] = []
            warnings: list[str] = []
            for index in range(config.repetitions):
                suffix = "" if index == 0 else f".repetition-{index + 1:03d}"
                output = context.run_directory / f"raw_harness_output{suffix}.json"
                telemetry_path = context.run_directory / f"resource_telemetry{suffix}.csv"
                telemetry = TelemetrySession(
                    telemetry_path,
                    pid=server.pid,
                    interval_ms=config.telemetry_interval_ms,
                    provider=config.provider,
                    container_name=context.provider_context.container_name,
                    collect_gpu=config.hardware_type in {"gpu", "hybrid"},
                )
                telemetry.start()
                try:
                    raw = asyncio.run(run_api_benchmark(config, workload, base_url=base_url))
                finally:
                    telemetry.stop()
                _json_write(output, raw)
                warnings.extend(str(item) for item in raw.get("warnings", []))
                raw_paths.append(output)
                if telemetry_path.is_file() and telemetry_path.stat().st_size > 0:
                    telemetry_paths.append(telemetry_path)
                if raw["failed_requests"]:
                    return raw_paths, telemetry_paths, warnings, (
                        f"API measured repetition {index + 1} had failed requests"
                    )
            return raw_paths, telemetry_paths, list(dict.fromkeys(warnings)), None
        except (ConfigurationError, OSError) as exc:
            return [], [], [], str(exc)
        finally:
            _stop_process(server)
            _remove_container(config.provider, context.provider_context.container_name or "")
            server_log.close()


RUNNER_REGISTRY: dict[str, type[BenchmarkRunner]] = {
    "smoke": ApiBenchmarkRunner,
    "offline": OfflineBenchmarkRunner,
    "serving": ApiBenchmarkRunner,
}


def validate_runtime(
    config: ExperimentConfig,
    *,
    require_artifact: bool = True,
) -> dict[str, Any]:
    _configure_cache_environment()
    backend = get_backend(config.backend)
    provider = get_provider(config.provider)
    checks = [
        *backend.validate(config, require_artifact=require_artifact),
        *provider.validate(config),
    ]
    dry_run_directory = (Path.cwd() / ".llm-bench-dry-run").resolve()
    provider_context = ProviderContext(
        run_directory=dry_run_directory,
        artifact_root=(
            Path(config.artifact_path).expanduser().resolve().parent.parent
            if config.artifact_path
            else None
        ),
        cache_root=(
            Path(os.environ["MODEL_CACHE_DIR"]).expanduser().resolve()
            if os.environ.get("MODEL_CACHE_DIR")
            else None
        ),
        container_name="llm-bench-dry-run" if config.provider == "docker" else None,
    )
    for label, root in (
        ("artifact mount", provider_context.artifact_root),
        ("model cache mount", provider_context.cache_root),
    ):
        if root is not None and not root.is_dir():
            raise ConfigurationError(f"{label} does not exist or is not a directory: {root}")
        if root is not None:
            checks.append(f"{label}: {root}")
    offline_command = None
    server_command = None
    if config.benchmark_type == "offline":
        offline_command = provider.wrap(
            backend.offline_command(config, dry_run_directory / "RAW_OUTPUT.json"),
            config,
            provider_context,
        )
    else:
        host, port = endpoint(config)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((host, port))
        except OSError as exc:
            raise ConfigurationError(
                f"cannot bind benchmark endpoint {host}:{port}: {exc}"
            ) from exc
        checks.append(f"endpoint is available: {host}:{port}")
        server_command = provider.wrap(
            backend.server_command(config), config, provider_context, server=True
        )
    return {
        "status": "valid",
        "experiment_name": config.experiment_name,
        "backend": config.backend,
        "provider": config.provider,
        "benchmark_type": config.benchmark_type,
        "checks": checks,
        "telemetry": telemetry_capabilities(),
        "offline_command": offline_command,
        "server_command": server_command,
        "resolved_host": endpoint(config)[0],
        "resolved_port": endpoint(config)[1],
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    source_paths: dict[str, Path],
    results_root: str | Path,
    run_directory: str | Path | None = None,
    repository: str | Path | None = None,
) -> Path:
    """Execute one fully resolved experiment and return its result directory."""

    _configure_cache_environment()
    validate_runtime(config)
    root = Path(results_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = (
        Path(run_directory).expanduser().resolve()
        if run_directory is not None
        else _new_run_directory(config, root)
    )
    destination.mkdir(parents=True, exist_ok=False)
    configuration_hashes = {name: _file_hash(path) for name, path in source_paths.items()}
    resolved = config.to_dict()
    resolved["configuration_hashes"] = configuration_hashes
    (destination / "resolved_experiment.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )

    metadata = collect_metadata(config, run_id=destination.name, repository=repository)
    metadata["configuration_hashes"] = configuration_hashes
    artifact_manifest = _artifact_provenance(config)
    if artifact_manifest:
        if config.backend == "llamacpp":
            selected = artifact_manifest.get("artifacts", {}).get(
                config.artifact_variant, {}
            )
            metadata["model_artifact_sha256"] = selected.get("sha256")
        metadata["model_artifact_source_revision"] = artifact_manifest.get("resolved_revision")
        metadata["resolved_model_revision"] = (
            artifact_manifest.get("resolved_revision") or metadata.get("resolved_model_revision")
        )
        _json_write(destination / "model_artifact_provenance.json", artifact_manifest)
    elif config.backend == "vllm":
        metadata["model_artifact_source_revision"] = metadata.get(
            "resolved_model_revision"
        )
    write_metadata(destination / "metadata.json", metadata)
    summary = create_summary(config, metadata)
    summary["model_artifact_sha256"] = metadata.get("model_artifact_sha256")
    summary["model_artifact_source_revision"] = metadata.get("model_artifact_source_revision")
    summary["configuration_hashes"] = configuration_hashes
    write_summary(destination / "summary.json", summary)

    backend = get_backend(config.backend)
    provider = get_provider(config.provider)
    container_name = (
        f"llm-bench-{os.getpid()}-"
        f"{hashlib.sha256(destination.name.encode('utf-8')).hexdigest()[:12]}"
        if config.provider == "docker"
        else None
    )
    context = RunContext(
        config=config,
        run_directory=destination,
        backend=backend,
        provider=provider,
        provider_context=ProviderContext(
            destination,
            Path(config.artifact_path).parent.parent if config.artifact_path else None,
            Path(os.environ["MODEL_CACHE_DIR"]).expanduser()
            if os.environ.get("MODEL_CACHE_DIR")
            else None,
            container_name,
        ),
        summary=summary,
    )
    runner_type = RUNNER_REGISTRY[config.benchmark_type]
    runner = runner_type()
    summary["measurement_method"] = runner.measurement_method
    summary["measurement_scope"] = runner.measurement_scope
    interrupted = False
    try:
        raw_paths, telemetry_paths, warnings, error = runner.run(context)
    except KeyboardInterrupt:
        interrupted = True
        raw_paths = sorted(destination.glob("raw_*output*.json"))
        telemetry_paths = sorted(destination.glob("*telemetry*.csv"))
        warnings = ["Benchmark was interrupted; backend cleanup was requested."]
        error = "interrupted by signal"
    except (ConfigurationError, OSError) as exc:
        raw_paths = sorted(destination.glob("raw_*output*.json"))
        telemetry_paths = sorted(destination.glob("*telemetry*.csv"))
        warnings = []
        error = str(exc)
    metadata["container_id"] = summary.get("container_id")
    metadata["measurement_method"] = summary.get("measurement_method")
    metadata["measurement_scope"] = summary.get("measurement_scope")
    metadata["backend_internal_warmup"] = summary.get("backend_internal_warmup")
    metadata["workload_manifest_sha256"] = summary.get("workload_manifest_sha256")
    metadata["model_load_time_seconds"] = summary.get("model_load_time_seconds")
    write_metadata(destination / "metadata.json", metadata)
    normalized = normalize_summary(
        summary,
        raw_output_paths=raw_paths,
        telemetry_paths=telemetry_paths,
        status="failed" if error else "completed",
        model_load_time_seconds=summary.get("model_load_time_seconds"),
        warnings=warnings,
        error=error,
    )
    normalized["workload_manifest_sha256"] = summary.get("workload_manifest_sha256")
    measurements = create_measurements(
        normalized,
        run_directory=destination,
        raw_output_paths=raw_paths,
        telemetry_paths=telemetry_paths,
    )
    write_measurements(destination / "measurements.json", measurements)
    write_summary(destination / "summary.json", normalized)
    if interrupted:
        raise KeyboardInterrupt
    if error:
        raise ConfigurationError(f"benchmark failed: {error}; partial result: {destination}")
    return destination
