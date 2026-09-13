"""Portable process, Docker, NVIDIA, and optional Linux RAPL telemetry."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


def _number(value: str) -> float | None:
    try:
        return float(value.strip().rstrip("%"))
    except (AttributeError, ValueError):
        return None


def _process_sample(
    pid: int, tracked: dict[int, Any] | None = None
) -> tuple[float | None, float | None]:
    # cpu_percent(interval=None) reports the load since the previous call *on
    # that same Process object* and returns 0.0 the first time it is asked.
    # Re-deriving the objects every sample therefore pins CPU utilisation at
    # zero forever, so the caller keeps them alive in `tracked`.
    try:
        import psutil
    except ImportError:
        return None, None
    if tracked is None:
        tracked = {}
    try:
        root = tracked.get(pid)
        if root is None:
            root = tracked[pid] = psutil.Process(pid)
        processes = []
        for process in (root, *root.children(recursive=True)):
            processes.append(tracked.setdefault(process.pid, process))
        live = [process for process in processes if process.is_running()]
        cpu = sum(process.cpu_percent(interval=None) for process in live)
        rss = sum(process.memory_info().rss for process in live)
        for dead in [key for key, value in tracked.items() if not value.is_running()]:
            del tracked[dead]
        return cpu, rss / (1024 * 1024)
    except (psutil.Error, OSError):
        return None, None


def _docker_sample(name: str) -> tuple[float | None, float | None]:
    try:
        completed = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(completed.stdout.strip()) if completed.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None, None
    cpu = _number(data.get("CPUPerc", ""))
    memory_text = str(data.get("MemUsage", "")).split("/", 1)[0].strip()
    multiplier = 1.0
    if memory_text.endswith("GiB"):
        multiplier = 1024.0
    elif memory_text.endswith("KiB"):
        multiplier = 1 / 1024
    memory = _number(memory_text.removesuffix("GiB").removesuffix("MiB").removesuffix("KiB"))
    return cpu, memory * multiplier if memory is not None else None


def _gpu_sample() -> tuple[float | None, float | None, float | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None, None
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get(
        "NVIDIA_VISIBLE_DEVICES"
    )
    if not visibility or visibility.strip().lower() in {"", "all", "none", "void", "-1"}:
        return None, None, None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                visibility,
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    rows: list[tuple[float, float, float]] = []
    for line in completed.stdout.splitlines() if completed.returncode == 0 else []:
        values = [_number(value) for value in line.split(",")]
        if len(values) == 3 and all(value is not None for value in values):
            rows.append((values[0], values[1], values[2]))  # type: ignore[arg-type]
    if not rows:
        return None, None, None
    return (
        sum(row[0] for row in rows) / len(rows),
        sum(row[1] for row in rows),
        sum(row[2] for row in rows),
    )


def _rapl_energy() -> float | None:
    values: list[float] = []
    for path in Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"):
        try:
            values.append(float(path.read_text(encoding="utf-8").strip()) / 1_000_000)
        except (OSError, ValueError):
            continue
    return sum(values) if values else None


class TelemetryCollector(Protocol):
    """Typed contract for one independently sampled telemetry source."""

    name: str

    def sample(self, monotonic_time: float) -> dict[str, float | None]: ...


class ProcessTelemetryCollector:
    name = "process"

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._tracked: dict[int, Any] = {}

    def sample(self, monotonic_time: float) -> dict[str, float | None]:
        del monotonic_time
        cpu, memory = _process_sample(self.pid, self._tracked)
        return {"cpu_utilization_percent": cpu, "rss_mib": memory}


class DockerTelemetryCollector:
    name = "docker"

    def __init__(self, container_name: str) -> None:
        self.container_name = container_name

    def sample(self, monotonic_time: float) -> dict[str, float | None]:
        del monotonic_time
        cpu, memory = _docker_sample(self.container_name)
        return {"cpu_utilization_percent": cpu, "rss_mib": memory}


class NvidiaTelemetryCollector:
    name = "nvidia"

    def sample(self, monotonic_time: float) -> dict[str, float | None]:
        del monotonic_time
        utilization, memory, power = _gpu_sample()
        return {
            "gpu_utilization_percent": utilization,
            "memory_used_mib": memory,
            "power_draw_watts": power,
        }


class RaplTelemetryCollector:
    name = "rapl"

    def __init__(self) -> None:
        self.previous_time: float | None = None
        self.previous_energy: float | None = None

    def sample(self, monotonic_time: float) -> dict[str, float | None]:
        energy = _rapl_energy()
        power = None
        if (
            energy is not None
            and self.previous_energy is not None
            and self.previous_time is not None
        ):
            elapsed = monotonic_time - self.previous_time
            if elapsed > 0 and energy >= self.previous_energy:
                power = (energy - self.previous_energy) / elapsed
        self.previous_time = monotonic_time
        self.previous_energy = energy
        return {"package_power_watts": power}


TELEMETRY_COLLECTOR_REGISTRY: dict[str, type[TelemetryCollector]] = {
    "process": ProcessTelemetryCollector,
    "docker": DockerTelemetryCollector,
    "nvidia": NvidiaTelemetryCollector,
    "rapl": RaplTelemetryCollector,
}


class TelemetrySession:
    """Sample resources in a daemon thread and write one portable CSV."""

    FIELDS = (
        "timestamp",
        "cpu_utilization_percent",
        "rss_mib",
        "package_power_watts",
        "gpu_utilization_percent",
        "memory_used_mib",
        "power_draw_watts",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        pid: int,
        interval_ms: int,
        provider: str,
        container_name: str | None = None,
        collect_gpu: bool = True,
    ) -> None:
        self.path = Path(path)
        self.pid = pid
        self.interval = interval_ms / 1000
        self.provider = provider
        self.container_name = container_name
        self.collect_gpu = collect_gpu
        resource_collector: TelemetryCollector
        if provider == "docker" and container_name:
            resource_collector = TELEMETRY_COLLECTOR_REGISTRY["docker"](container_name)
        else:
            resource_collector = TELEMETRY_COLLECTOR_REGISTRY["process"](pid)
        self.collectors: list[TelemetryCollector] = [
            resource_collector,
            TELEMETRY_COLLECTOR_REGISTRY["rapl"](),
        ]
        if collect_gpu:
            self.collectors.append(TELEMETRY_COLLECTOR_REGISTRY["nvidia"]())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval * 2))

    def _sample(self) -> None:
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                now = time.monotonic()
                row: dict[str, str | float | None] = {
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                for collector in self.collectors:
                    row.update(collector.sample(now))
                writer.writerow(row)
                stream.flush()
                self._stop.wait(self.interval)


def telemetry_capabilities() -> dict[str, bool]:
    try:
        import psutil  # noqa: F401

        process = True
    except ImportError:
        process = False
    return {
        "process": process,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "rapl": any(Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj")),
        "docker_stats": shutil.which("docker") is not None,
        "memory_bandwidth_tool": bool(
            shutil.which("pcm-memory") or shutil.which("likwid-perfctr")
        ),
        "memory_bandwidth_collection": False,
    }
