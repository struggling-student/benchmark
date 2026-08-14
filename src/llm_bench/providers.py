"""Safe command providers for native, Docker, and Apptainer execution."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import ConfigurationError, ExperimentConfig

_ENVIRONMENT_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "OMP_NUM_THREADS",
    "TRANSFORMERS_CACHE",
)
_OCI_DIGEST = re.compile(r"@sha256:[0-9a-fA-F]{64}$")


@dataclass(frozen=True, slots=True)
class Mount:
    source: Path
    destination: Path
    read_only: bool = True


@dataclass(frozen=True, slots=True)
class ProviderContext:
    run_directory: Path
    artifact_root: Path | None
    cache_root: Path | None
    container_name: str | None = None


class CommandProvider(Protocol):
    name: str

    def validate(self, config: ExperimentConfig) -> list[str]: ...

    def wrap(
        self,
        command: list[str],
        config: ExperimentConfig,
        context: ProviderContext,
        *,
        server: bool = False,
    ) -> list[str]: ...


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value or "<" in value or ">" in value:
        raise ConfigurationError(f"{name} is required for this provider")
    return value


def _allowed_environment() -> list[tuple[str, str]]:
    return [(name, os.environ[name]) for name in _ENVIRONMENT_ALLOWLIST if name in os.environ]


def _memory_prefix(config: ExperimentConfig) -> list[str]:
    if not config.memory_binding:
        return []
    numactl = shutil.which("numactl")
    if numactl is None:
        raise ConfigurationError("memory_binding requires the numactl executable")
    return [numactl, "--membind", config.memory_binding]


def _container_command(command: list[str], config: ExperimentConfig) -> list[str]:
    mapped = list(command)
    if config.backend == "llamacpp" and mapped[0] in {
        "llama-bench",
        "llama-quantize",
        "llama-server",
    }:
        mapped[0] = f"/app/{mapped[0]}"
    return mapped


class NativeProvider:
    name = "native"

    def _resolve(self, command: str, config: ExperimentConfig) -> str | None:
        if config.backend == "llamacpp" and os.environ.get("LLAMA_CPP_BIN_DIR"):
            candidate = Path(os.environ["LLAMA_CPP_BIN_DIR"]).expanduser() / command
            return (
                str(candidate.resolve())
                if candidate.is_file() and os.access(candidate, os.X_OK)
                else None
            )
        if config.backend == "vllm" and command == "vllm" and os.environ.get("VLLM_BIN"):
            candidate = Path(os.environ["VLLM_BIN"]).expanduser()
            return (
                str(candidate.resolve())
                if candidate.is_file() and os.access(candidate, os.X_OK)
                else None
            )
        return shutil.which(command)

    def validate(self, config: ExperimentConfig) -> list[str]:
        commands = (
            ("vllm",)
            if config.backend == "vllm"
            else ("llama-bench",)
            if config.benchmark_type == "offline"
            else ("llama-server",)
        )
        missing = [command for command in commands if self._resolve(command, config) is None]
        if missing:
            raise ConfigurationError("missing native executable(s): " + ", ".join(missing))
        checks: list[str] = []
        for command in commands:
            executable = self._resolve(command, config)
            assert executable is not None
            try:
                completed = subprocess.run(
                    [executable, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise ConfigurationError(
                    f"cannot execute native runtime version check: {executable}"
                ) from exc
            if completed.returncode != 0:
                raise ConfigurationError(
                    f"native runtime version check failed ({completed.returncode}): {executable}"
                )
            version = (completed.stdout or completed.stderr).strip().splitlines()
            identity = version[0] if version else "version output unavailable"
            checks.append(f"native executable: {executable} ({identity})")
        return checks

    def wrap(
        self,
        command: list[str],
        config: ExperimentConfig,
        context: ProviderContext,
        *,
        server: bool = False,
    ) -> list[str]:
        del context, server
        resolved = self._resolve(command[0], config)
        if resolved is None:
            raise ConfigurationError(f"native executable unavailable: {command[0]}")
        wrapped = [resolved, *command[1:]]
        wrapped = [*_memory_prefix(config), *wrapped]
        return wrapped


class DockerProvider:
    name = "docker"

    def _image(self, config: ExperimentConfig) -> str:
        variable = "VLLM_DOCKER_IMAGE" if config.backend == "vllm" else "LLAMA_CPP_DOCKER_IMAGE"
        image = _required_env(variable)
        if _OCI_DIGEST.search(image) is None:
            raise ConfigurationError(f"{variable} must pin an immutable @sha256 digest")
        return image

    def validate(self, config: ExperimentConfig) -> list[str]:
        if shutil.which("docker") is None:
            raise ConfigurationError("docker executable is unavailable")
        image = self._image(config)
        return ["docker executable available", f"pinned image: {image}"]

    def wrap(
        self,
        command: list[str],
        config: ExperimentConfig,
        context: ProviderContext,
        *,
        server: bool = False,
    ) -> list[str]:
        command = _container_command(command, config)
        name = context.container_name or f"llm-bench-{os.getpid()}"
        wrapped = ["docker", "run", "--rm", "--name", name]
        if config.hardware_type in {"gpu", "hybrid"}:
            visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
            gpu_selection = (
                f"device={visibility}"
                if visibility and visibility.strip().lower() != "all"
                else "all"
            )
            wrapped.extend(("--gpus", gpu_selection))
        for name, _value in _allowed_environment():
            wrapped.extend(("--env", name))
        if server:
            wrapped.extend(("-p", f"{config_port(config)}:{config_port(config)}"))
            if "--host" in command:
                command = list(command)
                command[command.index("--host") + 1] = "0.0.0.0"
        mounts = _mounts(context)
        for mount in mounts:
            mode = "ro" if mount.read_only else "rw"
            specification = (
                f"type=bind,src={mount.source},dst={mount.destination},{mode}"
            )
            wrapped.extend(("--mount", specification))
        wrapped.extend(("--entrypoint", command[0], self._image(config), *command[1:]))
        return [*_memory_prefix(config), *wrapped]


class ApptainerProvider:
    name = "apptainer"

    def _runtime(self) -> str:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
        if runtime is None:
            raise ConfigurationError("apptainer/singularity executable is unavailable")
        return runtime

    def _image(self, config: ExperimentConfig) -> str:
        variable = (
            "VLLM_APPTAINER_IMAGE" if config.backend == "vllm" else "LLAMA_CPP_APPTAINER_IMAGE"
        )
        image = _required_env(variable)
        if image.startswith("docker://") and _OCI_DIGEST.search(image) is None:
            raise ConfigurationError(f"{variable} OCI references must pin an @sha256 digest")
        if not image.startswith("docker://") and not Path(image).expanduser().is_file():
            raise ConfigurationError(f"Apptainer image not found: {image}")
        return image

    def validate(self, config: ExperimentConfig) -> list[str]:
        runtime = self._runtime()
        image = self._image(config)
        return [f"container runtime available: {Path(runtime).name}", f"image: {image}"]

    def wrap(
        self,
        command: list[str],
        config: ExperimentConfig,
        context: ProviderContext,
        *,
        server: bool = False,
    ) -> list[str]:
        del server
        command = _container_command(command, config)
        wrapped = [self._runtime(), "exec", "--cleanenv"]
        if config.hardware_type in {"gpu", "hybrid"}:
            wrapped.append("--nv")
        for name, value in _allowed_environment():
            wrapped.extend(("--env", f"{name}={value}"))
        for mount in _mounts(context):
            suffix = ":ro" if mount.read_only else ":rw"
            wrapped.extend(("--bind", f"{mount.source}:{mount.destination}{suffix}"))
        wrapped.extend((self._image(config), *command))
        return [*_memory_prefix(config), *wrapped]


class SourceBuildProvider:
    """Reserved provider contract for a future pinned source-build workflow."""

    name = "source-build"

    def validate(self, config: ExperimentConfig) -> list[str]:
        del config
        raise ConfigurationError("source-build provider automation is reserved but not implemented")

    def wrap(
        self,
        command: list[str],
        config: ExperimentConfig,
        context: ProviderContext,
        *,
        server: bool = False,
    ) -> list[str]:
        del command, config, context, server
        raise ConfigurationError("source-build provider automation is reserved but not implemented")


def config_port(config: ExperimentConfig) -> int:
    return int(os.environ.get("LLM_BENCH_PORT") or str(config.port))


def _mounts(context: ProviderContext) -> list[Mount]:
    mounts = [Mount(context.run_directory.resolve(), context.run_directory.resolve(), False)]
    for root in (context.artifact_root, context.cache_root):
        if root is not None:
            mounts.append(Mount(root.resolve(), root.resolve(), True))
    return mounts


PROVIDER_REGISTRY: dict[str, type[CommandProvider]] = {
    "native": NativeProvider,
    "docker": DockerProvider,
    "apptainer": ApptainerProvider,
}


def get_provider(name: str) -> CommandProvider:
    try:
        provider_type = PROVIDER_REGISTRY[name]
    except KeyError as exc:
        raise ConfigurationError(f"unknown provider: {name}") from exc
    return provider_type()
