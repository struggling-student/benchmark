"""Composable model, workload, and execution-profile configuration."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_TYPES = ("smoke", "offline", "serving")
BACKENDS = ("vllm", "llamacpp")
PROVIDERS = ("native", "docker", "apptainer")
CONFIG_SCHEMA_VERSION = "2.0"


class ConfigurationError(ValueError):
    """Raised when a configuration file or combination is invalid."""


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    format: str
    filename: str | None = None


@dataclass(frozen=True, slots=True)
class ModelVariant:
    name: str
    precision: str
    quantization: str | None
    artifacts: dict[str, ArtifactConfig]


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_key: str
    model_id: str
    tokenizer_id: str
    model_revision: str | None
    revision_policy: str
    model_parameter_scale: str | None
    max_model_len: int
    variants: dict[str, ModelVariant]
    source_path: Path


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    workload_key: str
    benchmark_type: str
    seed: int
    number_of_prompts: int
    input_length: int
    output_length: int
    repetitions: int
    warmup_runs: int
    telemetry_interval_ms: int
    temperature: float | None
    top_p: float | None
    ignore_eos: bool | None
    request_rate: float | None
    maximum_concurrency: int | None
    source_path: Path


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    profile_name: str
    backend: str
    hardware_type: str
    host: str
    port: int
    tensor_parallel_size: int
    gpu_memory_utilization: float | None
    thread_count: int | None
    thread_count_batch: int | None
    cpu_mask: str | None
    numa_policy: str | None
    memory_binding: str | None
    memory_type: str | None
    memory_mode: str | None
    gpu_layers: int | str | None
    batch_size: int | None
    ubatch_size: int | None
    parallel_slots: int | None
    source_path: Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Fully resolved run contract consumed by runners and result writers."""

    experiment_name: str
    backend: str
    model_id: str
    model_revision: str | None = None
    model_revision_policy: str | None = None
    tokenizer_id: str | None = None
    model_parameter_scale: str | None = None
    benchmark_type: str = "offline"
    dtype: str = "auto"
    quantization: str | None = None
    tensor_parallel_size: int = 1
    seed: int = 0
    number_of_prompts: int = 1
    input_length: int = 128
    output_length: int = 32
    max_model_len: int | None = None
    generation_config: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    ignore_eos: bool | None = None
    request_rate: float | None = None
    maximum_concurrency: int | None = None
    gpu_memory_utilization: float | None = None
    repetitions: int = 1
    warmup_runs: int = 0
    telemetry_interval_ms: int = 1000
    model_key: str | None = None
    workload_key: str | None = None
    profile_name: str | None = None
    provider: str = "native"
    artifact_variant: str = "f16"
    artifact_format: str | None = None
    artifact_path: str | None = None
    artifact_manifest_path: str | None = None
    hardware_type: str = "unknown"
    host: str = "127.0.0.1"
    port: int = 8000
    thread_count: int | None = None
    thread_count_batch: int | None = None
    cpu_mask: str | None = None
    numa_policy: str | None = None
    memory_binding: str | None = None
    memory_type: str | None = None
    memory_mode: str | None = None
    gpu_layers: int | str | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    parallel_slots: int | None = None

    @property
    def tokenizer(self) -> str:
        return self.tokenizer_id or self.model_id

    @property
    def resolved_benchmark_type(self) -> str:
        return self.benchmark_type

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tokenizer_id"] = self.tokenizer
        return data


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a YAML mapping")
    return value


def _strict(data: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown {label} field(s): " + ", ".join(unknown))


def _string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _string(name, value)


def _integer(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_integer(name: str, value: object, minimum: int = 1) -> int | None:
    return None if value is None else _integer(name, value, minimum)


def _positive_number(name: str, value: object, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a positive number")
    number = float(value)
    if number <= 0 or not math.isfinite(number) or (maximum is not None and number > maximum):
        suffix = f" <= {maximum}" if maximum is not None else ""
        raise ConfigurationError(f"{name} must be a finite number > 0{suffix}")
    return number


def _read_yaml(path: str | Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"{label} not found: {source}")
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read {source}: {exc}") from exc
    return source, _mapping(data, label)


def _schema(data: Mapping[str, Any], label: str) -> None:
    if str(data.get("schema_version")) != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"{label} schema_version must be {CONFIG_SCHEMA_VERSION}; legacy experiment "
            "YAML is no longer accepted"
        )


def load_model_manifest(path: str | Path) -> ModelManifest:
    source, data = _read_yaml(path, "model manifest")
    _schema(data, "model manifest")
    _strict(
        data,
        {
            "schema_version",
            "model_key",
            "model_id",
            "tokenizer_id",
            "model_revision",
            "revision_policy",
            "model_parameter_scale",
            "max_model_len",
            "variants",
        },
        "model manifest",
    )
    variants_data = _mapping(data.get("variants"), "variants")
    variants: dict[str, ModelVariant] = {}
    for raw_name, raw_variant in variants_data.items():
        name = _string("variant name", raw_name)
        variant = _mapping(raw_variant, f"variant {name}")
        _strict(variant, {"precision", "quantization", "artifacts"}, f"variant {name}")
        artifact_data = _mapping(variant.get("artifacts"), f"variant {name} artifacts")
        artifacts: dict[str, ArtifactConfig] = {}
        for raw_backend, raw_artifact in artifact_data.items():
            backend = _string("artifact backend", raw_backend)
            if backend not in BACKENDS:
                raise ConfigurationError(f"unsupported artifact backend: {backend}")
            artifact = _mapping(raw_artifact, f"{name}/{backend} artifact")
            _strict(artifact, {"format", "filename"}, f"{name}/{backend} artifact")
            filename = _optional_string("filename", artifact.get("filename"))
            artifact_format = _string("format", artifact.get("format"))
            if artifact_format == "gguf" and filename is None:
                raise ConfigurationError(f"{name}/{backend} GGUF artifact requires filename")
            artifacts[backend] = ArtifactConfig(artifact_format, filename)
        if not artifacts:
            raise ConfigurationError(f"variant {name} must support at least one backend")
        variants[name] = ModelVariant(
            name=name,
            precision=_string("precision", variant.get("precision")),
            quantization=_optional_string("quantization", variant.get("quantization")),
            artifacts=artifacts,
        )
    if not variants:
        raise ConfigurationError("model manifest requires at least one variant")
    model_id = _string("model_id", data.get("model_id"))
    revision_policy = _string("revision_policy", data.get("revision_policy"))
    if revision_policy not in {"pinned", "resolve-and-record"}:
        raise ConfigurationError("revision_policy must be pinned or resolve-and-record")
    model_revision = _optional_string("model_revision", data.get("model_revision"))
    if revision_policy == "pinned" and model_revision is None:
        raise ConfigurationError("pinned revision_policy requires model_revision")
    return ModelManifest(
        model_key=_string("model_key", data.get("model_key")),
        model_id=model_id,
        tokenizer_id=_optional_string("tokenizer_id", data.get("tokenizer_id")) or model_id,
        model_revision=model_revision,
        revision_policy=revision_policy,
        model_parameter_scale=_optional_string(
            "model_parameter_scale", data.get("model_parameter_scale")
        ),
        max_model_len=_integer("max_model_len", data.get("max_model_len")),
        variants=variants,
        source_path=source,
    )


def load_workload(path: str | Path) -> WorkloadConfig:
    source, data = _read_yaml(path, "workload")
    _schema(data, "workload")
    _strict(
        data,
        {
            "schema_version",
            "workload_key",
            "benchmark_type",
            "seed",
            "number_of_prompts",
            "input_length",
            "output_length",
            "repetitions",
            "warmup_runs",
            "telemetry_interval_ms",
            "temperature",
            "top_p",
            "ignore_eos",
            "request_rate",
            "maximum_concurrency",
        },
        "workload",
    )
    benchmark_type = _string("benchmark_type", data.get("benchmark_type"))
    if benchmark_type not in BENCHMARK_TYPES:
        raise ConfigurationError(f"benchmark_type must be one of {', '.join(BENCHMARK_TYPES)}")
    temperature = data.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ConfigurationError("temperature must be null or a non-negative number")
        temperature = float(temperature)
        if temperature < 0 or not math.isfinite(temperature):
            raise ConfigurationError("temperature must be null or a non-negative finite number")
    top_p = data.get("top_p")
    if top_p is not None:
        top_p = _positive_number("top_p", top_p, maximum=1.0)
    ignore_eos = data.get("ignore_eos")
    if ignore_eos is not None and not isinstance(ignore_eos, bool):
        raise ConfigurationError("ignore_eos must be null or a boolean")
    request_rate = data.get("request_rate")
    if request_rate is not None:
        request_rate = _positive_number("request_rate", request_rate)
    maximum_concurrency = _optional_integer(
        "maximum_concurrency", data.get("maximum_concurrency")
    )
    if benchmark_type == "offline":
        if any(
            value is not None
            for value in (temperature, top_p, ignore_eos, request_rate, maximum_concurrency)
        ):
            raise ConfigurationError("offline workloads require API generation controls to be null")
    else:
        missing = [
            name
            for name, value in (
                ("temperature", temperature),
                ("top_p", top_p),
                ("ignore_eos", ignore_eos),
                ("maximum_concurrency", maximum_concurrency),
            )
            if value is None
        ]
        if benchmark_type == "serving" and request_rate is None:
            missing.append("request_rate")
        if missing:
            raise ConfigurationError(
                f"{benchmark_type} workload requires: " + ", ".join(missing)
            )
        if ignore_eos is not True:
            raise ConfigurationError("API workloads require ignore_eos: true")
    input_length = _integer("input_length", data.get("input_length"))
    output_length = _integer("output_length", data.get("output_length"))
    return WorkloadConfig(
        workload_key=_string("workload_key", data.get("workload_key")),
        benchmark_type=benchmark_type,
        seed=_integer("seed", data.get("seed", 0), 0),
        number_of_prompts=_integer("number_of_prompts", data.get("number_of_prompts")),
        input_length=input_length,
        output_length=output_length,
        repetitions=_integer("repetitions", data.get("repetitions", 1)),
        warmup_runs=_integer("warmup_runs", data.get("warmup_runs", 0), 0),
        telemetry_interval_ms=_integer(
            "telemetry_interval_ms", data.get("telemetry_interval_ms", 1000)
        ),
        temperature=temperature,
        top_p=top_p,
        ignore_eos=ignore_eos,
        request_rate=request_rate,
        maximum_concurrency=maximum_concurrency,
        source_path=source,
    )


def load_execution_profile(path: str | Path) -> ExecutionProfile:
    source, data = _read_yaml(path, "execution profile")
    _schema(data, "execution profile")
    _strict(
        data,
        {
            "schema_version",
            "profile_name",
            "backend",
            "hardware_type",
            "host",
            "port",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "thread_count",
            "thread_count_batch",
            "cpu_mask",
            "numa_policy",
            "memory_binding",
            "memory_type",
            "memory_mode",
            "gpu_layers",
            "batch_size",
            "ubatch_size",
            "parallel_slots",
        },
        "execution profile",
    )
    backend = _string("backend", data.get("backend"))
    if backend not in BACKENDS:
        raise ConfigurationError(f"backend must be one of {', '.join(BACKENDS)}")
    hardware = _string("hardware_type", data.get("hardware_type"))
    if hardware not in {"cpu", "gpu", "hybrid"}:
        raise ConfigurationError("hardware_type must be cpu, gpu, or hybrid")
    gpu_memory = data.get("gpu_memory_utilization")
    if gpu_memory is not None:
        gpu_memory = _positive_number("gpu_memory_utilization", gpu_memory, maximum=1.0)
    gpu_layers = data.get("gpu_layers")
    if gpu_layers != "all" and gpu_layers is not None:
        gpu_layers = _integer("gpu_layers", gpu_layers, 0)
    profile = ExecutionProfile(
        profile_name=_string("profile_name", data.get("profile_name")),
        backend=backend,
        hardware_type=hardware,
        host=_string("host", data.get("host", "127.0.0.1")),
        port=_integer("port", data.get("port", 8000)),
        tensor_parallel_size=_integer(
            "tensor_parallel_size", data.get("tensor_parallel_size", 1)
        ),
        gpu_memory_utilization=gpu_memory,
        thread_count=_optional_integer("thread_count", data.get("thread_count")),
        thread_count_batch=_optional_integer(
            "thread_count_batch", data.get("thread_count_batch")
        ),
        cpu_mask=_optional_string("cpu_mask", data.get("cpu_mask")),
        numa_policy=_optional_string("numa_policy", data.get("numa_policy")),
        memory_binding=_optional_string("memory_binding", data.get("memory_binding")),
        memory_type=_optional_string("memory_type", data.get("memory_type")),
        memory_mode=_optional_string("memory_mode", data.get("memory_mode")),
        gpu_layers=gpu_layers,
        batch_size=_optional_integer("batch_size", data.get("batch_size")),
        ubatch_size=_optional_integer("ubatch_size", data.get("ubatch_size")),
        parallel_slots=_optional_integer("parallel_slots", data.get("parallel_slots")),
        source_path=source,
    )
    if backend == "vllm" and any(
        value is not None
        for value in (
            profile.thread_count,
            profile.thread_count_batch,
            profile.cpu_mask,
            profile.numa_policy,
            profile.memory_binding,
            profile.gpu_layers,
            profile.batch_size,
            profile.ubatch_size,
            profile.parallel_slots,
        )
    ):
        raise ConfigurationError("vLLM profiles cannot contain llama.cpp execution controls")
    if backend == "llamacpp" and hardware == "cpu" and gpu_layers not in (None, 0):
        raise ConfigurationError("CPU llama.cpp profiles require gpu_layers: 0")
    if hardware == "cpu" and (profile.memory_type is None or profile.memory_mode is None):
        raise ConfigurationError("CPU profiles require explicit memory_type and memory_mode")
    if backend == "llamacpp" and hardware in {"gpu", "hybrid"} and gpu_layers in (None, 0):
        raise ConfigurationError("GPU/hybrid llama.cpp profiles require non-zero gpu_layers")
    return profile


def resolve_experiment(
    model: ModelManifest,
    workload: WorkloadConfig,
    profile: ExecutionProfile,
    *,
    provider: str,
    variant: str,
    artifact_root: str | Path | None = None,
) -> ExperimentConfig:
    if provider not in PROVIDERS:
        raise ConfigurationError(f"provider must be one of {', '.join(PROVIDERS)}")
    if variant not in model.variants:
        raise ConfigurationError(
            f"unknown model variant {variant!r}; choose: {', '.join(sorted(model.variants))}"
        )
    selected = model.variants[variant]
    if profile.backend not in selected.artifacts:
        raise ConfigurationError(
            f"model variant {variant!r} does not support backend {profile.backend!r}"
        )
    artifact = selected.artifacts[profile.backend]
    artifact_path: str | None = None
    effective_revision = model.model_revision
    root: Path | None = None
    artifact_manifest_path: str | None = None
    root_value = artifact_root or os.environ.get("MODEL_ARTIFACT_ROOT")
    if root_value:
        root = Path(root_value).expanduser()
        if not root.is_absolute():
            raise ConfigurationError("MODEL_ARTIFACT_ROOT must be an absolute path")
        manifest_path = root / model.model_key / "artifact-manifest.json"
        artifact_manifest_path = str(manifest_path.resolve())
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"cannot read artifact manifest: {manifest_path}") from exc
            if manifest.get("model_id") != model.model_id:
                raise ConfigurationError("artifact manifest model_id does not match model manifest")
            recorded_revision = manifest.get("resolved_revision")
            if isinstance(recorded_revision, str) and recorded_revision:
                if model.model_revision and model.model_revision != recorded_revision:
                    raise ConfigurationError(
                        "artifact manifest revision does not match pinned model_revision"
                    )
                effective_revision = recorded_revision
    if artifact.filename:
        if root is not None:
            artifact_path = str((root / model.model_key / artifact.filename).resolve())
        else:
            artifact_path = artifact.filename
    if model.max_model_len < workload.input_length + workload.output_length:
        raise ConfigurationError(
            "model max_model_len must be at least workload input_length + output_length"
        )
    api_workload = workload.benchmark_type in {"smoke", "serving"}
    return ExperimentConfig(
        experiment_name=(
            f"{model.model_key}-{workload.workload_key}-{profile.profile_name}-{variant}"
        ),
        backend=profile.backend,
        model_id=model.model_id,
        model_revision=effective_revision,
        model_revision_policy=model.revision_policy,
        tokenizer_id=model.tokenizer_id,
        model_parameter_scale=model.model_parameter_scale,
        benchmark_type=workload.benchmark_type,
        dtype=selected.precision,
        quantization=selected.quantization,
        tensor_parallel_size=profile.tensor_parallel_size,
        seed=workload.seed,
        number_of_prompts=workload.number_of_prompts,
        input_length=workload.input_length,
        output_length=workload.output_length,
        max_model_len=model.max_model_len,
        generation_config="controlled" if api_workload else None,
        temperature=workload.temperature,
        top_p=workload.top_p,
        ignore_eos=workload.ignore_eos,
        request_rate=workload.request_rate,
        maximum_concurrency=workload.maximum_concurrency,
        gpu_memory_utilization=profile.gpu_memory_utilization,
        repetitions=workload.repetitions,
        warmup_runs=workload.warmup_runs,
        telemetry_interval_ms=workload.telemetry_interval_ms,
        model_key=model.model_key,
        workload_key=workload.workload_key,
        profile_name=profile.profile_name,
        provider=provider,
        artifact_variant=variant,
        artifact_format=artifact.format,
        artifact_path=artifact_path,
        artifact_manifest_path=artifact_manifest_path,
        hardware_type=profile.hardware_type,
        host=profile.host,
        port=profile.port,
        thread_count=profile.thread_count,
        thread_count_batch=profile.thread_count_batch,
        cpu_mask=profile.cpu_mask,
        numa_policy=profile.numa_policy,
        memory_binding=profile.memory_binding,
        memory_type=profile.memory_type,
        memory_mode=profile.memory_mode,
        gpu_layers=profile.gpu_layers,
        batch_size=profile.batch_size,
        ubatch_size=profile.ubatch_size,
        parallel_slots=profile.parallel_slots,
    )


def load_composed_experiment(
    model_path: str | Path,
    workload_path: str | Path,
    profile_path: str | Path,
    *,
    provider: str,
    variant: str,
    artifact_root: str | Path | None = None,
) -> ExperimentConfig:
    return resolve_experiment(
        load_model_manifest(model_path),
        load_workload(workload_path),
        load_execution_profile(profile_path),
        provider=provider,
        variant=variant,
        artifact_root=artifact_root,
    )


def config_from_mapping(data: Mapping[str, Any]) -> ExperimentConfig:
    """Build a resolved config for internal/tests; file-based legacy loading is removed."""

    if not isinstance(data, Mapping):
        raise ConfigurationError("resolved experiment configuration must be a mapping")
    allowed = set(ExperimentConfig.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError("unknown resolved field(s): " + ", ".join(unknown))
    missing = [name for name in ("experiment_name", "backend", "model_id") if name not in data]
    if missing:
        raise ConfigurationError("missing required field(s): " + ", ".join(missing))
    config = ExperimentConfig(**dict(data))
    if config.backend not in BACKENDS:
        raise ConfigurationError(f"backend must be one of {', '.join(BACKENDS)}")
    if config.benchmark_type not in BENCHMARK_TYPES:
        raise ConfigurationError(f"benchmark_type must be one of {', '.join(BENCHMARK_TYPES)}")
    return config


def load_experiment(path: str | Path) -> ExperimentConfig:
    raise ConfigurationError(
        "legacy monolithic experiment YAML is no longer supported; use --model, --workload, "
        "--profile, --provider, and --variant"
    )
