"""Load and validate the intentionally small experiment configuration format."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_TYPES = ("smoke", "offline", "serving")


class ConfigurationError(ValueError):
    """Raised when an experiment file is missing or contains invalid values."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Parameters shared by the smoke, offline, and serving workflows."""

    experiment_name: str
    backend: str
    model_id: str
    model_revision: str | None = None
    tokenizer_id: str | None = None
    model_parameter_scale: str | None = None
    benchmark_type: str | None = None
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

    @property
    def tokenizer(self) -> str:
        """Return the explicit tokenizer or the model tokenizer by default."""

        return self.tokenizer_id or self.model_id

    @property
    def resolved_benchmark_type(self) -> str:
        """Return the explicit type or infer an unambiguous name suffix."""

        if self.benchmark_type is not None:
            return self.benchmark_type
        lowered = self.experiment_name.lower()
        matches = [kind for kind in BENCHMARK_TYPES if kind in lowered]
        if len(matches) == 1:
            return matches[0]
        raise ConfigurationError(
            "benchmark_type is required when experiment_name does not identify exactly "
            "one of: smoke, offline, serving"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe representation with tokenizer fallback resolved."""

        data = asdict(self)
        data["tokenizer_id"] = self.tokenizer
        data["benchmark_type"] = self.resolved_benchmark_type
        return data


def _require_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")


def _require_integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ConfigurationError(f"{name} must be an integer {qualifier}")


def _require_optional_string(name: str, value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ConfigurationError(f"{name} must be null or a non-empty string")


def _validate(config: ExperimentConfig) -> None:
    for name in ("experiment_name", "backend", "model_id", "dtype"):
        _require_string(name, getattr(config, name))
    if config.backend != "vllm":
        raise ConfigurationError(
            "backend must be 'vllm' in this initial implementation; future CPU backends "
            "must be added explicitly"
        )
    for name in (
        "model_revision",
        "tokenizer_id",
        "model_parameter_scale",
        "quantization",
        "generation_config",
    ):
        _require_optional_string(name, getattr(config, name))

    if config.benchmark_type is not None and config.benchmark_type not in BENCHMARK_TYPES:
        raise ConfigurationError(f"benchmark_type must be one of {', '.join(BENCHMARK_TYPES)}")
    # Resolve it now so ambiguous names fail during loading rather than during a run.
    benchmark_type = config.resolved_benchmark_type

    for name in (
        "tensor_parallel_size",
        "number_of_prompts",
        "input_length",
        "output_length",
        "repetitions",
        "telemetry_interval_ms",
    ):
        _require_integer(name, getattr(config, name), minimum=1)
    _require_integer("warmup_runs", config.warmup_runs, minimum=0)
    _require_integer("seed", config.seed, minimum=0)

    if config.request_rate is not None:
        if isinstance(config.request_rate, bool) or not isinstance(
            config.request_rate, (int, float)
        ):
            raise ConfigurationError("request_rate must be null or a positive number")
        if config.request_rate <= 0 or not math.isfinite(config.request_rate):
            raise ConfigurationError("request_rate must be null or a positive finite number")
    if config.maximum_concurrency is not None:
        _require_integer("maximum_concurrency", config.maximum_concurrency, minimum=1)
    if config.max_model_len is not None:
        _require_integer("max_model_len", config.max_model_len, minimum=1)
        minimum_workload_length = config.input_length + config.output_length
        if config.max_model_len < minimum_workload_length:
            raise ConfigurationError(
                "max_model_len must be at least input_length + output_length "
                f"({minimum_workload_length})"
            )
    if config.gpu_memory_utilization is not None:
        value = config.gpu_memory_utilization
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError("gpu_memory_utilization must be null or a number in (0, 1]")
        if not 0 < value <= 1:
            raise ConfigurationError("gpu_memory_utilization must be null or a number in (0, 1]")
    if config.temperature is not None:
        value = config.temperature
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError("temperature must be null or a non-negative number")
        if value < 0 or not math.isfinite(value):
            raise ConfigurationError(
                "temperature must be null or a non-negative finite number"
            )
    if config.top_p is not None:
        value = config.top_p
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError("top_p must be null or a number in (0, 1]")
        if not 0 < value <= 1 or not math.isfinite(value):
            raise ConfigurationError("top_p must be null or a finite number in (0, 1]")
    if config.ignore_eos is not None and not isinstance(config.ignore_eos, bool):
        raise ConfigurationError("ignore_eos must be null or a boolean")
    if benchmark_type == "serving":
        missing = [
            name
            for name in (
                "generation_config",
                "temperature",
                "top_p",
                "ignore_eos",
                "request_rate",
                "maximum_concurrency",
            )
            if getattr(config, name) is None
        ]
        if missing:
            raise ConfigurationError("serving experiments require: " + ", ".join(missing))
        if config.generation_config != "vllm":
            raise ConfigurationError(
                "serving generation_config must be 'vllm' so model-specific defaults "
                "cannot change the benchmark workload"
            )
        if config.ignore_eos is not True:
            raise ConfigurationError(
                "serving ignore_eos must be true so every request attempts the "
                "configured output length"
            )
    elif any(
        value is not None
        for value in (
            config.generation_config,
            config.temperature,
            config.top_p,
            config.ignore_eos,
            config.request_rate,
            config.maximum_concurrency,
        )
    ):
        raise ConfigurationError(
            f"{benchmark_type} experiments require serving-only generation, rate, and "
            "concurrency controls to be null because this workflow does not apply them"
        )


def config_from_mapping(data: Mapping[str, Any]) -> ExperimentConfig:
    """Validate a mapping and return an immutable experiment configuration."""

    if not isinstance(data, Mapping):
        raise ConfigurationError("experiment configuration must be a YAML mapping")
    allowed = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError("unknown configuration field(s): " + ", ".join(unknown))
    required = ("experiment_name", "backend", "model_id")
    missing = [name for name in required if name not in data]
    if missing:
        raise ConfigurationError("missing required field(s): " + ", ".join(missing))
    try:
        config = ExperimentConfig(**dict(data))
    except TypeError as exc:
        raise ConfigurationError(f"invalid experiment configuration: {exc}") from exc
    _validate(config)
    return config


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Read one YAML experiment file and validate every supported field."""

    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(f"experiment configuration not found: {source}")
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read {source}: {exc}") from exc
    if data is None:
        raise ConfigurationError(f"experiment configuration is empty: {source}")
    return config_from_mapping(data)
