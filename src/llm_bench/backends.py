"""Backend adapters and their backend-specific command construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .config import ConfigurationError, ExperimentConfig
from .preparation import load_artifact_manifest, sha256_file


class BackendAdapter(Protocol):
    name: str

    def validate(self, config: ExperimentConfig, *, require_artifact: bool = True) -> list[str]: ...

    def offline_command(self, config: ExperimentConfig, output: Path) -> list[str]: ...

    def server_command(self, config: ExperimentConfig) -> list[str]: ...

    def health_url(self, host: str, port: int) -> str: ...


class VllmAdapter:
    name = "vllm"

    def validate(self, config: ExperimentConfig, *, require_artifact: bool = True) -> list[str]:
        if config.hardware_type not in {"cpu", "gpu"}:
            raise ConfigurationError("the vLLM adapter requires a CPU or GPU profile")
        if config.quantization is not None:
            raise ConfigurationError("the shipped vLLM model variants are unquantized")
        if (
            config.hardware_type == "cpu"
            and config.cpu_isa_target == "amx"
            and config.dtype != "bfloat16"
        ):
            raise ConfigurationError("the vLLM CPU AMX treatment requires dtype bfloat16")
        checks = [
            f"vLLM {config.hardware_type} profile and Hugging Face artifact are compatible"
        ]
        if require_artifact:
            try:
                from huggingface_hub import snapshot_download

                snapshot = snapshot_download(
                    repo_id=config.model_id,
                    revision=config.model_revision,
                    cache_dir=os.environ.get("MODEL_CACHE_DIR"),
                    local_files_only=True,
                    token=os.environ.get("HF_TOKEN"),
                    # Meta's repositories ship a second copy of the weights as
                    # a consolidated PyTorch checkpoint under original/, which
                    # vLLM never reads: it downloads by allow_patterns. Since
                    # huggingface_hub 1.31 one absent file makes the whole
                    # cached snapshot "incomplete", so a cache that is entirely
                    # sufficient for the run would otherwise fail validation.
                    ignore_patterns=["original/*"],
                )
            except Exception as exc:
                raise ConfigurationError(
                    "vLLM Hugging Face snapshot is not available locally; run prepare-model "
                    "outside the timed job"
                ) from exc
            checks.append(f"Hugging Face snapshot: {Path(snapshot).resolve()}")
        return checks

    def offline_command(self, config: ExperimentConfig, output: Path) -> list[str]:
        command = [
            "vllm",
            "bench",
            "throughput",
            "--model",
            config.model_id,
            "--tokenizer",
            config.tokenizer,
            "--dataset-name",
            "random",
            "--num-prompts",
            str(config.number_of_prompts),
            "--random-input-len",
            str(config.input_length),
            "--random-output-len",
            str(config.output_length),
            "--dtype",
            config.dtype,
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
            "--seed",
            str(config.seed),
            "--output-json",
            str(output),
        ]
        if config.model_revision:
            command.extend(("--revision", config.model_revision))
        if config.gpu_memory_utilization is not None:
            command.extend(("--gpu-memory-utilization", str(config.gpu_memory_utilization)))
        if config.max_model_len is not None:
            command.extend(("--max-model-len", str(config.max_model_len)))
        return command

    def server_command(self, config: ExperimentConfig) -> list[str]:
        host, port = endpoint(config)
        command = [
            "vllm",
            "serve",
            config.model_id,
            "--served-model-name",
            config.model_id,
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            config.dtype,
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
            "--generation-config",
            "vllm",
        ]
        if config.model_revision:
            command.extend(("--revision", config.model_revision))
        if config.gpu_memory_utilization is not None:
            command.extend(("--gpu-memory-utilization", str(config.gpu_memory_utilization)))
        if config.max_model_len is not None:
            command.extend(("--max-model-len", str(config.max_model_len)))
        return command

    def health_url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}/health"


class LlamaCppAdapter:
    name = "llamacpp"

    def validate(self, config: ExperimentConfig, *, require_artifact: bool = True) -> list[str]:
        if config.artifact_format != "gguf":
            raise ConfigurationError("llama.cpp requires a GGUF artifact")
        if (
            config.benchmark_type == "offline"
            and config.thread_count_batch is not None
            and config.thread_count_batch != config.thread_count
        ):
            raise ConfigurationError(
                "llama-bench has one thread-count control; offline profiles require "
                "thread_count_batch to match thread_count"
            )
        if not config.artifact_path:
            raise ConfigurationError("llama.cpp artifact path is unresolved")
        artifact = Path(config.artifact_path)
        if require_artifact and not artifact.is_file():
            raise ConfigurationError(f"GGUF artifact not found: {artifact}")
        if require_artifact:
            manifest = load_artifact_manifest(config.artifact_path)
            if manifest is None:
                raise ConfigurationError(
                    f"artifact provenance manifest not found beside GGUF: {artifact}"
                )
            if manifest.get("model_id") != config.model_id:
                raise ConfigurationError("GGUF provenance model_id does not match the run")
            record = manifest.get("artifacts", {}).get(config.artifact_variant)
            if not isinstance(record, dict) or record.get("sha256") != sha256_file(artifact):
                raise ConfigurationError("GGUF artifact SHA-256 does not match its manifest")
            if config.model_revision and manifest.get("resolved_revision") != config.model_revision:
                raise ConfigurationError("GGUF source revision does not match the resolved run")
        return [f"GGUF artifact: {artifact}"]

    def _placement(self, config: ExperimentConfig, *, short: bool) -> list[str]:
        command: list[str] = []
        if config.thread_count is not None:
            command.extend((("-t" if short else "--threads"), str(config.thread_count)))
        if not short and config.thread_count_batch is not None:
            command.extend(
                (("-tb" if short else "--threads-batch"), str(config.thread_count_batch))
            )
        if config.cpu_mask:
            command.extend(("--cpu-mask", config.cpu_mask))
        if config.numa_policy:
            command.extend(("--numa", config.numa_policy))
        if config.gpu_layers is not None:
            layers = "999" if config.gpu_layers == "all" else str(config.gpu_layers)
            command.extend((("-ngl" if short else "--n-gpu-layers"), layers))
        if config.batch_size is not None:
            command.extend((("-b" if short else "--batch-size"), str(config.batch_size)))
        if config.ubatch_size is not None:
            command.extend((("-ub" if short else "--ubatch-size"), str(config.ubatch_size)))
        return command

    def offline_command(self, config: ExperimentConfig, output: Path) -> list[str]:
        del output  # llama-bench writes JSON to stdout; the runner captures it.
        # llama-bench keeps its built-in pp512/tg128 tests unless -p and -n are
        # cleared, so -pg alone measures two shapes nobody asked for. Its -r is
        # repetitions per test, not a prompt count; the runner already repeats
        # the whole invocation warmup_runs + repetitions times.
        return [
            "llama-bench",
            "-m",
            str(config.artifact_path),
            *self._placement(config, short=True),
            "-p",
            "0",
            "-n",
            "0",
            "-r",
            "1",
            "-o",
            "json",
            "-pg",
            f"{config.input_length},{config.output_length}",
        ]

    def server_command(self, config: ExperimentConfig) -> list[str]:
        host, port = endpoint(config)
        command = [
            "llama-server",
            "--model",
            str(config.artifact_path),
            "--ctx-size",
            str(config.max_model_len),
            *self._placement(config, short=False),
            "--host",
            host,
            "--port",
            str(port),
            "--alias",
            config.model_id,
            "--metrics",
            "--ignore-eos",
        ]
        if config.parallel_slots is not None:
            command.extend(("--parallel", str(config.parallel_slots)))
        return command

    def health_url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}/health"


def endpoint(config: ExperimentConfig) -> tuple[str, int]:
    host = os.environ.get("LLM_BENCH_HOST") or config.host
    port = int(os.environ.get("LLM_BENCH_PORT") or str(config.port))
    return host, port


BACKEND_REGISTRY: dict[str, type[BackendAdapter]] = {
    "vllm": VllmAdapter,
    "llamacpp": LlamaCppAdapter,
}


def get_backend(name: str) -> BackendAdapter:
    try:
        adapter_type = BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise ConfigurationError(f"unknown backend: {name}") from exc
    return adapter_type()
