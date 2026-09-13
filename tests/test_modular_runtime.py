from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_bench import runner as runner_module
from llm_bench.api_client import run_api_benchmark
from llm_bench.backends import LlamaCppAdapter, VllmAdapter
from llm_bench.config import config_from_mapping, load_composed_experiment, load_model_manifest
from llm_bench.preparation import prepare_model
from llm_bench.providers import ApptainerProvider, DockerProvider, NativeProvider, ProviderContext
from llm_bench.results import ResultError, check_backend_treatment_compatibility, read_raw_metrics
from llm_bench.runner import run_experiment, validate_runtime
from llm_bench.telemetry import (
    DockerTelemetryCollector,
    NvidiaTelemetryCollector,
    ProcessTelemetryCollector,
    RaplTelemetryCollector,
)
from llm_bench.workloads import build_native_workload_manifest, build_workload_manifest

ROOT = Path(__file__).parents[1]


def _config(**overrides: Any):
    values: dict[str, Any] = {
        "experiment_name": "modular-test",
        "benchmark_type": "serving",
        "backend": "llamacpp",
        "model_id": "test/model",
        "tokenizer_id": "test/model",
        "dtype": "float16",
        "number_of_prompts": 2,
        "input_length": 4,
        "output_length": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "maximum_concurrency": 1,
        "request_rate": None,
        "artifact_format": "gguf",
        "artifact_path": "/models/model.gguf",
        "provider": "native",
        "hardware_type": "cpu",
        "gpu_layers": 0,
        "thread_count": 4,
        "batch_size": 32,
        "ubatch_size": 16,
        "parallel_slots": 1,
    }
    values.update(overrides)
    return config_from_mapping(values)


def test_backend_command_builders_keep_backend_specific_controls() -> None:
    llama = _config()
    server = LlamaCppAdapter().server_command(llama)
    bench = LlamaCppAdapter().offline_command(
        _config(benchmark_type="offline", temperature=None, top_p=None, ignore_eos=None),
        Path("ignored.json"),
    )
    assert server[:3] == ["llama-server", "--model", "/models/model.gguf"]
    assert "--parallel" in server
    assert "-pg" in bench
    assert "-tb" not in bench
    assert bench[bench.index("-pg") + 1] == "4,2"

    vllm = _config(
        backend="vllm",
        artifact_format="huggingface",
        artifact_path=None,
        hardware_type="gpu",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        thread_count=None,
        gpu_layers=None,
        batch_size=None,
        ubatch_size=None,
        parallel_slots=None,
    )
    command = VllmAdapter().server_command(vllm)
    assert command[:3] == ["vllm", "serve", "test/model"]
    assert command[command.index("--tensor-parallel-size") + 1] == "2"


def test_llamacpp_load_mode_reaches_both_entry_points() -> None:
    # Mapping the GGUF makes the compute threads fault against Lustre while they
    # work; "none" reads the weights into anonymous memory once instead. Both
    # llama-bench and llama-server take the same flag, so both must carry it or
    # the offline and serving numbers stop describing the same memory behaviour.
    bench = LlamaCppAdapter().offline_command(
        _config(
            benchmark_type="offline", temperature=None, top_p=None, ignore_eos=None,
            load_mode="none",
        ),
        Path("ignored.json"),
    )
    server = LlamaCppAdapter().server_command(_config(load_mode="none"))

    assert bench[bench.index("-lm") + 1] == "none"
    assert server[server.index("--load-mode") + 1] == "none"

    plain = LlamaCppAdapter().server_command(_config())
    assert "--load-mode" not in plain


def test_native_provider_uses_configured_binary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("LLAMA_CPP_BIN_DIR", str(tmp_path))
    config = _config()
    provider = NativeProvider()

    assert provider.validate(config) == [
        f"native executable: {binary} (version output unavailable)"
    ]
    assert provider.wrap(
        ["llama-server", "--version"], config, ProviderContext(tmp_path, None, None)
    )[0] == str(binary)


def test_native_provider_selects_isa_specific_binary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("LLAMA_CPP_AMX_BIN_DIR", str(tmp_path))
    config = _config(cpu_isa_target="amx", cpu_features_required=("amx_int8",))

    wrapped = NativeProvider().wrap(
        ["llama-server", "--version"], config, ProviderContext(tmp_path, None, None)
    )

    assert wrapped[0] == str(binary)


def _vllm_cpu_config(**overrides: Any):
    values: dict[str, Any] = {
        "backend": "vllm",
        "hardware_type": "cpu",
        "artifact_format": "huggingface",
        "artifact_path": None,
        "gpu_layers": None,
        "batch_size": None,
        "ubatch_size": None,
        "parallel_slots": None,
        "thread_count": None,
        "cpu_isa_target": "amx",
        "cpu_features_required": ("amx_tile", "amx_bf16"),
        "vllm_cpu_kvcache_space_gib": 8,
        "vllm_cpu_omp_threads_bind": "auto",
        "vllm_cpu_num_reserved_cpu": 1,
    }
    values.update(overrides)
    return _config(**values)


def test_native_vllm_cpu_runtime_uses_dedicated_binary_and_profile_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "vllm"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("VLLM_CPU_BIN", str(binary))
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")
    config = _vllm_cpu_config()

    wrapped = NativeProvider().wrap(
        ["vllm", "serve", "test/model"], config, ProviderContext(tmp_path, None, None)
    )

    assert wrapped[:5] == [
        "/usr/bin/env",
        "VLLM_CPU_KVCACHE_SPACE=8",
        "VLLM_CPU_OMP_THREADS_BIND=auto",
        "VLLM_CPU_NUM_OF_RESERVED_CPU=1",
        str(binary),
    ]


def test_native_vllm_cpu_flat_runtime_binds_requested_memory_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "vllm"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("VLLM_CPU_BIN", str(binary))
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")
    config = _vllm_cpu_config(
        memory_mode="flat", memory_type="hbm2e", memory_binding="2,3"
    )

    wrapped = NativeProvider().wrap(
        ["vllm", "serve", "test/model"], config, ProviderContext(tmp_path, None, None)
    )

    assert wrapped[:3] == ["/usr/bin/numactl", "--membind", "2,3"]
    assert wrapped[3] == "/usr/bin/env"
    assert str(binary) in wrapped


def test_interleave_memory_policy_stripes_pages_across_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default first-touch hands every page to the socket that wins the load race,
    # and the winner changes from launch to launch. "all" is the whole-machine
    # form used when no explicit binding narrows the set.
    binary = tmp_path / "vllm"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("VLLM_CPU_BIN", str(binary))
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")
    config = _vllm_cpu_config(memory_mode="cache", memory_type="hbm2e+ddr5",
                              memory_policy="interleave")

    wrapped = NativeProvider().wrap(
        ["vllm", "serve", "test/model"], config, ProviderContext(tmp_path, None, None)
    )

    assert wrapped[:3] == ["/usr/bin/numactl", "--interleave", "all"]


def test_interleave_memory_policy_respects_an_explicit_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "vllm"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("VLLM_CPU_BIN", str(binary))
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")
    config = _vllm_cpu_config(
        memory_mode="flat", memory_type="hbm2e", memory_binding="2,3",
        memory_policy="interleave",
    )

    wrapped = NativeProvider().wrap(
        ["vllm", "serve", "test/model"], config, ProviderContext(tmp_path, None, None)
    )

    assert wrapped[:3] == ["/usr/bin/numactl", "--interleave", "2,3"]


def test_docker_vllm_cpu_flat_runtime_binds_requested_memory_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_CPU_DOCKER_IMAGE", "docker.io/vllm/vllm-openai@sha256:" + "b" * 64)
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")
    config = _vllm_cpu_config(
        provider="docker", memory_mode="flat", memory_type="hbm2e", memory_binding="2,3"
    )

    wrapped = DockerProvider().wrap(
        ["vllm", "serve", "test/model"],
        config,
        ProviderContext(tmp_path, None, None, "bench-container"),
    )

    assert wrapped[wrapped.index("--cpuset-mems") + 1] == "2,3"


def test_container_providers_pin_images_and_build_safe_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(provider="docker")
    monkeypatch.setenv(
        "LLAMA_CPP_DOCKER_IMAGE", "ghcr.io/ggml-org/llama.cpp:server@sha256:" + "a" * 64
    )
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")
    context = ProviderContext(tmp_path, Path("/models"), None, "bench-container")
    docker = DockerProvider().wrap(
        ["llama-server", "--host", "127.0.0.1"], config, context, server=True
    )
    assert docker[:3] == ["docker", "run", "--rm"]
    assert "--entrypoint" in docker
    assert docker[docker.index("--host") + 1] == "0.0.0.0"

    image = tmp_path / "llama.sif"
    image.write_bytes(b"sif")
    monkeypatch.setenv("LLAMA_CPP_APPTAINER_IMAGE", str(image))
    apptainer = ApptainerProvider().wrap(
        ["llama-server", "--version"],
        _config(provider="apptainer"),
        context,
    )
    assert apptainer[:3] == ["/usr/bin/apptainer", "exec", "--cleanenv"]
    library_path_index = apptainer.index("LD_LIBRARY_PATH=/app")
    assert apptainer[library_path_index - 1 : library_path_index + 1] == [
        "--env",
        "LD_LIBRARY_PATH=/app",
    ]


def test_docker_uses_container_memory_node_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(provider="docker", memory_binding="2,3")
    monkeypatch.setenv(
        "LLAMA_CPP_DOCKER_IMAGE", "ghcr.io/ggml-org/llama.cpp:server@sha256:" + "a" * 64
    )
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")

    wrapped = DockerProvider().wrap(
        ["llama-server", "--host", "127.0.0.1"],
        config,
        ProviderContext(tmp_path, Path("/models"), None, "bench-container"),
        server=True,
    )

    assert wrapped[:3] == ["docker", "run", "--rm"]
    assert wrapped[wrapped.index("--cpuset-mems") + 1] == "2,3"


class _Tokenizer:
    all_special_ids = [0]
    vocab_size = 32

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return " ".join(str(value) for value in token_ids)

    def encode(self, text: str, **_: Any) -> list[int]:
        return [int(value) for value in text.split()]


def test_workload_manifest_is_deterministic_and_exact_length() -> None:
    config = _config()
    first = build_workload_manifest(config, tokenizer=_Tokenizer())
    second = build_workload_manifest(config, tokenizer=_Tokenizer())

    assert first == second
    assert {prompt["token_count"] for prompt in first["prompts"]} == {4}
    assert len(first["sha256"]) == 64

    native = build_native_workload_manifest(
        _config(benchmark_type="offline", temperature=None, top_p=None, ignore_eos=None)
    )
    assert native["kind"] == "backend_native_shape"
    assert native["number_of_prompts"] == 2


def test_shared_api_client_normalizes_streaming_evidence() -> None:
    config = _config()
    manifest = build_workload_manifest(config, tokenizer=_Tokenizer())

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["ignore_eos"] is True
        body = "".join(
            (
                'data: {"choices":[{"text":"a"}]}\n\n',
                'data: {"choices":[{"text":"b"}],"usage":'
                '{"prompt_tokens":4,"completion_tokens":2}}\n\n',
                "data: [DONE]\n\n",
            )
        )
        return httpx.Response(200, text=body)

    raw = asyncio.run(
        run_api_benchmark(
            config,
            manifest,
            base_url="http://test",
            transport=httpx.MockTransport(handler),
        )
    )

    assert raw["successful_requests"] == 2
    assert raw["failed_requests"] == 0
    assert raw["actual_input_tokens"] == 8
    assert raw["actual_output_tokens"] == 4
    assert raw["mean_ttft_ms"] is not None
    assert raw["workload_manifest_sha256"] == manifest["sha256"]


def test_shared_api_client_records_malformed_sse_as_request_failures() -> None:
    config = _config()
    manifest = build_workload_manifest(config, tokenizer=_Tokenizer())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: {malformed-json}\n\n")

    raw = asyncio.run(
        run_api_benchmark(
            config,
            manifest,
            base_url="http://test",
            transport=httpx.MockTransport(handler),
        )
    )

    assert raw["successful_requests"] == 0
    assert raw["failed_requests"] == 2
    assert all(record["error"].startswith("JSONDecodeError") for record in raw["requests"])


def test_llamacpp_native_parser_uses_combined_prompt_generation_row(tmp_path: Path) -> None:
    raw = tmp_path / "llama.json"
    raw.write_text(
        json.dumps(
            [
                {
                    "n_prompt": 4,
                    "n_gen": 2,
                    "avg_ns": 1_000_000_000,
                    "avg_ts": 6,
                    "samples_ns": [900_000_000, 1_100_000_000],
                }
            ]
        ),
        encoding="utf-8",
    )

    metrics = read_raw_metrics(raw, backend="llamacpp", measurement_method="llamacpp_bench")

    assert metrics["successful_requests"] == 2
    assert metrics["actual_input_tokens"] == 8
    assert metrics["actual_output_tokens"] == 4
    assert metrics["total_throughput_tokens_per_second"] == pytest.approx(6)

    with pytest.raises(ResultError, match="no raw metric parser registered"):
        read_raw_metrics(raw, backend="llamacpp", measurement_method="vllm_bench_throughput")


@pytest.mark.parametrize(
    ("filename", "backend", "method", "expected_requests"),
    (
        ("raw_vllm_throughput.json", "vllm", "vllm_bench_throughput", 4),
        ("raw_llamacpp_bench.json", "llamacpp", "llamacpp_bench", 2),
        ("raw_shared_harness.json", "llamacpp", "shared_openai_streaming", 2),
    ),
)
def test_registered_metric_parsers_use_explicit_raw_fixtures(
    filename: str, backend: str, method: str, expected_requests: int
) -> None:
    metrics = read_raw_metrics(
        ROOT / "tests/fixtures" / filename,
        backend=backend,
        measurement_method=method,
    )
    assert metrics["successful_requests"] == expected_requests


def test_telemetry_collectors_keep_process_docker_gpu_and_rapl_scopes_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "llm_bench.telemetry._process_sample", lambda pid, tracked=None: (pid / 10, 12.0)
    )
    monkeypatch.setattr("llm_bench.telemetry._docker_sample", lambda name: (3.0, len(name)))
    monkeypatch.setattr("llm_bench.telemetry._gpu_sample", lambda: (4.0, 5.0, 6.0))
    energies = iter((10.0, 12.0))
    monkeypatch.setattr("llm_bench.telemetry._rapl_energy", lambda: next(energies))

    assert ProcessTelemetryCollector(20).sample(1.0)["cpu_utilization_percent"] == 2.0
    assert DockerTelemetryCollector("container").sample(1.0)["rss_mib"] == 9
    assert NvidiaTelemetryCollector().sample(1.0)["power_draw_watts"] == 6.0
    rapl = RaplTelemetryCollector()
    assert rapl.sample(1.0)["package_power_watts"] is None
    assert rapl.sample(2.0)["package_power_watts"] == 2.0


def _backend_summary(backend: str) -> dict[str, Any]:
    return {
        "backend": backend,
        "benchmark_type": "serving",
        "model_id": "model",
        "resolved_model_revision": "a" * 40,
        "tokenizer_id": "model",
        "resolved_tokenizer_revision": "a" * 40,
        "model_precision": "float16",
        "quantization": None,
        "model_artifact_source_revision": "a" * 40,
        "execution_provider": "native",
        "hardware_type": "gpu",
        "accelerator_name": "GPU",
        "accelerator_count": 1,
        "cpu_model": "CPU",
        "socket_count": 1,
        "numa_node_count": 1,
        "measurement_method": "shared_openai_streaming",
        "measurement_scope": "client_observed_end_to_end",
        "workload_manifest_sha256": "b" * 64,
        "input_length": 4,
        "output_length": 2,
        "actual_input_tokens": 8,
        "actual_output_tokens": 4,
        "generation_config": "controlled",
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "number_of_requests": 2,
        "request_rate": 1.0,
        "maximum_concurrency": 1,
        "seed": 1,
        "warmup_runs": 1,
        "repetitions": 1,
        "status": "completed",
    }


def test_backend_treatment_requires_shared_controlled_evidence() -> None:
    left = _backend_summary("vllm")
    right = _backend_summary("llamacpp")
    assert check_backend_treatment_compatibility(left, right)["status"] == "compatible"

    right["measurement_method"] = "llamacpp_bench"
    result = check_backend_treatment_compatibility(left, right)
    assert result["status"] == "incompatible"
    assert any(item["field"] == "measurement_method" for item in result["evidence"])


def test_model_preparation_records_revision_hashes_and_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = load_model_manifest(ROOT / "configs/models/llama32_1b.yaml")
    snapshot = tmp_path / "cache/models--test/snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    converter = tmp_path / "convert.py"
    converter.write_text("# converter", encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP_CONVERT_SCRIPT", str(converter))
    monkeypatch.setattr("llm_bench.preparation._snapshot", lambda *args, **kwargs: snapshot)

    def fake_run(command: list[str], log: Path) -> None:
        del log
        if "--outfile" in command:
            output = Path(command[command.index("--outfile") + 1])
        else:
            output = Path(command[-2])
        output.write_bytes(("artifact:" + command[-1]).encode())

    monkeypatch.setattr("llm_bench.preparation._run", fake_run)
    manifest = prepare_model(
        model,
        provider_name="native",
        variants=("f16", "q8_0"),
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
    )

    assert manifest["resolved_revision"] == "c" * 40
    assert set(manifest["artifacts"]) == {"f16", "q8_0"}
    assert all(len(value["sha256"]) == 64 for value in manifest["artifacts"].values())
    assert manifest["tool_provenance"]["converter"]["sha256"]

    calls = 0

    def fail_if_called(command: list[str], log: Path) -> None:
        nonlocal calls
        del command, log
        calls += 1

    monkeypatch.setattr("llm_bench.preparation._run", fail_if_called)
    reused = prepare_model(
        model,
        provider_name="native",
        variants=("f16", "q8_0"),
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
    )
    assert calls == 0
    assert reused["artifacts"]["f16"]["reused"] is True
    assert reused["artifacts"]["q8_0"]["reused"] is True

    vllm = load_composed_experiment(
        ROOT / "configs/models/llama32_1b.yaml",
        ROOT / "configs/workloads/fixed_32_32.yaml",
        ROOT / "configs/profiles/vllm_gpu.yaml",
        provider="native",
        variant="f16",
        artifact_root=tmp_path / "artifacts",
    )
    assert vllm.model_revision == "c" * 40


def test_vllm_bf16_preparation_only_caches_the_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # llama31_8b's bf16 is vLLM-only; llama32_1b's also carries a GGUF artifact.
    model = load_model_manifest(ROOT / "configs/models/llama31_8b.yaml")
    snapshot = tmp_path / "cache/models--test/snapshots" / ("d" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setattr("llm_bench.preparation._snapshot", lambda *args, **kwargs: snapshot)

    def fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("snapshot-only preparation must not run GGUF tools")

    monkeypatch.setattr("llm_bench.preparation._run", fail_if_called)

    manifest = prepare_model(
        model,
        provider_name="native",
        variants=("bf16",),
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
    )

    assert manifest["resolved_revision"] == "d" * 40
    assert manifest["snapshot_path"] == str(snapshot)
    assert manifest["tool_provenance"] == {"operation": "huggingface_snapshot_only"}
    assert manifest["artifacts"] == {}


def test_bf16_gguf_is_converted_from_the_snapshot_not_from_the_f16_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = load_model_manifest(ROOT / "configs/models/llama32_1b.yaml")
    snapshot = tmp_path / "cache/models--test/snapshots" / ("e" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setattr("llm_bench.preparation._snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setenv("LLAMA_CPP_CONVERT_SCRIPT", str(tmp_path / "convert_hf_to_gguf.py"))
    (tmp_path / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")

    commands: list[list[str]] = []

    def record(command: list[str], log: Path) -> None:
        commands.append(list(command))
        # The converter writes to the temporary path that follows --outfile.
        Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF")

    monkeypatch.setattr("llm_bench.preparation._run", record)

    manifest = prepare_model(
        model,
        provider_name="native",
        variants=("bf16",),
        artifact_root=tmp_path / "artifacts",
        cache_root=tmp_path / "cache",
    )

    assert len(commands) == 1, "bf16 must not also build the f16 base"
    assert commands[0][-2:] == ["--outtype", "bf16"]
    assert str(snapshot) in commands[0]
    assert "llama-quantize" not in " ".join(commands[0])
    assert set(manifest["artifacts"]) == {"bf16"}
    record_bf16 = manifest["artifacts"]["bf16"]
    assert record_bf16["precision"] == "bfloat16"
    assert record_bf16["quantization"] is None
    assert record_bf16["path"].endswith("llama32-1b-instruct-bf16.gguf")


def test_dry_run_contains_provider_wrapped_server_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "LLAMA_CPP_DOCKER_IMAGE", "ghcr.io/ggml-org/llama.cpp:server@sha256:" + "d" * 64
    )
    monkeypatch.setattr("llm_bench.providers.shutil.which", lambda name: f"/usr/bin/{name}")
    report = validate_runtime(
        _config(provider="docker", artifact_path=str(tmp_path / "missing.gguf")),
        require_artifact=False,
    )

    assert report["server_command"][:3] == ["docker", "run", "--rm"]
    assert "--entrypoint" in report["server_command"]
    assert report["resolved_port"] == 8000


def test_mocked_end_to_end_runner_writes_schema_two_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        backend="vllm",
        benchmark_type="offline",
        artifact_format="huggingface",
        artifact_path=None,
        artifact_manifest_path=None,
        hardware_type="gpu",
        temperature=None,
        top_p=None,
        ignore_eos=None,
        thread_count=None,
        gpu_layers=None,
        batch_size=None,
        ubatch_size=None,
        parallel_slots=None,
        repetitions=1,
    )
    sources = {}
    for name in ("model", "workload", "profile"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"kind: {name}\n", encoding="utf-8")
        sources[name] = path

    class FakeRunner:
        measurement_method = "vllm_bench_throughput"
        measurement_scope = "vllm_bench_throughput_native"

        def run(self, context: Any):
            raw = context.run_directory / "raw_backend_output.json"
            raw.write_text(
                json.dumps(
                    {
                        "elapsed_time": 1.0,
                        "num_requests": 2,
                        "total_input_tokens": 8,
                        "total_generated_tokens": 4,
                        "requests_per_second": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            return [raw], [], [], None

    monkeypatch.setattr("llm_bench.runner.validate_runtime", lambda config: {})
    monkeypatch.setattr(
        "llm_bench.runner.collect_metadata",
        lambda *args, **kwargs: {
            "run_id": "mocked-run",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "benchmark_type": "offline",
            "backend": "vllm",
            "resolved_model_revision": "a" * 40,
            "resolved_tokenizer_revision": "a" * 40,
            "software_versions": {},
            "warnings": [],
        },
    )
    monkeypatch.setitem(runner_module.RUNNER_REGISTRY, "offline", FakeRunner)
    monkeypatch.delenv("MODEL_ARTIFACT_ROOT", raising=False)

    destination = run_experiment(
        config,
        source_paths=sources,
        results_root=tmp_path / "results",
        run_directory=tmp_path / "results/run",
        repository=ROOT,
    )

    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "2.0"
    assert summary["status"] == "completed"
    assert set(summary["configuration_hashes"]) == {"model", "workload", "profile"}
    assert (destination / "resolved_experiment.yaml").is_file()
    assert (destination / "measurements.json").is_file()
