from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.config import (
    ConfigurationError,
    load_composed_experiment,
    load_execution_profile,
    load_experiment,
    load_model_manifest,
    load_workload,
    resolve_experiment,
)

ROOT = Path(__file__).parents[1]


def test_shipped_manifests_compose_complete_f16_matrix() -> None:
    models = [load_model_manifest(path) for path in (ROOT / "configs/models").glob("*.yaml")]
    workloads = [load_workload(path) for path in (ROOT / "configs/workloads").glob("*.yaml")]
    profiles = [
        load_execution_profile(path) for path in (ROOT / "configs/profiles").glob("*.yaml")
    ]

    resolved = [
        resolve_experiment(
            model,
            workload,
            profile,
            provider="native",
            variant="f16",
            artifact_root="/opt/llm-bench-artifacts",
        )
        for model in models
        for workload in workloads
        for profile in profiles
    ]

    assert len(resolved) == 18
    assert {item.backend for item in resolved} == {"vllm", "llamacpp"}
    assert {item.benchmark_type for item in resolved} == {"smoke", "offline", "serving"}
    assert all(item.tokenizer == item.model_id for item in resolved)


def test_quantized_variants_are_llamacpp_only() -> None:
    model = load_model_manifest(ROOT / "configs/models/llama32_1b.yaml")
    workload = load_workload(ROOT / "configs/workloads/offline.yaml")
    llama = load_execution_profile(ROOT / "configs/profiles/llamacpp_cpu.yaml")
    vllm = load_execution_profile(ROOT / "configs/profiles/vllm_gpu.yaml")

    config = resolve_experiment(
        model,
        workload,
        llama,
        provider="docker",
        variant="q4_k_m",
        artifact_root="/models",
    )
    assert config.quantization == "Q4_K_M"
    assert config.artifact_path == "/models/llama32_1b/llama32-1b-instruct-q4_k_m.gguf"

    with pytest.raises(ConfigurationError, match="does not support backend"):
        resolve_experiment(model, workload, vllm, provider="native", variant="q4_k_m")


def test_legacy_monolithic_experiment_is_rejected(tmp_path: Path) -> None:
    legacy = tmp_path / "experiment.yaml"
    legacy.write_text("backend: vllm\nmodel_id: model\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="legacy monolithic"):
        load_experiment(legacy)


def test_unknown_workload_field_is_rejected(tmp_path: Path) -> None:
    workload = tmp_path / "workload.yaml"
    workload.write_text(
        """
schema_version: "2.0"
workload_key: smoke
benchmark_type: smoke
seed: 1
number_of_prompts: 1
input_length: 8
output_length: 4
repetitions: 1
warmup_runs: 0
telemetry_interval_ms: 1000
temperature: 0.0
top_p: 1.0
ignore_eos: true
request_rate: null
maximum_concurrency: 1
reqeust_rate: 2
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="reqeust_rate"):
        load_workload(workload)


def test_serving_workload_requires_rate(tmp_path: Path) -> None:
    original = (ROOT / "configs/workloads/serving.yaml").read_text(encoding="utf-8")
    workload = tmp_path / "serving.yaml"
    workload.write_text(
        original.replace("request_rate: 4.0", "request_rate: null"), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="request_rate"):
        load_workload(workload)


def test_profile_rejects_cpu_gpu_offload(tmp_path: Path) -> None:
    original = (ROOT / "configs/profiles/llamacpp_cpu.yaml").read_text(encoding="utf-8")
    profile = tmp_path / "profile.yaml"
    profile.write_text(original.replace("gpu_layers: 0", "gpu_layers: 4"), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="CPU llama.cpp"):
        load_execution_profile(profile)


def test_model_context_must_fit_workload(tmp_path: Path) -> None:
    original = (ROOT / "configs/models/llama32_1b.yaml").read_text(encoding="utf-8")
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        original.replace("max_model_len: 8192", "max_model_len: 100"), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match=r"input_length \+ output_length"):
        load_composed_experiment(
            model_path,
            ROOT / "configs/workloads/serving.yaml",
            ROOT / "configs/profiles/vllm_gpu.yaml",
            provider="native",
            variant="f16",
        )
