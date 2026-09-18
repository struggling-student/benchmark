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

    assert (len(models), len(workloads), len(profiles)) == (2, 10, 13)
    assert len(resolved) == len(models) * len(workloads) * len(profiles)
    assert {item.backend for item in resolved} == {"vllm", "llamacpp"}
    assert {item.benchmark_type for item in resolved} == {"offline"}
    assert {item.max_model_len for item in resolved} == {131072}
    assert all(item.tokenizer == item.model_id for item in resolved)


def test_literature_matched_workloads_preserve_reported_token_shapes() -> None:
    expected = {
        "fixed_32_32": (32, 32),
        "fixed_128_32": (128, 32),
        "fixed_256_32": (256, 32),
        "fixed_512_32": (512, 32),
        "fixed_1024_32": (1024, 32),
        "fixed_128_128": (128, 128),
        "fixed_512_8": (512, 8),
        "fixed_512_128": (512, 128),
        "fixed_1024_128": (1024, 128),
        "fixed_30000_10000": (30000, 10000),
    }

    actual = {}
    for path in (ROOT / "configs/workloads").glob("*.yaml"):
        workload = load_workload(path)
        actual[workload.workload_key] = (workload.input_length, workload.output_length)

    assert actual == expected


def test_literature_results_document_covers_every_shipped_workload() -> None:
    documentation = (ROOT / "docs/06_LITERATURE_RESULTS.md").read_text(encoding="utf-8")

    for path in (ROOT / "configs/workloads").glob("*.yaml"):
        workload = load_workload(path)
        assert f"`{workload.workload_key}`" in documentation


def test_cpu_isa_and_hbm_mode_are_part_of_resolved_profile() -> None:
    config = load_composed_experiment(
        ROOT / "configs/models/llama32_1b.yaml",
        ROOT / "configs/workloads/fixed_128_32.yaml",
        ROOT / "configs/profiles/llamacpp_cpu_amx_hbm_cache.yaml",
        provider="native",
        variant="f16",
        artifact_root="/models",
    )

    assert config.hardware_type == "cpu"
    assert config.cpu_isa_target == "amx"
    assert "amx_int8" in config.cpu_features_required
    assert config.memory_type == "hbm2e+ddr5"
    assert config.memory_mode == "cache"


def test_vllm_cpu_amx_profile_resolves_bf16_runtime_controls() -> None:
    config = load_composed_experiment(
        ROOT / "configs/models/llama32_1b.yaml",
        ROOT / "configs/workloads/fixed_32_32.yaml",
        ROOT / "configs/profiles/vllm_cpu_amx_hbm_cache.yaml",
        provider="native",
        variant="bf16",
        artifact_root="/models",
    )

    assert config.backend == "vllm"
    assert config.hardware_type == "cpu"
    assert config.dtype == "bfloat16"
    assert config.cpu_isa_target == "amx"
    assert config.tensor_parallel_size == 2
    assert config.vllm_cpu_kvcache_space_gib == 8
    assert config.vllm_cpu_omp_threads_bind == "auto"
    assert config.vllm_cpu_num_reserved_cpu == 1


@pytest.mark.parametrize("isa", ["amx", "avx512"])
def test_vllm_cpu_flat_profile_binds_a_single_memory_tier(isa: str) -> None:
    config = load_composed_experiment(
        ROOT / "configs/models/llama32_1b.yaml",
        ROOT / "configs/workloads/fixed_32_32.yaml",
        ROOT / f"configs/profiles/vllm_cpu_{isa}_hbm_flat.yaml",
        provider="native",
        variant="bf16",
        artifact_root="/models",
    )

    assert config.backend == "vllm"
    assert config.hardware_type == "cpu"
    assert config.cpu_isa_target == isa
    assert config.memory_mode == "flat"
    assert config.memory_type == "hbm2e"
    assert config.memory_binding == "hbm"


def test_vllm_gpu_profile_still_rejects_memory_binding(tmp_path: Path) -> None:
    original = (ROOT / "configs/profiles/vllm_gpu.yaml").read_text(encoding="utf-8")
    profile = tmp_path / "gpu.yaml"
    profile.write_text(
        original.replace("memory_binding: null", "memory_binding: hbm"), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="GPU profiles cannot contain CPU memory"):
        load_execution_profile(profile)


def test_vllm_profile_still_rejects_llamacpp_controls(tmp_path: Path) -> None:
    original = (
        ROOT / "configs/profiles/vllm_cpu_amx_hbm_flat.yaml"
    ).read_text(encoding="utf-8")
    profile = tmp_path / "vllm-threads.yaml"
    profile.write_text(
        original.replace("thread_count: null", "thread_count: 112"), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="llama.cpp execution controls"):
        load_execution_profile(profile)


def test_flat_single_tier_profile_requires_memory_binding(tmp_path: Path) -> None:
    original = (
        ROOT / "configs/profiles/llamacpp_cpu_amx_hbm_flat.yaml"
    ).read_text(encoding="utf-8")
    profile = tmp_path / "flat.yaml"
    profile.write_text(
        original.replace("memory_binding: hbm", "memory_binding: null"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="requires an explicit memory_binding"):
        load_execution_profile(profile)


def test_flat_profile_can_select_discovered_ddr_tier(tmp_path: Path) -> None:
    original = (
        ROOT / "configs/profiles/llamacpp_cpu_avx512_hbm_flat.yaml"
    ).read_text(encoding="utf-8")
    profile = tmp_path / "flat-ddr.yaml"
    profile.write_text(
        original.replace("memory_binding: hbm", "memory_binding: ddr").replace(
            "memory_type: hbm2e", "memory_type: ddr5"
        ),
        encoding="utf-8",
    )

    loaded = load_execution_profile(profile)

    assert loaded.memory_mode == "flat"
    assert loaded.memory_type == "ddr5"
    assert loaded.memory_binding == "ddr"


def test_quantized_variants_are_llamacpp_only() -> None:
    model = load_model_manifest(ROOT / "configs/models/llama32_1b.yaml")
    workload = load_workload(ROOT / "configs/workloads/fixed_128_32.yaml")
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
    workload = tmp_path / "serving.yaml"
    workload.write_text(
        """
schema_version: "2.0"
workload_key: serving_fixture
benchmark_type: serving
seed: 42
number_of_prompts: 2
input_length: 32
output_length: 16
temperature: 0.0
top_p: 1.0
ignore_eos: true
request_rate: null
maximum_concurrency: 1
repetitions: 1
warmup_runs: 0
telemetry_interval_ms: 1000
""".lstrip(),
        encoding="utf-8",
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
        original.replace("max_model_len: 131072", "max_model_len: 100"), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match=r"input_length \+ output_length"):
        load_composed_experiment(
            model_path,
            ROOT / "configs/workloads/fixed_128_32.yaml",
            ROOT / "configs/profiles/vllm_gpu.yaml",
            provider="native",
            variant="f16",
        )
