from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.config import ConfigurationError, config_from_mapping, load_experiment


def valid_mapping(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "experiment_name": "llama32_1b_serving",
        "benchmark_type": "serving",
        "backend": "vllm",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "tokenizer_id": None,
        "number_of_prompts": 16,
        "input_length": 128,
        "output_length": 32,
        "generation_config": "vllm",
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "request_rate": 2.0,
        "maximum_concurrency": 4,
        "repetitions": 2,
        "warmup_runs": 1,
    }
    values.update(overrides)
    return values


def test_loads_valid_yaml_and_resolves_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
experiment_name: llama32_1b_offline
benchmark_type: offline
backend: vllm
model_id: meta-llama/Llama-3.2-1B-Instruct
number_of_prompts: 8
input_length: 64
output_length: 16
repetitions: 1
warmup_runs: 0
""".lstrip(),
        encoding="utf-8",
    )

    config = load_experiment(path)

    assert config.model_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert config.tokenizer == config.model_id
    assert config.resolved_benchmark_type == "offline"


def test_rejects_unknown_field_with_readable_name() -> None:
    with pytest.raises(ConfigurationError, match=r"unknown.*reqeust_rate"):
        config_from_mapping(valid_mapping(reqeust_rate=2.0))


def test_rejects_missing_required_field() -> None:
    values = valid_mapping()
    del values["model_id"]

    with pytest.raises(ConfigurationError, match=r"missing required field.*model_id"):
        config_from_mapping(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("number_of_prompts", 0),
        ("input_length", -1),
        ("repetitions", True),
        ("warmup_runs", -1),
        ("gpu_memory_utilization", 1.1),
        ("temperature", -0.1),
        ("top_p", 1.1),
        ("ignore_eos", 1),
    ],
)
def test_rejects_invalid_ranges(field: str, value: object) -> None:
    with pytest.raises(ConfigurationError, match=field):
        config_from_mapping(valid_mapping(**{field: value}))


def test_serving_requires_rate_and_concurrency() -> None:
    with pytest.raises(ConfigurationError, match=r"serving experiments require.*request_rate"):
        config_from_mapping(valid_mapping(request_rate=None))


def test_serving_requires_explicit_backend_generation_defaults() -> None:
    with pytest.raises(ConfigurationError, match=r"generation_config must be 'vllm'"):
        config_from_mapping(valid_mapping(generation_config="auto"))


def test_serving_requires_fixed_requested_output_length() -> None:
    with pytest.raises(ConfigurationError, match=r"ignore_eos must be true"):
        config_from_mapping(valid_mapping(ignore_eos=False))


def test_rejects_backend_not_yet_implemented() -> None:
    with pytest.raises(ConfigurationError, match=r"backend must be 'vllm'"):
        config_from_mapping(valid_mapping(backend="future-cpu-backend"))


def test_offline_rejects_unapplied_serving_controls() -> None:
    with pytest.raises(ConfigurationError, match=r"offline experiments require.*null"):
        config_from_mapping(
            valid_mapping(
                benchmark_type="offline",
                experiment_name="llama32_1b_offline",
                request_rate=2.0,
                maximum_concurrency=None,
            )
        )


def test_infers_type_from_unambiguous_experiment_name() -> None:
    config = config_from_mapping(valid_mapping(benchmark_type=None))

    assert config.resolved_benchmark_type == "serving"
