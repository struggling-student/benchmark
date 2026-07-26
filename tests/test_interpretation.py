from __future__ import annotations

from llm_bench.interpretation import comparison_narrative, repetition_variability
from llm_bench.results import compare_summaries


def _summary(run_id: str, throughput: float) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "completed",
        "warnings": [],
        "benchmark_type": "smoke",
        "backend": "vllm",
        "backend_version": "1",
        "model_id": f"model-{run_id}",
        "model_precision": "float16",
        "input_length": 16,
        "output_length": 8,
        "max_model_len": 32,
        "generation_config": None,
        "temperature": None,
        "top_p": None,
        "ignore_eos": None,
        "number_of_requests": 2,
        "request_rate": None,
        "maximum_concurrency": None,
        "gpu_memory_utilization": 0.9,
        "seed": 42,
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "quantization": None,
        "warmup_runs": 0,
        "repetitions": 2,
        "measured_repetitions": 2,
        "telemetry_interval_ms": 1000,
        "git_commit": "abc",
        "git_dirty": False,
        "hardware_type": "gpu",
        "accelerator_name": "gpu",
        "accelerator_count": 1,
        "cpu_model": "cpu",
        "socket_count": 1,
        "numa_node_count": 1,
        "software_versions": {
            "python": "3.10",
            "llm_bench": "1",
            "vllm": "1",
            "torch": "1",
            "cuda_runtime": "1",
            "nvidia_driver": "1",
        },
        "output_throughput_tokens_per_second": throughput,
    }


def _measurements(first: float, second: float) -> dict[str, object]:
    return {
        "records": [
            {
                "status": "completed",
                "metrics": {"output_throughput_tokens_per_second": first},
            },
            {
                "status": "completed",
                "metrics": {"output_throughput_tokens_per_second": second},
            },
        ]
    }


def test_compatible_narrative_is_directional_but_not_a_score() -> None:
    left = _summary("left", 10.0)
    right = _summary("right", 12.0)
    comparison = compare_summaries(left, right)

    statements = comparison_narrative(
        comparison,
        left,
        right,
        left_measurements=_measurements(9, 11),
        right_measurements=_measurements(11, 13),
    )
    joined = " ".join(statements)

    assert comparison["compatibility"]["status"] == "compatible"
    assert "20.0% higher" in joined
    assert "Smoke runs primarily" in joined
    assert "not combined into an overall score" in joined
    assert "confidence intervals" in joined


def test_incompatible_narrative_suppresses_ratios_and_winner_language() -> None:
    left = _summary("left", 10.0)
    right = _summary("right", 12.0)
    right["input_length"] = 32
    comparison = compare_summaries(left, right)
    statements = comparison_narrative(comparison, left, right)

    assert comparison["compatibility"]["status"] == "incompatible"
    assert all(values["right_to_left_ratio"] is None for values in comparison["metrics"].values())
    assert "incompatible" in " ".join(statements)
    assert "winner" not in " ".join(statements).lower()


def test_variability_requires_two_finite_observations() -> None:
    variability = repetition_variability(
        _measurements(9, 11), "output_throughput_tokens_per_second"
    )

    assert variability is not None
    assert variability["mean"] == 10
    assert variability["standard_deviation"] > 0
