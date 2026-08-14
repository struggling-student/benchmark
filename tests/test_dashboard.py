from __future__ import annotations

from pathlib import Path

import pytest

from llm_bench.config import config_from_mapping
from llm_bench.results import create_summary, write_summary

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _config(benchmark_type: str, model: str) -> object:
    values: dict[str, object] = {
        "experiment_name": f"{model}-{benchmark_type}",
        "benchmark_type": benchmark_type,
        "backend": "vllm",
        "model_id": model,
        "dtype": "float16",
        "number_of_prompts": 2,
        "input_length": 16,
        "output_length": 8,
        "max_model_len": 32,
        "repetitions": 1,
        "warmup_runs": 0,
    }
    if benchmark_type == "serving":
        values.update(
            {
                "generation_config": "vllm",
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "request_rate": 2.0,
                "maximum_concurrency": 2,
            }
        )
    return config_from_mapping(values)


def _write_run(root: Path, benchmark_type: str, model: str, index: int) -> None:
    metadata = {
        "run_id": f"run-{index}",
        "timestamp": f"2026-07-2{index}T12:00:00+00:00",
        "backend_version": "1",
        "hardware_type": "gpu",
        "accelerator_name": "test-gpu",
        "accelerator_count": 1,
        "cpu_model": "test-cpu",
        "socket_count": 1,
        "numa_node_count": 1,
        "software_versions": {},
    }
    summary = create_summary(_config(benchmark_type, model), metadata)
    summary.update(
        {
            "status": "completed",
            "measured_repetitions": 1,
            "warnings": [],
            "request_throughput_requests_per_second": float(index + 1),
            "output_throughput_tokens_per_second": float((index + 1) * 8),
        }
    )
    write_summary(root / f"run-{index}" / "summary.json", summary)


def test_streamlit_dashboard_renders_all_views(tmp_path: Path) -> None:
    _write_run(tmp_path, "smoke", "model-1b", 0)
    _write_run(tmp_path, "offline", "model-8b", 1)
    _write_run(tmp_path, "serving", "model-1b", 2)
    script = f"from llm_bench.dashboard import run_dashboard\nrun_dashboard({str(tmp_path)!r})\n"

    app = AppTest.from_string(script, default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "LLM benchmark studio"
    assert app.sidebar.segmented_control[0].value == "Light"

    app.sidebar.segmented_control[0].set_value("Dark")
    app.run(timeout=10)
    assert not app.exception
    assert app.sidebar.segmented_control[0].value == "Dark"
    assert any("--bench-bg: #0A0F1D" in item.value for item in app.markdown)

    for page in ("Run detail", "Explorer", "Memory study", "Compare"):
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=10)
        assert not app.exception


def test_empty_results_root_defaults_to_full_simulated_study(tmp_path: Path) -> None:
    script = f"from llm_bench.dashboard import run_dashboard\nrun_dashboard({str(tmp_path)!r})\n"

    app = AppTest.from_string(script, default_timeout=20).run()

    assert not app.exception
    assert app.sidebar.selectbox[0].value == "Demo study"
    assert app.metric[0].value == "17"
    assert not list(tmp_path.iterdir())

    for page in ("Run detail", "Explorer", "Memory study", "Compare"):
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=20)
        assert not app.exception

    assert not list(tmp_path.iterdir())


def test_omitted_results_root_opens_demo_study() -> None:
    script = "from llm_bench.dashboard import run_dashboard\nrun_dashboard()\n"

    app = AppTest.from_string(script, default_timeout=20).run()

    assert not app.exception
    assert app.sidebar.selectbox[0].options == ["Demo study"]
    assert app.sidebar.selectbox[0].value == "Demo study"
    assert app.metric[0].value == "17"
