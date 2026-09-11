from __future__ import annotations

import pytest

from llm_bench.config import ConfigurationError, config_from_mapping
from llm_bench.hardware import (
    NumaNode,
    detect_xeon_max_hbm_mode,
    inspect_hardware,
    resolve_memory_binding,
    validate_hardware,
)


def _config(**overrides: object):
    values: dict[str, object] = {
        "experiment_name": "hardware-test",
        "backend": "llamacpp",
        "model_id": "test/model",
        "hardware_type": "cpu",
        "cpu_isa_target": "amx",
        "cpu_features_required": ["amx_int8", "amx_bf16"],
        "memory_type": "hbm2e+ddr5",
        "memory_mode": "cache",
    }
    values.update(overrides)
    return config_from_mapping(values)


def test_xeon_max_flat_mode_has_memory_only_numa_nodes() -> None:
    nodes = (
        NumaNode(0, "0-55", 512 * 1024**2),
        NumaNode(1, "56-111", 512 * 1024**2),
        NumaNode(2, "", 64 * 1024**2),
        NumaNode(3, "", 64 * 1024**2),
    )

    assert detect_xeon_max_hbm_mode("Intel(R) Xeon(R) CPU Max 9480", nodes, 2) == (
        "flat",
        "xeon_max_memory_only_numa_nodes",
    )


def test_xeon_max_cache_mode_hides_hbm_from_numa() -> None:
    nodes = (
        NumaNode(0, "0-55", 512 * 1024**2),
        NumaNode(1, "56-111", 512 * 1024**2),
    )

    assert detect_xeon_max_hbm_mode("Intel(R) Xeon(R) CPU Max 9480", nodes, 2) == (
        "cache",
        "xeon_max_no_memory_only_numa_nodes",
    )


def test_hardware_report_keeps_requested_and_detected_evidence_separate(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_bench.hardware.lscpu_values",
        lambda: {
            "Architecture": "x86_64",
            "Model name": "Intel(R) Xeon(R) CPU Max 9480",
            "Socket(s)": "2",
            "Flags": "avx avx2 avx512f amx_tile amx_int8 amx_bf16",
        },
    )
    monkeypatch.setattr(
        "llm_bench.hardware.numa_nodes",
        lambda: (
            NumaNode(0, "0-55", 512 * 1024**2),
            NumaNode(1, "56-111", 512 * 1024**2),
        ),
    )

    report = inspect_hardware(_config())

    assert report["cpu_isa_verified"] is True
    assert report["memory_mode_requested"] == "cache"
    assert report["memory_mode_detected"] == "cache"
    assert report["memory_mode_verified"] is True


def test_symbolic_hbm_binding_resolves_from_flat_mode_evidence() -> None:
    report = {
        "hbm_numa_nodes": [2, 3],
        "ddr_numa_nodes": [0, 1],
    }

    assert resolve_memory_binding(
        _config(memory_mode="flat", memory_type="hbm2e", memory_binding="hbm"),
        report,
    ) == "2,3"


def test_runtime_validation_rejects_hbm_mode_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_bench.hardware.inspect_hardware",
        lambda config: {
            "cpu_features_required": ["amx_tile"],
            "cpu_features_missing": [],
            "memory_mode_requested": config.memory_mode,
            "memory_mode_detected": "flat",
            "memory_mode_detection_method": "test",
            "numa_nodes": [],
            "hbm_numa_nodes": [],
            "ddr_numa_nodes": [],
        },
    )

    with pytest.raises(ConfigurationError, match="HBM mode mismatch"):
        validate_hardware(_config(memory_mode="cache"))
