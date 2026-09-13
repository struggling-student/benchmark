from __future__ import annotations

import logging
from typing import Any

import pytest

from llm_bench.cli import _dashboard, build_parser, main
from llm_bench.logging_config import LOG_LEVEL_ENV_VAR


def test_dashboard_command_defaults_to_demo_and_minimal_toolbar(
    monkeypatch: Any,
) -> None:
    args = build_parser().parse_args(["dashboard"])
    captured: list[str] = []

    def fake_call(command: list[str]) -> int:
        captured.extend(command)
        return 0

    monkeypatch.setattr("llm_bench.cli.subprocess.call", fake_call)

    assert args.results_root is None
    assert _dashboard(args) == 0
    assert captured[captured.index("--client.toolbarMode") + 1] == "minimal"
    assert "--results-root" not in captured


def test_log_level_flag_defaults_to_none() -> None:
    args = build_parser().parse_args(["dashboard"])
    assert args.log_level is None


def test_log_level_flag_must_precede_subcommand_and_is_validated() -> None:
    args = build_parser().parse_args(["--log-level", "DEBUG", "dashboard"])
    assert args.log_level == "DEBUG"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["dashboard", "--log-level", "DEBUG"])


def test_main_configures_llm_bench_logger_from_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
    monkeypatch.setattr("llm_bench.cli.subprocess.call", lambda command: 0)
    main(["--log-level", "DEBUG", "dashboard"])
    assert logging.getLogger("llm_bench").level == logging.DEBUG

