from __future__ import annotations

import logging

import pytest

from llm_bench.cli import build_parser, main
from llm_bench.logging_config import LOG_LEVEL_ENV_VAR


def test_log_level_flag_defaults_to_none() -> None:
    args = build_parser().parse_args(["compare", "left", "right", "--output-dir", "out"])
    assert args.log_level is None


def test_log_level_flag_must_precede_subcommand_and_is_validated() -> None:
    args = build_parser().parse_args(
        ["--log-level", "DEBUG", "compare", "left", "right", "--output-dir", "out"]
    )
    assert args.log_level == "DEBUG"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["compare", "--log-level", "DEBUG", "left", "right", "--output-dir", "out"]
        )


def test_main_configures_llm_bench_logger_from_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "llm_bench.cli.compare_paths",
        lambda left, right, output_dir: ({"compatibility": {"status": "compatible"}}, left, right),
    )
    main(["--log-level", "DEBUG", "compare", "left", "right", "--output-dir", "out"])
    assert logging.getLogger("llm_bench").level == logging.DEBUG


def test_cli_logger_stays_in_the_llm_bench_hierarchy() -> None:
    # scripts/run_benchmark.sh invokes `python -m llm_bench.cli`, which makes
    # this module's __name__ "__main__" at runtime. A logger bound to
    # __name__ would then sit outside the "llm_bench" logger tree and never
    # reach the handler configure_logging() installs, so cli.py must bind an
    # explicit "llm_bench.cli" name instead.
    import llm_bench.cli as cli_module

    assert cli_module.logger.name == "llm_bench.cli"
