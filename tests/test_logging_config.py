from __future__ import annotations

import logging

import pytest

from llm_bench.logging_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_ENV_VAR,
    configure_logging,
    resolve_log_level,
)


def test_resolve_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
    assert resolve_log_level() == DEFAULT_LOG_LEVEL


def test_resolve_log_level_reads_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "debug")
    assert resolve_log_level() == "DEBUG"


def test_explicit_argument_overrides_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "DEBUG")
    assert resolve_log_level("warning") == "WARNING"


def test_unknown_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="log level must be one of"):
        resolve_log_level("verbose")


def test_configure_logging_sets_level_and_is_idempotent() -> None:
    configure_logging("DEBUG")
    logger = logging.getLogger("llm_bench")
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1

    configure_logging("WARNING")
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1


def test_configure_logging_does_not_propagate_to_root() -> None:
    configure_logging("INFO")
    assert logging.getLogger("llm_bench").propagate is False
