from __future__ import annotations

import io
import logging

import pytest

from llm_bench.logging_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_ENV_VAR,
    configure_logging,
    format_argv,
    format_json,
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


def test_format_json_pretty_prints_a_mapping() -> None:
    rendered = format_json({"b": 2, "a": 1})
    assert rendered == '{\n  "a": 1,\n  "b": 2\n}'


def test_format_argv_pairs_each_flag_with_its_value() -> None:
    rendered = format_argv(["numactl", "--membind", "0,1", "echo", "hello world"])
    assert rendered == "numactl \\\n  --membind 0,1 \\\n  echo \\\n  'hello world'"


def test_format_argv_leaves_a_bare_flag_before_another_flag_alone() -> None:
    rendered = format_argv(["apptainer", "exec", "--cleanenv", "--env", "X=1", "image"])
    assert rendered == "apptainer \\\n  exec \\\n  --cleanenv \\\n  --env X=1 \\\n  image"


def test_multiline_debug_message_indents_continuation_lines_under_the_prefix() -> None:
    configure_logging("DEBUG")
    logger = logging.getLogger("llm_bench.test_pretty_printing")
    stream = io.StringIO()
    handler = logging.getLogger("llm_bench").handlers[0]
    original_stream = handler.stream
    handler.stream = stream
    try:
        logger.debug("payload:\n%s", format_json({"a": 1, "b": 2}))
    finally:
        handler.stream = original_stream
    lines = stream.getvalue().splitlines()
    assert len(lines) == 5  # "prefix: payload:", then the four JSON lines
    head, *continuation = lines
    prefix_len = len(head) - len("payload:")
    assert head.endswith("payload:")
    for line in continuation:
        assert line[:prefix_len].strip() == ""


def test_single_line_debug_message_is_unaffected() -> None:
    configure_logging("DEBUG")
    logger = logging.getLogger("llm_bench.test_single_line")
    stream = io.StringIO()
    handler = logging.getLogger("llm_bench").handlers[0]
    original_stream = handler.stream
    handler.stream = stream
    try:
        logger.debug("plain message: %s", "value")
    finally:
        handler.stream = original_stream
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("plain message: value")
