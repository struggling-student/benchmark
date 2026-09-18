"""Global diagnostic logging for the benchmark harness.

Every module logs through the standard library `logging` package under the
`llm_bench` logger namespace, to stderr, so stdout stays reserved for the
machine-readable output the CLI already prints (JSON, result paths). The level
is decided once, globally, for a whole job:

1. an explicit ``--log-level`` CLI flag, if given;
2. the ``LLM_BENCH_LOG_LEVEL`` environment variable, so a cluster config file
   or Slurm job script can fix the level for an entire allocation without
   touching the CLI invocation;
3. ``INFO`` otherwise.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from collections.abc import Sequence
from typing import Any

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL_ENV_VAR = "LLM_BENCH_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"
_LOGGER_NAME = "llm_bench"


class _MultilineFormatter(logging.Formatter):
    """Indent embedded newlines so multi-line messages stay readable.

    A DEBUG payload (a resolved config, a wrapped command argv) is much easier
    to scan as its own indented block than as one very long single line. This
    keeps the usual one-line format for ordinary messages and, for a message
    that already contains "\\n" (see `format_json`/`format_argv` below),
    lines up every continuation line under where the first line's message
    text starts.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if "\n" not in message:
            return super().format(record)
        lines = message.split("\n")
        original_msg, original_args = record.msg, record.args
        try:
            record.msg, record.args = lines[0], None
            head = super().format(record)
        finally:
            record.msg, record.args = original_msg, original_args
        indent = " " * (len(head) - len(lines[0]))
        return "\n".join([head, *(indent + line for line in lines[1:])])


def format_json(value: Any) -> str:
    """Render a dict/list as indented JSON for a multi-line DEBUG message.

    Use as ``logger.debug("resolved config:\\n%s", format_json(config))`` —
    the leading ``\\n`` puts the (often long) block on its own indented lines
    instead of one unreadable single-line dump.
    """

    return json.dumps(value, indent=2, sort_keys=True, default=str)


def format_argv(command: Sequence[str]) -> str:
    """Render a subprocess argv as a readable, shell-quoted multi-line block.

    Every flag (a token starting with "-") is kept on the same line as the
    single value that follows it, the way one would hand-format a long shell
    command, so a provider-wrapped invocation (numactl + apptainer/docker +
    the backend binary and its flags) reads as one line per option instead of
    one very long single line, or one token per line with flags and values
    torn apart.
    """

    tokens = [shlex.quote(str(part)) for part in command]
    lines: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        has_value = (
            token.startswith("-")
            and index + 1 < len(tokens)
            and not tokens[index + 1].startswith("-")
        )
        if has_value:
            lines.append(f"{token} {tokens[index + 1]}")
            index += 2
        else:
            lines.append(token)
            index += 1
    return " \\\n  ".join(lines)


def resolve_log_level(explicit: str | None = None) -> str:
    """Resolve the effective level name from an explicit value, env var, or default."""

    candidate = explicit or os.environ.get(LOG_LEVEL_ENV_VAR) or DEFAULT_LOG_LEVEL
    normalized = candidate.strip().upper()
    if normalized not in LOG_LEVELS:
        raise ValueError(
            f"log level must be one of {', '.join(LOG_LEVELS)}; got {candidate!r}"
        )
    return normalized


def configure_logging(explicit: str | None = None) -> str:
    """Configure the llm_bench logger namespace and return the level applied.

    Idempotent: safe to call more than once per process (tests invoke the CLI
    entry point in-process), and never touches the root logger so embedding
    applications keep their own configuration.
    """

    level_name = resolve_log_level(explicit)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level_name)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler()  # defaults to stderr
    handler.setFormatter(
        _MultilineFormatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    return level_name
