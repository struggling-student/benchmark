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

import logging
import os

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL_ENV_VAR = "LLM_BENCH_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"
_LOGGER_NAME = "llm_bench"


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

    Idempotent: safe to call more than once per process (the dashboard and
    tests both invoke the CLI entry point in-process), and never touches the
    root logger so embedding applications keep their own configuration.
    """

    level_name = resolve_log_level(explicit)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level_name)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler()  # defaults to stderr
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    return level_name
