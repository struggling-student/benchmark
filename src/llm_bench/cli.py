"""Argparse entry points used by the shell and Slurm wrappers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import BENCHMARK_TYPES, ConfigurationError, load_experiment
from .measurements import create_measurements, write_measurements
from .metadata import collect_metadata, write_metadata
from .results import (
    ResultError,
    compare_paths,
    create_summary,
    load_summary,
    normalize_summary,
    write_summary,
)


def _benchmark_type(config_type: str, requested: str | None) -> str:
    if requested is not None and requested != config_type:
        raise ConfigurationError(
            f"requested benchmark type {requested!r} does not match configuration "
            f"type {config_type!r}"
        )
    return requested or config_type


def _validate_config(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    selected = _benchmark_type(config.resolved_benchmark_type, args.benchmark_type)
    print(f"valid: {config.experiment_name} ({selected}, {config.backend}, {config.model_id})")
    return 0


def _get_config(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    values = config.to_dict()
    if args.field not in values:
        valid = ", ".join(sorted(values))
        raise ConfigurationError(f"unknown configuration field {args.field!r}; choose: {valid}")
    value = values[args.field]
    if args.json:
        print(json.dumps(value, allow_nan=False))
    elif value is None:
        print("")
    elif isinstance(value, bool):
        print(str(value).lower())
    else:
        print(value)
    return 0


def _init_run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_experiment(config_path)
    selected = _benchmark_type(config.resolved_benchmark_type, args.benchmark_type)
    run_directory = Path(args.run_dir).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(
        config,
        benchmark_type=selected,
        run_id=args.run_id,
        repository=args.repository,
    )
    shutil.copyfile(config_path, run_directory / "experiment.yaml")
    write_metadata(run_directory / "metadata.json", metadata)
    summary = create_summary(config, metadata)
    write_summary(run_directory / "summary.json", summary)
    print(run_directory)
    return 0


def _normalize_results(args: argparse.Namespace) -> int:
    config = load_experiment(args.config)
    selected = _benchmark_type(config.resolved_benchmark_type, args.benchmark_type)
    run_directory = Path(args.run_dir).resolve()
    summary_path = run_directory / "summary.json"
    summary = load_summary(summary_path)
    if summary.get("benchmark_type") != selected:
        raise ResultError(
            "summary benchmark_type does not match the requested experiment type: "
            f"{summary.get('benchmark_type')!r} != {selected!r}"
        )
    normalized = normalize_summary(
        summary,
        raw_output_paths=args.raw_output,
        telemetry_paths=args.telemetry,
        status=args.status,
        duration_seconds=args.duration_seconds,
        model_load_time_seconds=args.model_load_time_seconds,
        warnings=args.warning,
        error=args.error,
    )
    measurements = create_measurements(
        normalized,
        run_directory=run_directory,
        raw_output_paths=args.raw_output,
        telemetry_paths=args.telemetry,
    )
    write_measurements(run_directory / "measurements.json", measurements)
    write_summary(summary_path, normalized)
    print(summary_path)
    return 0


def _compare(args: argparse.Namespace) -> int:
    comparison, json_path, csv_path = compare_paths(args.left, args.right, args.output_dir)
    print(f"compatibility: {comparison['compatibility']['status']}")
    print(json_path)
    print(csv_path)
    return 0


def _dashboard(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise ResultError(
            "dashboard dependencies are unavailable; install them with "
            'pip install -e ".[dashboard]"'
        )
    dashboard_path = Path(__file__).with_name("dashboard.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
    ]
    if args.results_root is not None:
        command.extend(
            (
                "--",
                "--results-root",
                str(args.results_root.expanduser().resolve()),
            )
        )
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


def _add_type_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-type", choices=BENCHMARK_TYPES)


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser, kept separate for focused unit tests."""

    parser = argparse.ArgumentParser(
        prog="llm-bench", description="Thin metadata and result utilities for vLLM runs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate one experiment YAML file")
    validate.add_argument("config", type=Path)
    _add_type_argument(validate)
    validate.set_defaults(handler=_validate_config)

    get_config = subparsers.add_parser(
        "get-config", help="print one validated experiment field for a shell wrapper"
    )
    get_config.add_argument("config", type=Path)
    get_config.add_argument("field")
    get_config.add_argument("--json", action="store_true", help="print JSON, including null")
    get_config.set_defaults(handler=_get_config)

    initialize = subparsers.add_parser(
        "init-run", help="copy the experiment and initialize metadata.json and summary.json"
    )
    initialize.add_argument("--config", type=Path, required=True)
    initialize.add_argument("--run-dir", type=Path, required=True)
    _add_type_argument(initialize)
    initialize.add_argument("--run-id", help="reuse a run ID allocated by the wrapper")
    initialize.add_argument(
        "--repository", type=Path, default=Path.cwd(), help="repository used for git_commit"
    )
    initialize.set_defaults(handler=_init_run)

    normalize = subparsers.add_parser(
        "normalize-results",
        help="aggregate explicitly supplied measured repetitions into summary.json",
    )
    normalize.add_argument("--config", type=Path, required=True)
    normalize.add_argument("--run-dir", type=Path, required=True)
    _add_type_argument(normalize)
    normalize.add_argument("--status", choices=("completed", "failed"), default="completed")
    normalize.add_argument(
        "--raw-output",
        type=Path,
        action="append",
        default=[],
        help="measured raw vLLM JSON; repeat for every measured repetition",
    )
    normalize.add_argument(
        "--telemetry",
        type=Path,
        action="append",
        default=[],
        help="measured nvidia-smi CSV; repeat for every telemetry file",
    )
    normalize.add_argument(
        "--duration-seconds",
        type=float,
        help="fallback duration when the backend JSON does not contain one",
    )
    normalize.add_argument(
        "--model-load-time-seconds", type=float, help="measured model loading duration"
    )
    normalize.add_argument("--warning", action="append", default=[])
    normalize.add_argument("--error", help="failure detail saved without dropping the run")
    normalize.set_defaults(handler=_normalize_results)

    compare = subparsers.add_parser(
        "compare", help="compare exactly two explicit summaries or result directories"
    )
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(handler=_compare)

    dashboard = subparsers.add_parser(
        "dashboard", help="launch the optional interactive results dashboard"
    )
    dashboard.add_argument(
        "--results-root",
        type=Path,
        help="results directory; omit to open the in-memory demo study",
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(handler=_dashboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a subcommand and render domain errors without a Python traceback."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler: Any = args.handler
        return int(handler(args))
    except (ConfigurationError, ResultError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
