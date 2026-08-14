"""Public CLI for composed benchmark preparation, execution, and analysis."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import (
    PROVIDERS,
    ConfigurationError,
    load_composed_experiment,
    load_model_manifest,
)
from .preparation import prepare_model
from .results import ResultError, compare_paths
from .runner import run_experiment, validate_runtime


def _composed(args: argparse.Namespace):
    return load_composed_experiment(
        args.model,
        args.workload,
        args.profile,
        provider=args.provider,
        variant=args.variant,
        artifact_root=args.artifact_root,
    )


def _validate(args: argparse.Namespace) -> int:
    config = _composed(args)
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    config = _composed(args)
    report = validate_runtime(config, require_artifact=not args.allow_missing_artifact)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _prepare_model(args: argparse.Namespace) -> int:
    model = load_model_manifest(args.model)
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if not variants:
        raise ConfigurationError("--variants must contain at least one variant")
    artifact_root = args.artifact_root or os.environ.get("MODEL_ARTIFACT_ROOT")
    cache_root = args.cache_root or os.environ.get("MODEL_CACHE_DIR")
    if not artifact_root or not cache_root:
        raise ConfigurationError(
            "prepare-model requires --artifact-root/--cache-root or "
            "MODEL_ARTIFACT_ROOT/MODEL_CACHE_DIR"
        )
    manifest = prepare_model(
        model,
        provider_name=args.provider,
        variants=variants,
        artifact_root=artifact_root,
        cache_root=cache_root,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _run(args: argparse.Namespace) -> int:
    config = _composed(args)
    if args.dry_run:
        report = validate_runtime(config, require_artifact=not args.allow_missing_artifact)
        report["resolved_experiment"] = config.to_dict()
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    results_root = args.results_root or os.environ.get("RESULTS_ROOT") or Path.cwd() / "results"
    destination = run_experiment(
        config,
        source_paths={
            "model": Path(args.model).expanduser().resolve(),
            "workload": Path(args.workload).expanduser().resolve(),
            "profile": Path(args.profile).expanduser().resolve(),
        },
        results_root=results_root,
        run_directory=args.run_dir,
        repository=args.repository,
    )
    print(destination)
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
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).with_name("dashboard.py")),
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
        command.extend(("--", "--results-root", str(args.results_root.expanduser().resolve())))
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


def _add_composed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, required=True, help="model manifest YAML")
    parser.add_argument("--workload", type=Path, required=True, help="workload YAML")
    parser.add_argument("--profile", type=Path, required=True, help="execution profile YAML")
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--variant", required=True, help="model artifact variant, such as f16")
    parser.add_argument("--artifact-root", type=Path, help="override MODEL_ARTIFACT_ROOT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description="Composable vLLM and llama.cpp inference benchmarking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a composed experiment")
    _add_composed_arguments(validate)
    validate.set_defaults(handler=_validate)

    prepare = subparsers.add_parser(
        "prepare-model", help="download, convert, and optionally quantize a model"
    )
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--provider", choices=PROVIDERS, required=True)
    prepare.add_argument("--variants", required=True, help="comma-separated GGUF variants")
    prepare.add_argument("--artifact-root", type=Path)
    prepare.add_argument("--cache-root", type=Path)
    prepare.add_argument("--local-files-only", action="store_true")
    prepare.set_defaults(handler=_prepare_model)

    preflight = subparsers.add_parser("preflight", help="validate runtime and artifacts")
    _add_composed_arguments(preflight)
    preflight.add_argument("--allow-missing-artifact", action="store_true", help=argparse.SUPPRESS)
    preflight.set_defaults(handler=_preflight)

    run = subparsers.add_parser("run", help="run one composed benchmark")
    _add_composed_arguments(run)
    run.add_argument("--results-root", type=Path)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--repository", type=Path, default=Path.cwd())
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--allow-missing-artifact", action="store_true", help=argparse.SUPPRESS)
    run.set_defaults(handler=_run)

    compare = subparsers.add_parser("compare", help="compare two summaries or result directories")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.set_defaults(handler=_compare)

    dashboard = subparsers.add_parser("dashboard", help="launch the read-only Streamlit dashboard")
    dashboard.add_argument("--results-root", type=Path)
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8501)
    dashboard.set_defaults(handler=_dashboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler: Any = args.handler
        return int(handler(args))
    except (ConfigurationError, ResultError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
