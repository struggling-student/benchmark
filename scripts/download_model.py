#!/usr/bin/env python3
"""Pre-download a configured model and tokenizer into an external HF cache."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

from llm_bench.config import ConfigurationError, load_experiment

_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_CACHE_KEYS = {"HF_HOME", "HF_HUB_CACHE", "MODEL_CACHE_DIR"}


def _read_cache_settings(path: Path) -> dict[str, str]:
    """Read only cache assignments; never execute the cluster environment file."""

    settings: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read cluster configuration {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT.match(stripped)
        if match is None or match.group(1) not in _CACHE_KEYS:
            continue
        key, raw_value = match.groups()
        try:
            tokens = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"invalid {key} assignment at {path}:{number}: {exc}") from exc
        if len(tokens) > 1:
            raise ValueError(f"{key} at {path}:{number} must contain one path")
        settings[key] = tokens[0] if tokens else ""
    return settings


def _cache_directory(settings: dict[str, str], override: Path | None) -> Path:
    raw = (
        str(override)
        if override is not None
        else settings.get("MODEL_CACHE_DIR")
        or settings.get("HF_HUB_CACHE")
        or os.environ.get("MODEL_CACHE_DIR")
        or os.environ.get("HF_HUB_CACHE")
    )
    if not raw:
        raise ValueError(
            "no model cache configured; set MODEL_CACHE_DIR/HF_HUB_CACHE in the cluster "
            "configuration or pass --cache-dir"
        )
    if "<" in raw or ">" in raw:
        raise ValueError("model cache is still a placeholder; configure an absolute path")
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not expanded.is_absolute():
        raise ValueError(f"model cache must be an absolute path: {expanded}")
    cache = expanded.resolve()
    repository = Path(__file__).resolve().parents[1]
    try:
        cache.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError(f"refusing to cache model weights inside the Git repository: {cache}")
    return cache


def _redacted_error(exc: Exception) -> str:
    # Some HTTP exceptions can include request details. The exception class is enough
    # for diagnosis here and guarantees that a credential can never be echoed.
    return type(exc).__name__


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-download a model and any distinct tokenizer. Authentication is read "
            "from HF_TOKEN or the Hugging Face credential store and is never printed."
        )
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="cluster environment configuration"
    )
    parser.add_argument("--experiment", type=Path, required=True, help="validated experiment YAML")
    parser.add_argument("--cache-dir", type=Path, help="override the configured cache")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="verify/use cached files without attempting network access",
    )
    args = parser.parse_args()

    try:
        experiment = load_experiment(args.experiment)
        settings = _read_cache_settings(args.config)
        cache_dir = _cache_directory(settings, args.cache_dir)
    except (ConfigurationError, ValueError) as exc:
        parser.error(str(exc))

    if settings.get("HF_HOME"):
        os.environ.setdefault("HF_HOME", settings["HF_HOME"])
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error(
            "huggingface_hub is not installed; install a compatible vLLM environment first"
        )

    token = os.environ.get("HF_TOKEN")
    downloads = [(experiment.model_id, experiment.model_revision, "model")]
    if experiment.tokenizer != experiment.model_id:
        downloads.append((experiment.tokenizer, None, "tokenizer"))
        print(
            "warning: tokenizer_id differs from model_id; no separate tokenizer revision "
            "exists in this schema, so its configured/default revision will be downloaded",
            file=sys.stderr,
        )
    for repository_id, revision, kind in downloads:
        try:
            location = snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=args.local_files_only,
            )
        except Exception as exc:  # The library exposes optional HTTP error classes.
            parser.error(
                f"{kind} download failed; verify network access, cache space, login/token, "
                f"and gated-model approval ({_redacted_error(exc)})"
            )
        print(f"{kind} cached at: {location}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
