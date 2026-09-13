"""Reproducible Hugging Face snapshot and GGUF artifact preparation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigurationError, ModelManifest
from .providers import ProviderContext, get_provider

logger = logging.getLogger(__name__)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: str | Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def _snapshot(model: ModelManifest, cache_root: Path, *, local_files_only: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ConfigurationError("huggingface_hub is required to prepare models") from exc
    logger.info(
        "resolving Hugging Face snapshot: %s (revision=%s, local_files_only=%s)",
        model.model_id,
        model.model_revision or "latest",
        local_files_only,
    )
    try:
        location = snapshot_download(
            repo_id=model.model_id,
            revision=model.model_revision,
            cache_dir=cache_root,
            local_files_only=local_files_only,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        logger.error("model snapshot preparation failed: %s: %s", type(exc).__name__, exc)
        raise ConfigurationError(
            f"model snapshot preparation failed: {type(exc).__name__}"
        ) from exc
    resolved = Path(location).resolve()
    logger.info("snapshot resolved: %s", resolved)
    return resolved


def _resolved_revision(snapshot: Path) -> str | None:
    if snapshot.parent.name == "snapshots" and len(snapshot.name) >= 40:
        return snapshot.name
    return None


# Float GGUF variants come straight out of convert_hf_to_gguf.py at the matching
# --outtype. Quantized variants are produced from the F16 base by llama-quantize.
# BF16 is deliberately not routed through the F16 base: the source checkpoint is
# BF16, so f16 -> bf16 would round-trip through a narrower exponent range and the
# artifact would no longer be the model the upstream weights describe.
_CONVERTER_OUTTYPE = {
    "float32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
}


def _converter_outtype(variant_name: str, precision: str) -> str:
    outtype = _CONVERTER_OUTTYPE.get(precision)
    if outtype is None:
        raise ConfigurationError(
            f"variant {variant_name} has no quantization and precision {precision!r} "
            "is not a llama.cpp converter output type"
        )
    return outtype


def _native_conversion_command(snapshot: Path, output: Path, outtype: str) -> list[str]:
    script = os.environ.get("LLAMA_CPP_CONVERT_SCRIPT")
    if not script:
        raise ConfigurationError("LLAMA_CPP_CONVERT_SCRIPT is required for native conversion")
    source = Path(script).expanduser().resolve()
    if not source.is_file():
        raise ConfigurationError(f"llama.cpp converter not found: {source}")
    return [
        sys.executable,
        str(source),
        str(snapshot),
        "--outfile",
        str(output),
        "--outtype",
        outtype,
    ]


def _conversion_command(
    provider_name: str, snapshot: Path, output: Path, outtype: str
) -> list[str]:
    if provider_name == "native":
        return _native_conversion_command(snapshot, output, outtype)
    return [
        "python3",
        "/app/convert_hf_to_gguf.py",
        str(snapshot),
        "--outfile",
        str(output),
        "--outtype",
        outtype,
    ]


def _native_quantize_command(source: Path, output: Path, kind: str) -> list[str]:
    binary = os.environ.get("LLAMA_CPP_QUANTIZE_BIN") or "llama-quantize"
    return [binary, str(source), str(output), kind]


def _provider_command(
    provider_name: str,
    model: ModelManifest,
    command: list[str],
    model_root: Path,
    cache_root: Path,
) -> list[str]:
    if provider_name == "native":
        return command
    from .config import ExperimentConfig

    placeholder = ExperimentConfig(
        experiment_name=f"prepare-{model.model_key}",
        backend="llamacpp",
        model_id=model.model_id,
        provider=provider_name,
        artifact_format="gguf",
        artifact_path=str(model_root / "placeholder.gguf"),
        hardware_type="cpu",
    )
    return get_provider(provider_name).wrap(
        command,
        placeholder,
        # The preparation output lives under model_root, which is already mounted
        # read-write as the provider run directory.  Mounting model_root.parent as
        # the read-only artifact root as well creates overlapping binds; Singularity
        # applies the parent bind last and makes the temporary output read-only.
        ProviderContext(model_root, None, cache_root),
    )


def _run(command: Sequence[str], log: Path) -> None:
    logger.info("running preparation command (log=%s): %s", log, list(command))
    with log.open("a", encoding="utf-8") as stream:
        stream.write("command=" + json.dumps(list(command)) + "\n")
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        logger.error(
            "preparation command failed with exit %d; see %s", completed.returncode, log
        )
        raise ConfigurationError(
            f"model preparation command failed with exit {completed.returncode}; inspect {log}"
        )
    logger.debug("preparation command exited 0")


def _command_version(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output.splitlines()[0] if completed.returncode == 0 and output else None


def _git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _reusable_record(
    manifest: dict[str, Any] | None,
    *,
    model: ModelManifest,
    revision: str | None,
    variant: str,
    path: Path,
) -> dict[str, Any] | None:
    if not manifest or manifest.get("model_id") != model.model_id:
        return None
    if manifest.get("resolved_revision") != revision:
        return None
    record = manifest.get("artifacts", {}).get(variant)
    if not isinstance(record, dict) or not path.is_file():
        return None
    expected = record.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        return None
    reused = dict(record)
    reused["path"] = str(path)
    reused["size_bytes"] = path.stat().st_size
    reused["reused"] = True
    return reused


def _assert_no_unproven_artifact(path: Path, variant: str) -> None:
    if path.exists():
        raise ConfigurationError(
            f"existing {variant} artifact does not match its recorded model revision/hash: "
            f"{path}; move it aside or restore its matching artifact manifest"
        )


def prepare_model(
    model: ModelManifest,
    *,
    provider_name: str,
    variants: Sequence[str],
    artifact_root: str | Path,
    cache_root: str | Path,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Cache a model snapshot and create any explicitly requested GGUF variants."""

    logger.info(
        "preparing model %s: variants=%s provider=%s",
        model.model_key,
        list(variants),
        provider_name,
    )
    requested = list(dict.fromkeys(variants))
    if provider_name not in {"native", "docker", "apptainer"}:
        raise ConfigurationError(f"unknown model preparation provider: {provider_name}")
    if not requested:
        raise ConfigurationError("at least one model variant must be requested")
    unknown = sorted(set(requested) - set(model.variants))
    if unknown:
        raise ConfigurationError("unknown model variant(s): " + ", ".join(unknown))
    llamacpp_requested = [
        name for name in requested if "llamacpp" in model.variants[name].artifacts
    ]
    root = Path(artifact_root).expanduser().resolve()
    cache = Path(cache_root).expanduser().resolve()
    if not root.is_absolute() or not cache.is_absolute():
        raise ConfigurationError("artifact and cache roots must be absolute")
    model_root = root / model.model_key
    model_root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = _snapshot(model, cache, local_files_only=local_files_only)
    resolved_revision = _resolved_revision(snapshot)
    if resolved_revision is None:
        raise ConfigurationError(
            "Hugging Face snapshot did not resolve to an immutable commit directory"
        )
    log = model_root / "preparation.log"
    manifest_path = model_root / "artifact-manifest.json"
    previous_manifest = _read_manifest(manifest_path)
    if not llamacpp_requested:
        manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "model_key": model.model_key,
            "model_id": model.model_id,
            "requested_revision": model.model_revision,
            "resolved_revision": resolved_revision,
            "tokenizer_id": model.tokenizer_id,
            "snapshot_path": str(snapshot),
            "provider": provider_name,
            "provider_image": None,
            "tool_provenance": {"operation": "huggingface_snapshot_only"},
            "artifacts": {},
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "model preparation complete (snapshot only): %s -> %s",
            model.model_key,
            manifest_path,
        )
        return manifest
    # Only quantized variants need the F16 base; a float variant is converted from
    # the snapshot directly, so requesting bf16 alone must not force an F16 build.
    needs_f16_base = any(
        name == "f16" or model.variants[name].quantization for name in llamacpp_requested
    )
    f16_artifact = model.variants.get("f16")
    if needs_f16_base and (f16_artifact is None or "llamacpp" not in f16_artifact.artifacts):
        raise ConfigurationError("GGUF preparation requires an f16 llama.cpp artifact definition")
    f16_name = (
        f16_artifact.artifacts["llamacpp"].filename
        if f16_artifact is not None and "llamacpp" in f16_artifact.artifacts
        else None
    )
    f16_path = model_root / f16_name if f16_name is not None else None
    produced: dict[str, dict[str, Any]] = {}
    provider_image = (
        os.environ.get("LLAMA_CPP_DOCKER_IMAGE")
        if provider_name == "docker"
        else os.environ.get("LLAMA_CPP_APPTAINER_IMAGE")
        if provider_name == "apptainer"
        else None
    )
    converter_path = os.environ.get("LLAMA_CPP_CONVERT_SCRIPT")
    quantizer_path = os.environ.get("LLAMA_CPP_QUANTIZE_BIN") or "llama-quantize"
    tool_provenance: dict[str, Any] = {
        "provider": provider_name,
        "image": provider_image,
        "image_digest": (
            "sha256:" + provider_image.split("@sha256:", 1)[1]
            if provider_image and "@sha256:" in provider_image
            else "sha256:" + sha256_file(Path(provider_image).expanduser())
            if provider_image and Path(provider_image).expanduser().is_file()
            else None
        ),
        "converter": (
            {
                "path": str(Path(converter_path).expanduser().resolve()),
                "version": _git_revision(Path(converter_path).expanduser().resolve().parent),
                "sha256": _optional_sha256(Path(converter_path).expanduser()),
            }
            if provider_name == "native" and converter_path
            else {"path": "/app/convert_hf_to_gguf.py", "version": provider_image}
        ),
        "quantizer": (
            {
                "path": shutil.which(quantizer_path) or quantizer_path,
                "version": _command_version((quantizer_path, "--version")),
            }
            if provider_name == "native"
            else {"path": "llama-quantize", "version": provider_image}
        ),
    }

    with tempfile.TemporaryDirectory(prefix="llm-bench-prepare-", dir=model_root) as temporary:
        temporary_root = Path(temporary)
        if needs_f16_base:
            assert f16_artifact is not None and f16_name is not None and f16_path is not None
            reusable_f16 = _reusable_record(
                previous_manifest,
                model=model,
                revision=resolved_revision,
                variant="f16",
                path=f16_path,
            )
            if reusable_f16:
                logger.info("reusing existing f16 artifact: %s", f16_path)
                produced["f16"] = reusable_f16
            else:
                logger.info("converting snapshot to f16 GGUF: %s", f16_path)
                _assert_no_unproven_artifact(f16_path, "f16")
                temporary_f16 = temporary_root / f16_name
                _run(
                    _provider_command(
                        provider_name,
                        model,
                        _conversion_command(provider_name, snapshot, temporary_f16, "f16"),
                        model_root,
                        cache,
                    ),
                    log,
                )
                if not temporary_f16.is_file():
                    raise ConfigurationError("converter completed without producing the F16 GGUF")
                temporary_f16.replace(f16_path)
                produced["f16"] = {
                    "path": str(f16_path),
                    "format": "gguf",
                    "precision": f16_artifact.precision,
                    "quantization": f16_artifact.quantization,
                    "size_bytes": f16_path.stat().st_size,
                    "sha256": sha256_file(f16_path),
                    "reused": False,
                    "arguments": [
                        "SOURCE_SNAPSHOT",
                        "--outfile",
                        f16_name,
                        "--outtype",
                        "f16",
                    ],
                }

        for name in llamacpp_requested:
            variant = model.variants[name]
            artifact = variant.artifacts["llamacpp"]
            assert artifact.filename is not None
            destination = model_root / artifact.filename
            if name == "f16":
                continue
            reusable = _reusable_record(
                previous_manifest,
                model=model,
                revision=resolved_revision,
                variant=name,
                path=destination,
            )
            if reusable:
                logger.info("reusing existing %s artifact: %s", name, destination)
                produced[name] = reusable
                continue
            _assert_no_unproven_artifact(destination, name)
            temporary_output = temporary_root / artifact.filename
            quantization = variant.quantization
            if quantization:
                assert f16_name is not None and f16_path is not None
                logger.info("quantizing %s -> %s (%s)", f16_path, destination, quantization)
                command = (
                    _native_quantize_command(f16_path, temporary_output, quantization)
                    if provider_name == "native"
                    else [
                        "llama-quantize",
                        str(f16_path),
                        str(temporary_output),
                        quantization,
                    ]
                )
                arguments = [f16_name, artifact.filename, quantization]
                failure = f"quantizer completed without producing variant {name}"
            else:
                outtype = _converter_outtype(name, variant.precision)
                logger.info("converting snapshot to %s GGUF: %s", outtype, destination)
                command = _conversion_command(provider_name, snapshot, temporary_output, outtype)
                arguments = [
                    "SOURCE_SNAPSHOT",
                    "--outfile",
                    artifact.filename,
                    "--outtype",
                    outtype,
                ]
                failure = f"converter completed without producing variant {name}"
            _run(
                _provider_command(provider_name, model, command, model_root, cache),
                log,
            )
            if not temporary_output.is_file():
                raise ConfigurationError(failure)
            temporary_output.replace(destination)
            produced[name] = {
                "path": str(destination),
                "format": artifact.format,
                "precision": variant.precision,
                "quantization": variant.quantization,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "reused": False,
                "arguments": arguments,
            }

    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_key": model.model_key,
        "model_id": model.model_id,
        "requested_revision": model.model_revision,
        "resolved_revision": resolved_revision,
        "tokenizer_id": model.tokenizer_id,
        "snapshot_path": str(snapshot),
        "provider": provider_name,
        "provider_image": provider_image,
        "tool_provenance": tool_provenance,
        "artifacts": produced,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("model preparation complete: %s -> %s", model.model_key, manifest_path)
    return manifest


def load_artifact_manifest(config_artifact_path: str | None) -> dict[str, Any] | None:
    if not config_artifact_path:
        return None
    path = Path(config_artifact_path)
    manifest_path = path.parent / "artifact-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
