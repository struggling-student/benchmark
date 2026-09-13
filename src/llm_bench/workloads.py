"""Deterministic prompt manifests shared by API-backed benchmark runners."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Protocol

from .config import ConfigurationError, ExperimentConfig


class Tokenizer(Protocol):
    all_special_ids: list[int]
    vocab_size: int

    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...


def _load_tokenizer(config: ExperimentConfig, *, local_files_only: bool) -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ConfigurationError(
            "transformers is required to create deterministic workload manifests; "
            "install the runner dependencies"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            config.tokenizer,
            revision=config.model_revision,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise ConfigurationError(
            f"cannot load tokenizer {config.tokenizer!r} for workload preparation: "
            f"{type(exc).__name__}"
        ) from exc


def _prompt(tokenizer: Tokenizer, rng: random.Random, length: int) -> tuple[str, list[int]]:
    special = set(tokenizer.all_special_ids)
    candidates = [index for index in range(tokenizer.vocab_size) if index not in special]
    if not candidates:
        raise ConfigurationError("tokenizer has no usable non-special tokens")
    # Resampling random token ids until one survives a decode/encode round trip
    # cannot work at realistic prompt lengths.  The decoded text re-tokenizes
    # into more pieces than it started with, and the drift grows with length:
    # for Llama-3.2 at length 256 the re-encoding was never shorter than 260
    # over 60 draws, so every attempt is rejected and the loop always exhausts.
    # Correct the length instead of resampling.  This reaches a fixed point in
    # about two iterations and stays deterministic for a given seed.
    token_ids = [rng.choice(candidates) for _ in range(length)]
    for _ in range(100):
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        round_trip = tokenizer.encode(text, add_special_tokens=False)
        if text and len(round_trip) == length:
            return text, round_trip
        token_ids = (
            round_trip[:length]
            if len(round_trip) > length
            else round_trip
            + [rng.choice(candidates) for _ in range(length - len(round_trip))]
        )
    raise ConfigurationError(
        f"could not create a stable {length}-token prompt after 100 deterministic attempts"
    )


def build_workload_manifest(
    config: ExperimentConfig,
    *,
    tokenizer: Tokenizer | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Create exact-length prompt text from the canonical tokenizer."""

    selected = tokenizer or _load_tokenizer(config, local_files_only=local_files_only)
    rng = random.Random(config.seed)
    prompts = []
    for index in range(config.number_of_prompts):
        text, token_ids = _prompt(selected, rng, config.input_length)
        prompts.append(
            {
                "index": index,
                "text": text,
                "token_count": len(token_ids),
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer,
        "workload_key": config.workload_key,
        "seed": config.seed,
        "input_length": config.input_length,
        "output_length": config.output_length,
        "prompts": prompts,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return document


def build_native_workload_manifest(config: ExperimentConfig) -> dict[str, Any]:
    """Record the deterministic shape mapped to backend-native offline generators."""

    document: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "backend_native_shape",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_id": config.tokenizer,
        "workload_key": config.workload_key,
        "seed": config.seed,
        "number_of_prompts": config.number_of_prompts,
        "input_length": config.input_length,
        "output_length": config.output_length,
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return document


def write_workload_manifest(path: str | Path, document: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
