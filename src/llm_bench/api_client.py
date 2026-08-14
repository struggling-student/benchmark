"""Backend-neutral asynchronous OpenAI-compatible streaming benchmark client."""

from __future__ import annotations

import asyncio
import json
import math
import time
from statistics import fmean, median
from typing import Any

from .config import ConfigurationError, ExperimentConfig


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float], prefix: str) -> dict[str, float | None]:
    return {
        f"mean_{prefix}_ms": fmean(values) if values else None,
        f"median_{prefix}_ms": median(values) if values else None,
        f"p95_{prefix}_ms": _percentile(values, 0.95),
        f"p99_{prefix}_ms": _percentile(values, 0.99),
    }


async def run_api_benchmark(
    config: ExperimentConfig,
    workload_manifest: dict[str, Any],
    *,
    base_url: str,
    timeout_seconds: float = 600.0,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Issue a scheduled workload and return stable raw harness JSON."""

    try:
        import httpx
    except ImportError as exc:
        raise ConfigurationError(
            "httpx is required for shared API benchmarks; install the runner dependencies"
        ) from exc

    maximum_concurrency = config.maximum_concurrency or 1
    request_rate = config.request_rate
    semaphore = asyncio.Semaphore(maximum_concurrency)
    benchmark_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport) as client:

        async def one(prompt: dict[str, Any]) -> dict[str, Any]:
            if request_rate is not None:
                due = benchmark_start + int(prompt["index"]) / request_rate
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
            async with semaphore:
                started = time.perf_counter()
                first_content: float | None = None
                content_times: list[float] = []
                output_parts: list[str] = []
                usage: dict[str, Any] = {}
                status_code: int | None = None
                error: str | None = None
                payload = {
                    "model": config.model_id,
                    "prompt": prompt["text"],
                    "max_tokens": config.output_length,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "ignore_eos": config.ignore_eos,
                    "seed": config.seed + int(prompt["index"]),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                try:
                    async with client.stream(
                        "POST",
                        f"{base_url.rstrip('/')}/v1/completions",
                        headers={"Authorization": "Bearer llm-bench-local"},
                        json=payload,
                    ) as response:
                        status_code = response.status_code
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            event = json.loads(data)
                            if isinstance(event.get("usage"), dict):
                                usage.update(event["usage"])
                            choices = event.get("choices") or []
                            text = choices[0].get("text", "") if choices else ""
                            if text:
                                now = time.perf_counter()
                                first_content = first_content or now
                                content_times.append(now)
                                output_parts.append(str(text))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finished = time.perf_counter()
                intervals = [
                    (right - left) * 1000
                    for left, right in zip(content_times, content_times[1:], strict=False)
                ]
                output_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")
                itl_defensible = (
                    isinstance(output_tokens, int)
                    and output_tokens > 1
                    and len(content_times) == output_tokens
                )
                ttft_ms = (first_content - started) * 1000 if first_content else None
                e2e_ms = (finished - started) * 1000
                tpot_ms = None
                if first_content and isinstance(output_tokens, int) and output_tokens > 1:
                    tpot_ms = (finished - first_content) * 1000 / (output_tokens - 1)
                return {
                    "index": prompt["index"],
                    "success": error is None,
                    "status_code": status_code,
                    "error": error,
                    "requested_input_tokens": prompt["token_count"],
                    "requested_output_tokens": config.output_length,
                    "actual_input_tokens": prompt_tokens,
                    "actual_output_tokens": output_tokens,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "itl_ms": intervals if itl_defensible else None,
                    "e2e_latency_ms": e2e_ms,
                    "response_text_length": len("".join(output_parts)),
                }

        records = await asyncio.gather(*(one(prompt) for prompt in workload_manifest["prompts"]))

    duration = time.perf_counter() - benchmark_start
    successful = [record for record in records if record["success"]]
    failed = len(records) - len(successful)
    ttft = [float(record["ttft_ms"]) for record in successful if record["ttft_ms"] is not None]
    tpot = [float(record["tpot_ms"]) for record in successful if record["tpot_ms"] is not None]
    itl = [
        float(value)
        for record in successful
        if isinstance(record["itl_ms"], list)
        for value in record["itl_ms"]
    ]
    e2e = [float(record["e2e_latency_ms"]) for record in successful]
    input_counts = [record["actual_input_tokens"] for record in successful]
    output_counts = [record["actual_output_tokens"] for record in successful]
    actual_input = sum(value for value in input_counts if isinstance(value, int))
    actual_output = sum(value for value in output_counts if isinstance(value, int))
    counts_complete = all(isinstance(value, int) for value in (*input_counts, *output_counts))
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "measurement_method": "shared_openai_streaming",
        "measurement_scope": "client_observed_end_to_end",
        "workload_manifest_sha256": workload_manifest["sha256"],
        "duration_seconds": duration,
        "successful_requests": len(successful),
        "failed_requests": failed,
        "actual_input_tokens": actual_input if counts_complete else None,
        "actual_output_tokens": actual_output if counts_complete else None,
        "request_throughput_requests_per_second": len(successful) / duration if duration else None,
        "input_throughput_tokens_per_second": (
            actual_input / duration if duration and counts_complete else None
        ),
        "output_throughput_tokens_per_second": (
            actual_output / duration if duration and counts_complete else None
        ),
        "total_throughput_tokens_per_second": (
            (actual_input + actual_output) / duration if duration and counts_complete else None
        ),
        **_summary(ttft, "ttft"),
        **_summary(tpot, "tpot"),
        **_summary(itl, "itl"),
        "mean_e2e_latency_ms": fmean(e2e) if e2e else None,
        "p95_e2e_latency_ms": _percentile(e2e, 0.95),
        "requests": records,
        "warnings": [],
    }
    if not counts_complete:
        result["warnings"].append(
            "One or more streaming responses omitted usage token counts; token throughput is null."
        )
    if any(record["actual_output_tokens"] != config.output_length for record in successful):
        result["warnings"].append(
            "One or more requests produced a different token count than requested."
        )
    if any(record["itl_ms"] is None for record in successful):
        result["warnings"].append(
            "ITL is null for responses whose streaming chunks could not be proven to map "
            "one-to-one to generated tokens."
        )
    return result
