"""Parse and record standalone memory-bandwidth microbenchmark results.

This covers STREAM and Intel MLC, run directly against a NUMA node (via
`numactl`) to characterize raw HBM/DDR bandwidth and latency on CRESCO8's Xeon
Max nodes. It is deliberately separate from `results.py`'s LLM-run summary
schema: a memory probe has no model, backend, or token throughput, and forcing
it into that ~150-field schema would be misleading rather than useful. The only
things reused from the rest of the package are NUMA/topology detection
(`hardware.py`) and run-naming/timestamp helpers (`metadata.py`), both already
tested elsewhere.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from .hardware import detect_xeon_max_hbm_mode, lscpu_values, numa_nodes
from .metadata import make_run_id, utc_timestamp

SCHEMA_VERSION = "1.0"

_STREAM_LINE = re.compile(
    r"^(?P<label>Copy|Scale|Add|Triad):\s*"
    r"(?P<best_mbps>[0-9.]+)\s+"
    r"(?P<avg_time_s>[0-9.]+)\s+"
    r"(?P<min_time_s>[0-9.]+)\s+"
    r"(?P<max_time_s>[0-9.]+)\s*$"
)

# Intel MLC's --bandwidth_matrix / --latency_matrix print a "Numa node" header row
# followed by one row per source node: the row's own node id, then one value per
# destination node. This has not yet been validated against a real MLC run on
# CRESCO8 (Intel MLC is not vendored here — see docs/07_MEMORY_BANDWIDTH_MICROBENCHMARK.md)
# — treat it as a best-effort parser to be confirmed once real output is captured.
_MLC_HEADER = re.compile(r"^\s*Numa node\s*$", re.IGNORECASE)
_MLC_COLUMN_HEADER = re.compile(r"^\s*Numa node\s+(?P<cols>[\d\s]+)$", re.IGNORECASE)
_MLC_ROW = re.compile(r"^\s*(?P<row>\d+)\s+(?P<values>[\d.\s]+)$")


def parse_stream_output(text: str) -> dict[str, dict[str, float]]:
    """Parse STREAM's Copy/Scale/Add/Triad results table.

    Returns e.g. ``{"triad": {"best_mbps": ..., "avg_time_s": ..., ...}}``.
    Raises ValueError if none of the four expected rows are found — a silent
    empty result would be indistinguishable from a genuine all-zero measurement.
    """

    results: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = _STREAM_LINE.match(line.strip())
        if not match:
            continue
        results[match["label"].lower()] = {
            "best_mbps": float(match["best_mbps"]),
            "avg_time_s": float(match["avg_time_s"]),
            "min_time_s": float(match["min_time_s"]),
            "max_time_s": float(match["max_time_s"]),
        }
    if not results:
        raise ValueError("no STREAM result rows (Copy/Scale/Add/Triad) found in output")
    return results


def parse_mlc_matrix(text: str) -> dict[str, dict[str, float]]:
    """Parse an Intel MLC ``--bandwidth_matrix``/``--latency_matrix`` NUMA table.

    Returns ``{"<row_node>": {"<col_node>": value, ...}, ...}``. Raises
    ValueError if no column header or no data rows are found.
    """

    columns: list[str] | None = None
    matrix: dict[str, dict[str, float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        col_match = _MLC_COLUMN_HEADER.match(line)
        if col_match:
            columns = col_match["cols"].split()
            continue
        if columns is None:
            continue
        row_match = _MLC_ROW.match(line)
        if not row_match:
            continue
        values = row_match["values"].split()
        if len(values) != len(columns):
            continue
        matrix[row_match["row"]] = dict(zip(columns, (float(v) for v in values), strict=True))
    if columns is None or not matrix:
        raise ValueError("no NUMA bandwidth/latency matrix found in MLC output")
    return matrix


def topology_snapshot() -> dict[str, Any]:
    """Capture the same NUMA/HBM topology evidence the LLM-run harness records."""

    lscpu = lscpu_values()
    nodes = numa_nodes()
    socket_count: int | None
    try:
        socket_count = int(lscpu["Socket(s)"])
    except (KeyError, ValueError):
        socket_count = None
    mode, method = detect_xeon_max_hbm_mode(lscpu.get("Model name"), nodes, socket_count)
    return {
        "cpu_model": lscpu.get("Model name"),
        "socket_count": socket_count,
        "hbm_mode_detected": mode,
        "hbm_mode_detection_method": method,
        "numa_nodes": [node.to_dict() for node in nodes],
    }


def build_membench_record(
    *,
    tool: str,
    target: str,
    parameters: dict[str, Any],
    results: dict[str, Any],
    host: str | None = None,
    cpu_bind: str | None = None,
    mem_bind: str | None = None,
    threads: int | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble one microbenchmark run's record, ready to write to disk."""

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": make_run_id(f"membench-{tool}-{target}"),
        "timestamp": utc_timestamp(),
        "tool": tool,
        "target": target,
        "host": host or platform.node(),
        "cpu_bind": cpu_bind,
        "mem_bind": mem_bind,
        "threads": threads,
        "parameters": parameters,
        "results": results,
        "warnings": warnings or [],
        "topology": topology_snapshot(),
    }


def write_membench_result(record: dict[str, Any], output_dir: str | Path) -> Path:
    """Write one record as its own JSON file under ``output_dir``."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_kv(pairs: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected key=value, got {pair!r}")
        parsed[key] = value
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse STREAM/MLC output and write a membench result record."
    )
    parser.add_argument("--tool", required=True, choices=["stream", "mlc-bandwidth", "mlc-latency"])
    parser.add_argument("--target", required=True, choices=["cache", "flat"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-bind")
    parser.add_argument("--mem-bind")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--host")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="key=value, repeatable; recorded verbatim under parameters",
    )
    parser.add_argument(
        "--input",
        help="path to the tool's captured stdout; defaults to reading stdin",
    )
    args = parser.parse_args(argv)

    text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()

    warnings: list[str] = []
    try:
        if args.tool == "stream":
            results: dict[str, Any] = parse_stream_output(text)
        else:
            results = parse_mlc_matrix(text)
    except ValueError as exc:
        results = {}
        warnings.append(str(exc))

    record = build_membench_record(
        tool="mlc" if args.tool.startswith("mlc") else args.tool,
        target=args.target,
        parameters={"kind": args.tool, **_parse_kv(args.param)},
        results=results,
        host=args.host,
        cpu_bind=args.cpu_bind,
        mem_bind=args.mem_bind,
        threads=args.threads,
        warnings=warnings,
    )
    path = write_membench_result(record, args.output_dir)
    print(path)
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
