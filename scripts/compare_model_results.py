#!/usr/bin/env python3
"""Compare exactly two explicitly selected benchmark result summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_bench.results import ResultError, compare_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare inference-system metrics from two explicit summary.json files or "
            "result directories. This does not evaluate model quality."
        )
    )
    parser.add_argument("left", type=Path, help="first summary.json or result directory")
    parser.add_argument("right", type=Path, help="second summary.json or result directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for comparison.json and comparison.csv",
    )
    args = parser.parse_args()
    try:
        comparison, json_path, csv_path = compare_paths(args.left, args.right, args.output_dir)
    except (OSError, ResultError, ValueError) as exc:
        parser.error(str(exc))
    print(f"compatibility: {comparison['compatibility']['status']}")
    if comparison["compatibility"]["mismatched_fields"]:
        print("mismatched fields:")
        for mismatch in comparison["compatibility"]["mismatched_fields"]:
            print(f"  {mismatch['field']}: {mismatch['left']!r} != {mismatch['right']!r}")
    if comparison["compatibility"]["missing_fields"]:
        print("missing compatibility evidence (comparison is partial):")
        for missing in comparison["compatibility"]["missing_fields"]:
            print(f"  {missing['field']}: left={missing['left']!r}, right={missing['right']!r}")
    if comparison["compatibility"]["failed_runs"]:
        print("one or both runs did not complete successfully; see comparison.json")
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
