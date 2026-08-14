from __future__ import annotations

from typing import Any

from llm_bench.cli import _dashboard, build_parser


def test_dashboard_command_defaults_to_demo_and_minimal_toolbar(
    monkeypatch: Any,
) -> None:
    args = build_parser().parse_args(["dashboard"])
    captured: list[str] = []

    def fake_call(command: list[str]) -> int:
        captured.extend(command)
        return 0

    monkeypatch.setattr("llm_bench.cli.subprocess.call", fake_call)

    assert args.results_root is None
    assert _dashboard(args) == 0
    assert captured[captured.index("--client.toolbarMode") + 1] == "minimal"
    assert "--results-root" not in captured

