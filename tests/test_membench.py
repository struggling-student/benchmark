from __future__ import annotations

import json

import pytest

from llm_bench.hardware import NumaNode
from llm_bench.membench import (
    build_membench_record,
    parse_mlc_matrix,
    parse_stream_output,
    topology_snapshot,
    write_membench_result,
)

STREAM_SAMPLE = """\
-------------------------------------------------------------
STREAM version $Revision: 5.10 $
-------------------------------------------------------------
This system uses 8 bytes per array element.
-------------------------------------------------------------
Array size = 90000000 (elements), Offset = 0 (elements)
Memory per array = 686.6 MiB (= 0.7 GiB).
Total memory required = 2059.7 MiB (= 2.0 GiB).
Each kernel will be executed 10 times.
-------------------------------------------------------------
Function    Best Rate MB/s  Avg time     Min time     Max time
Copy:          45231.2     0.031873     0.031802     0.031944
Scale:         44987.6     0.032100     0.032009     0.032201
Add:           47850.3     0.045200     0.045102     0.045301
Triad:         48012.9     0.045012     0.044980     0.045055
-------------------------------------------------------------
Solution Validates: avg error less than 1.000000e-13 on all three arrays
-------------------------------------------------------------
"""

MLC_BANDWIDTH_SAMPLE = """\
Intel(R) Memory Latency Checker - v3.11
Measuring Memory Bandwidths between nodes within system
Using all the threads from each core if Hyper-threading is enabled
Using Read-only traffic type
                Numa node
Numa node            0       1       2       3
       0         45000   21000   88000   19000
       1         20800   45500   18500   87500
       2         87200   18900   92000   17800
       3         19100   86800   17500   93500
"""


def test_parse_stream_output_extracts_all_four_rows() -> None:
    results = parse_stream_output(STREAM_SAMPLE)

    assert set(results) == {"copy", "scale", "add", "triad"}
    assert results["triad"]["best_mbps"] == pytest.approx(48012.9)
    assert results["copy"]["avg_time_s"] == pytest.approx(0.031873)


def test_parse_stream_output_rejects_output_with_no_rows() -> None:
    with pytest.raises(ValueError, match="no STREAM result rows"):
        parse_stream_output("nothing useful here\n")


def test_parse_mlc_matrix_reads_the_numa_bandwidth_table() -> None:
    matrix = parse_mlc_matrix(MLC_BANDWIDTH_SAMPLE)

    assert matrix["0"]["0"] == pytest.approx(45000.0)
    assert matrix["2"]["2"] == pytest.approx(92000.0)
    assert matrix["1"]["3"] == pytest.approx(87500.0)


def test_parse_mlc_matrix_rejects_output_with_no_table() -> None:
    with pytest.raises(ValueError, match="no NUMA bandwidth/latency matrix"):
        parse_mlc_matrix("mlc: command not found\n")


def test_topology_snapshot_reuses_hardware_detection(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_bench.membench.lscpu_values",
        lambda: {"Model name": "Intel(R) Xeon(R) CPU Max 9480", "Socket(s)": "2"},
    )
    monkeypatch.setattr(
        "llm_bench.membench.numa_nodes",
        lambda: (
            NumaNode(0, "0-55", 504 * 1024**2),
            NumaNode(1, "56-111", 504 * 1024**2),
            NumaNode(2, "", 64 * 1024**2),
            NumaNode(3, "", 64 * 1024**2),
        ),
    )

    snapshot = topology_snapshot()

    assert snapshot["hbm_mode_detected"] == "flat"
    assert snapshot["socket_count"] == 2
    assert len(snapshot["numa_nodes"]) == 4


def test_build_and_write_membench_record_round_trips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_bench.membench.lscpu_values",
        lambda: {"Model name": "Intel(R) Xeon(R) CPU Max 9480", "Socket(s)": "2"},
    )
    monkeypatch.setattr(
        "llm_bench.membench.numa_nodes",
        lambda: (NumaNode(0, "0-55", 504 * 1024**2), NumaNode(2, "", 64 * 1024**2)),
    )

    record = build_membench_record(
        tool="stream",
        target="flat",
        parameters={"array_size": "below_cliff"},
        results=parse_stream_output(STREAM_SAMPLE),
        cpu_bind="0",
        mem_bind="2",
        threads=56,
    )
    path = write_membench_result(record, tmp_path)

    assert path.parent == tmp_path
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["tool"] == "stream"
    assert on_disk["target"] == "flat"
    assert on_disk["mem_bind"] == "2"
    assert on_disk["results"]["triad"]["best_mbps"] == pytest.approx(48012.9)
    assert on_disk["warnings"] == []
