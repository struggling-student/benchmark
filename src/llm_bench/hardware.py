"""Runtime CPU capability and HBM topology discovery.

The profile describes the hardware treatment that a run expects.  This module
collects independent evidence from the execution node so the runner can reject
an allocation whose ISA or HBM mode does not match that treatment.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigurationError, ExperimentConfig

CPU_ISA_REQUIREMENTS: dict[str, frozenset[str]] = {
    "auto": frozenset(),
    "avx2": frozenset({"avx", "avx2"}),
    "avx512": frozenset({"avx512f"}),
    "amx": frozenset({"amx_tile"}),
}

_REPORTED_CPU_FEATURES = frozenset(
    {
        "avx",
        "avx2",
        "avx_vnni",
        "avx512f",
        "avx512bw",
        "avx512cd",
        "avx512dq",
        "avx512vl",
        "avx512_vbmi2",
        "avx512_vnni",
        "avx512_bf16",
        "avx512_fp16",
        "amx_tile",
        "amx_int8",
        "amx_bf16",
        "asimd",
        "f16c",
        "fma",
        "sve",
        "sve2",
    }
)
_NODE_DIRECTORY = re.compile(r"node(?P<node_id>[0-9]+)$")


@dataclass(frozen=True, slots=True)
class NumaNode:
    node_id: int
    cpus: str
    memory_total_kib: int | None

    @property
    def has_cpus(self) -> bool:
        return bool(self.cpus.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cpus": self.cpus,
            "has_cpus": self.has_cpus,
            "memory_total_gib": (
                self.memory_total_kib / (1024**2)
                if self.memory_total_kib is not None
                else None
            ),
        }


def _run(command: tuple[str, ...]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def lscpu_values() -> dict[str, str]:
    """Return normalized lscpu fields, with a plain-text fallback."""

    output = _run(("lscpu", "--json"))
    if output:
        try:
            rows = json.loads(output).get("lscpu", [])
            return {
                str(row["field"]).rstrip(":"): str(row["data"]).strip()
                for row in rows
                if row.get("field") and row.get("data") is not None
            }
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            pass
    plain = _run(("lscpu",))
    if not plain:
        return {}
    values: dict[str, str] = {}
    for line in plain.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def cpu_flags(lscpu: dict[str, str]) -> frozenset[str]:
    return frozenset((lscpu.get("Flags") or lscpu.get("Features") or "").split())


def cpu_isa_description(lscpu: dict[str, str]) -> str | None:
    architecture = lscpu.get("Architecture") or platform.machine() or None
    selected = sorted(cpu_flags(lscpu) & _REPORTED_CPU_FEATURES)
    if architecture and selected:
        return f"{architecture}:" + ",".join(selected)
    return architecture


def _node_memory_total_kib(path: Path) -> int | None:
    try:
        contents = (path / "meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^Node\s+[0-9]+\s+MemTotal:\s+([0-9]+)\s+kB$", contents, re.MULTILINE)
    return int(match.group(1)) if match else None


def numa_nodes(root: str | Path = "/sys/devices/system/node") -> tuple[NumaNode, ...]:
    """Read Linux NUMA nodes without requiring numactl or elevated access."""

    base = Path(root)
    nodes: list[NumaNode] = []
    for path in base.glob("node[0-9]*"):
        match = _NODE_DIRECTORY.fullmatch(path.name)
        if match is None:
            continue
        try:
            cpus = (path / "cpulist").read_text(encoding="utf-8").strip()
        except OSError:
            cpus = ""
        nodes.append(
            NumaNode(
                node_id=int(match.group("node_id")),
                cpus=cpus,
                memory_total_kib=_node_memory_total_kib(path),
            )
        )
    return tuple(sorted(nodes, key=lambda node: node.node_id))


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def detect_xeon_max_hbm_mode(
    cpu_model: str | None,
    nodes: tuple[NumaNode, ...],
    socket_count: int | None,
) -> tuple[str | None, str | None]:
    """Infer Xeon Max HBM mode from OS-visible memory topology.

    Intel Xeon Max flat mode exposes HBM and DDR as separate NUMA nodes. Cache
    mode exposes only DDR because HBM is a transparent memory-side cache.  An
    HBM-only system has no DDR and is distinguished conservatively by its
    visible capacity (64 GiB HBM per socket, with margin for firmware/kernel
    reservations).
    """

    if not cpu_model or not re.search(
        r"Xeon(?:\(R\))?.*CPU Max", cpu_model, re.IGNORECASE
    ):
        return None, None
    memory_nodes = [node for node in nodes if (node.memory_total_kib or 0) > 0]
    memory_only = [node for node in memory_nodes if not node.has_cpus]
    if memory_only:
        return "flat", "xeon_max_memory_only_numa_nodes"
    total_kib = sum(node.memory_total_kib or 0 for node in memory_nodes)
    sockets = socket_count or 1
    # Xeon CPU Max provides 64 GiB HBM per socket.  A 96 GiB/socket ceiling
    # leaves room for topology/reporting variation while remaining well below
    # normal DDR-backed cache-mode configurations.
    if memory_nodes and total_kib <= sockets * 96 * 1024**2:
        return "hbm_only", "xeon_max_hbm_capacity_only"
    if memory_nodes:
        return "cache", "xeon_max_no_memory_only_numa_nodes"
    return None, None


def _parse_node_spec(value: str) -> set[int] | None:
    """Parse the numeric subset of numactl's node-list grammar."""

    selected: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            return None
        if item.isdigit():
            selected.add(int(item))
            continue
        match = re.fullmatch(r"([0-9]+)-([0-9]+)", item)
        if match is None:
            return None
        start, end = (int(part) for part in match.groups())
        if end < start:
            return None
        selected.update(range(start, end + 1))
    return selected


def resolve_memory_binding(
    config: ExperimentConfig, report: dict[str, Any] | None = None
) -> str | None:
    """Resolve symbolic flat-mode tiers to the node's numeric NUMA node list."""

    requested = config.memory_binding
    if not requested:
        return None
    tier = requested.strip().lower()
    if tier not in {"hbm", "ddr"}:
        return requested
    if config.memory_mode != "flat":
        raise ConfigurationError(
            f"symbolic memory_binding {tier!r} is valid only in flat mode"
        )
    if report is None:
        nodes = numa_nodes()
        selected = [
            node.node_id
            for node in nodes
            if (not node.has_cpus if tier == "hbm" else node.has_cpus)
        ]
    else:
        selected = list(report[f"{tier}_numa_nodes"])
    if not selected:
        raise ConfigurationError(f"no addressable {tier.upper()} NUMA nodes were detected")
    return ",".join(str(node) for node in sorted(selected))


def inspect_hardware(config: ExperimentConfig) -> dict[str, Any]:
    lscpu = lscpu_values()
    flags = cpu_flags(lscpu)
    nodes = numa_nodes()
    socket_count = _optional_int(lscpu.get("Socket(s)"))
    if config.hardware_type in {"cpu", "hybrid"}:
        mode, mode_method = detect_xeon_max_hbm_mode(
            lscpu.get("Model name") or platform.processor() or None,
            nodes,
            socket_count,
        )
    else:
        mode, mode_method = None, None
    target = config.cpu_isa_target
    required = set(config.cpu_features_required)
    if target:
        required.update(CPU_ISA_REQUIREMENTS[target])
    missing = sorted(required - flags)
    memory_only_nodes = [node.node_id for node in nodes if not node.has_cpus]
    cpu_memory_nodes = [node.node_id for node in nodes if node.has_cpus]
    requested_mode = config.memory_mode if config.memory_mode in {"flat", "cache"} else None
    return {
        "cpu_isa": cpu_isa_description(lscpu),
        "cpu_isa_target": target,
        "cpu_features_required": sorted(required),
        "cpu_features_detected": sorted(flags & (_REPORTED_CPU_FEATURES | required)),
        "cpu_features_missing": missing,
        "cpu_isa_verified": None if not required else not missing,
        "memory_mode_requested": requested_mode,
        "memory_mode_detected": mode,
        "memory_mode_detection_method": mode_method,
        "memory_mode_verified": None if requested_mode is None else requested_mode == mode,
        "numa_nodes": [node.to_dict() for node in nodes],
        "hbm_numa_nodes": memory_only_nodes if mode == "flat" else [],
        "ddr_numa_nodes": cpu_memory_nodes if mode in {"flat", "cache"} else [],
    }


def validate_hardware(config: ExperimentConfig) -> tuple[dict[str, Any], list[str]]:
    """Validate configured CPU/ISA/HBM expectations on the execution node."""

    report = inspect_hardware(config)
    report["memory_binding_requested"] = config.memory_binding
    report["memory_binding_resolved"] = None
    checks = [f"hardware target: {config.hardware_type}"]
    if config.hardware_type not in {"cpu", "hybrid"}:
        return report, checks

    target = config.cpu_isa_target
    missing = report["cpu_features_missing"]
    if report["cpu_features_required"]:
        if missing:
            raise ConfigurationError(
                f"CPU ISA target {target!r} is unavailable; missing feature(s): "
                + ", ".join(missing)
            )
        checks.append(
            f"CPU ISA target verified: {target or 'custom'} "
            f"({', '.join(report['cpu_features_required'])})"
        )

    requested_mode = report["memory_mode_requested"]
    if requested_mode:
        detected_mode = report["memory_mode_detected"]
        if detected_mode is None:
            raise ConfigurationError(
                f"cannot verify requested HBM mode {requested_mode!r} on this CPU/topology"
            )
        if detected_mode != requested_mode:
            raise ConfigurationError(
                f"HBM mode mismatch: profile requests {requested_mode!r}, "
                f"execution node reports {detected_mode!r}"
            )
        checks.append(
            f"HBM mode verified: {detected_mode} "
            f"({report['memory_mode_detection_method']})"
        )

    if config.memory_binding:
        resolved_binding = resolve_memory_binding(config, report)
        report["memory_binding_resolved"] = resolved_binding
    else:
        resolved_binding = None

    if resolved_binding and requested_mode == "flat":
        selected = _parse_node_spec(resolved_binding)
        if selected is None:
            raise ConfigurationError(
                "flat-mode memory_binding must be 'hbm', 'ddr', or an explicit numeric "
                "NUMA node list such as '2,3' or '8-15'"
            )
        available = {
            int(node["node_id"])
            for node in report["numa_nodes"]
            if isinstance(node, dict) and node.get("node_id") is not None
        }
        unknown = sorted(selected - available)
        if unknown:
            raise ConfigurationError(
                "memory_binding names unavailable NUMA node(s): "
                + ", ".join(str(node) for node in unknown)
            )
        memory_type = (config.memory_type or "").lower()
        pure_hbm = "hbm" in memory_type and "ddr" not in memory_type
        pure_ddr = "ddr" in memory_type and "hbm" not in memory_type
        expected_nodes = set(
            report["hbm_numa_nodes"] if pure_hbm else report["ddr_numa_nodes"]
        )
        if (pure_hbm or pure_ddr) and not selected <= expected_nodes:
            tier = "HBM" if pure_hbm else "DDR"
            raise ConfigurationError(
                f"memory_binding {config.memory_binding!r} does not select only detected "
                f"{tier} NUMA nodes: {sorted(expected_nodes)}"
            )
        checks.append(
            f"NUMA memory binding verified: {config.memory_binding} -> {resolved_binding}"
        )
    return report, checks
