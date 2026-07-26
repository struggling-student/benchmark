"""Streamlit entry point for read-only benchmark result exploration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from llm_bench.catalog import CatalogEntry, ResultCatalog, catalog_row, discover_results
from llm_bench.dashboard_data import (
    DashboardDataset,
    combine_datasets,
    dataset_from_catalog,
    hardware_name,
    hardware_platform,
    measurements_for_entry,
    memory_system,
    simulated_dataset,
    telemetry_for_entry,
)
from llm_bench.interpretation import comparison_narrative, repetition_variability
from llm_bench.measurements import read_telemetry_series, resolve_recorded_path
from llm_bench.metrics import METRIC_SPECS, MetricSpec
from llm_bench.results import (
    COMPATIBILITY_FIELDS,
    CPU_COMPATIBILITY_FIELDS,
    ResultError,
    compare_summaries,
)

PLATFORM_COLORS = {
    "GPU": "#7257E8",
    "CPU · HBM": "#0E9F9A",
    "CPU · DDR": "#E58B32",
    "CPU": "#667085",
    "UNKNOWN": "#98A2B3",
}
STATUS_COLORS = {
    "completed": "#12B76A",
    "failed": "#F04438",
    "running": "#2E90FA",
    "partial": "#F79009",
}

_HEADLINE_FIELDS = {
    "smoke": (
        "model_load_time_seconds",
        "output_throughput_tokens_per_second",
        "duration_seconds",
        "energy_per_output_token_joules",
    ),
    "offline": (
        "output_throughput_tokens_per_second",
        "total_throughput_tokens_per_second",
        "duration_seconds",
        "energy_per_output_token_joules",
    ),
    "serving": (
        "request_throughput_requests_per_second",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "energy_per_request_joules",
    ),
}

_TELEMETRY_SPECS = {
    "average_gpu_utilization_percent": ("GPU utilization", "%", "GPU"),
    "total_gpu_memory_mib": ("GPU memory used", "MiB", "GPU"),
    "total_gpu_power_watts": ("GPU power", "W", "GPU"),
    "average_cpu_utilization_percent": ("CPU utilization", "%", "CPU"),
    "total_cpu_memory_mib": ("CPU memory used", "MiB", "CPU"),
    "total_cpu_power_watts": ("CPU package power", "W", "CPU"),
    "memory_bandwidth_gbps": ("Memory bandwidth", "GB/s", "Memory"),
}

_MEMORY_TREATMENT_FIELDS = {
    "memory_type",
    "memory_mode",
    "memory_binding",
}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _short_model(summary: Mapping[str, Any]) -> str:
    scale = summary.get("model_parameter_scale")
    if scale:
        return str(scale)
    model = str(summary.get("model_id") or "Unknown model")
    return model.rsplit("/", 1)[-1]


def _entry_label(entry: CatalogEntry) -> str:
    summary = entry.summary
    simulated = "SIM · " if summary.get("is_simulated") else ""
    timestamp = str(summary.get("timestamp") or "")[:16].replace("T", " ")
    run_id = str(summary.get("run_id") or "run")
    if len(run_id) > 34:
        run_id = f"{run_id[:20]}…{run_id[-10:]}"
    return (
        f"{simulated}{timestamp} · {_short_model(summary)} · "
        f"{summary.get('benchmark_type')} · {hardware_platform(summary)} · {run_id}"
    )


def _format_number(value: Any, unit: str = "") -> str:
    number = _finite(value)
    if number is None:
        return "—"
    display_unit = unit
    if unit == "MiB" and abs(number) >= 1024:
        number /= 1024
        display_unit = "GiB"
    if abs(number) >= 10_000:
        text = f"{number:,.0f}"
    elif abs(number) >= 100:
        text = f"{number:,.1f}"
    elif abs(number) >= 10:
        text = f"{number:,.2f}"
    elif abs(number) >= 1:
        text = f"{number:,.3f}"
    else:
        text = f"{number:.4g}"
    return f"{text} {display_unit}".strip()


def _metric_card_value(value: Any, unit: str) -> tuple[str, str]:
    formatted = _format_number(value, unit)
    if formatted == "—" or not unit:
        return formatted, unit
    number, display_unit = formatted.rsplit(" ", 1)
    return number, display_unit


def _metric_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for spec in METRIC_SPECS:
        if (value := _finite(summary.get(spec.field))) is not None:
            rows.append(
                {
                    "category": spec.category,
                    "metric": spec.label,
                    "value": value,
                    "display": _format_number(value, spec.unit),
                    "unit": spec.unit,
                    "aggregation": spec.aggregation,
                    "direction": spec.direction,
                    "scope": spec.scope,
                }
            )
    return rows


def _available_specs(entries: Sequence[CatalogEntry]) -> list[MetricSpec]:
    return [
        spec
        for spec in METRIC_SPECS
        if any(_finite(entry.summary.get(spec.field)) is not None for entry in entries)
    ]


def _spec(field: str) -> MetricSpec | None:
    return next((spec for spec in METRIC_SPECS if spec.field == field), None)


def _default_spec(specs: Sequence[MetricSpec], benchmark_type: str) -> int:
    preferred = (
        "output_throughput_tokens_per_second"
        if benchmark_type != "serving"
        else "request_throughput_requests_per_second"
    )
    return next((index for index, spec in enumerate(specs) if spec.field == preferred), 0)


def _apply_chart_style(figure: Any, *, height: int = 390) -> Any:
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin={"l": 20, "r": 20, "t": 62, "b": 92},
        font={"family": "Inter, ui-sans-serif, system-ui", "color": "#344054"},
        title_font={"size": 17, "color": "#101828"},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "left",
            "x": 0,
            "title": None,
            "font": {"size": 11},
        },
        hoverlabel={"font_size": 13},
    )
    figure.update_xaxes(showgrid=False, linecolor="#EAECF0")
    figure.update_yaxes(gridcolor="#EAECF0", zerolinecolor="#D0D5DD")
    return figure


def _inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        :root {
            --bench-ink: #101828;
            --bench-muted: #667085;
            --bench-border: #e4e7ec;
            --bench-violet: #7257e8;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 82% -10%, rgba(114,87,232,.10), transparent 30rem),
                #f8fafc;
            color: var(--bench-ink);
        }
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3 {
            color: var(--bench-ink);
        }
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
            color: var(--bench-muted);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1560px;
            padding-top: 1.8rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--bench-border);
            background: rgba(255,255,255,.92);
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [role="radiogroup"] p,
        [data-testid="stSidebar"] details > summary p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #344054;
        }
        [data-testid="stMetric"] {
            min-height: 112px;
            padding: 1.05rem 1.1rem;
            border: 1px solid var(--bench-border);
            border-radius: 16px;
            background: rgba(255,255,255,.92);
            box-shadow: 0 1px 2px rgba(16,24,40,.04);
        }
        [data-testid="stMetricLabel"] {
            color: var(--bench-muted);
        }
        [data-testid="stMetricLabel"] p {
            white-space: normal;
            line-height: 1.2;
        }
        [data-testid="stMetricValue"] {
            color: var(--bench-ink);
            letter-spacing: -.025em;
        }
        div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
            border: 1px solid var(--bench-border);
            border-radius: 16px;
            overflow: hidden;
            background: white;
        }
        .bench-kicker {
            color: var(--bench-violet);
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: -.45rem;
        }
        .bench-page {
            margin: .35rem 0 1.1rem;
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--bench-border);
            border-radius: 18px;
            background: linear-gradient(120deg, #fff, #fbfaff);
        }
        .bench-page h2 {
            color: var(--bench-ink);
            font-size: 1.55rem;
            letter-spacing: -.025em;
            margin: 0 0 .25rem;
        }
        .bench-page p {
            color: var(--bench-muted);
            margin: 0;
            max-width: 78ch;
        }
        .bench-run-card {
            padding: .9rem 1rem;
            border: 1px solid var(--bench-border);
            border-left: 4px solid var(--bench-violet);
            border-radius: 12px;
            background: white;
            color: var(--bench-muted);
            margin: .25rem 0 .75rem;
        }
        .bench-run-card strong { color: var(--bench-ink); }
        .bench-sim {
            padding: .8rem 1rem;
            border: 1px solid #fedf89;
            border-radius: 12px;
            background: #fffaeb;
            color: #7a2e0e;
            margin: .25rem 0 1rem;
        }
        .bench-measured {
            padding: .8rem 1rem;
            border: 1px solid #b2ddff;
            border-radius: 12px;
            background: #eff8ff;
            color: #175cd3;
            margin: .25rem 0 1rem;
        }
        .bench-section {
            color: #475467;
            font-size: .76rem;
            font-weight: 750;
            letter-spacing: .1em;
            text-transform: uppercase;
            margin: 1.2rem 0 .45rem;
        }
        button[kind="primary"] { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_header(st: Any, eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        (
            '<div class="bench-page">'
            f'<div class="bench-kicker">{eyebrow}</div>'
            f"<h2>{title}</h2><p>{description}</p></div>"
        ),
        unsafe_allow_html=True,
    )


def _simulated_notice(st: Any, entries: Sequence[CatalogEntry]) -> None:
    simulated = sum(bool(entry.summary.get("is_simulated")) for entry in entries)
    if simulated:
        st.markdown(
            (
                '<div class="bench-sim"><strong>Simulated preview data is visible.</strong> '
                f"{simulated} run(s) are deterministic UI fixtures, not benchmark evidence. "
                "They are excluded from the measured catalog export.</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            (
                '<div class="bench-measured"><strong>Measured evidence only.</strong> '
                "All visible runs come from the read-only filesystem catalog.</div>"
            ),
            unsafe_allow_html=True,
        )


def _filter_entries(
    st: Any, dataset: DashboardDataset, *, namespace: str
) -> list[CatalogEntry]:
    catalog = dataset.catalog
    rows = {entry.key: catalog_row(entry) for entry in catalog.entries}

    def widget_key(name: str) -> str:
        return f"{name}_{namespace}"

    def values(field: str) -> list[str]:
        return sorted(
            {str(row[field]) for row in rows.values() if row.get(field) not in (None, "")}
        )

    with st.sidebar.expander("Filters", expanded=True):
        platforms = st.multiselect(
            "Platform",
            values("platform"),
            default=values("platform"),
            key=widget_key("filter_platform"),
        )
        benchmark_types = st.multiselect(
            "Benchmark type",
            values("benchmark_type"),
            default=values("benchmark_type"),
            key=widget_key("filter_benchmark"),
        )
        models = st.multiselect(
            "Model",
            values("model_id"),
            default=values("model_id"),
            key=widget_key("filter_model"),
        )
        memory = st.multiselect(
            "Memory system",
            values("memory_type"),
            default=values("memory_type"),
            key=widget_key("filter_memory"),
        )
        statuses = st.multiselect(
            "Status",
            values("status"),
            default=values("status"),
            key=widget_key("filter_status"),
        )
        precision = st.multiselect(
            "Precision",
            values("precision"),
            default=values("precision"),
            key=widget_key("filter_precision"),
        )
        sources = st.multiselect(
            "Evidence source",
            values("source"),
            default=values("source"),
            key=widget_key("filter_source"),
        )
        workload_query = (
            st.text_input(
                "Workload contains",
                help="Examples: input=256, concurrency=8, or batch=8",
                key=widget_key("filter_workload"),
            )
            .strip()
            .lower()
        )
        st.caption("Clearing a filter means “all values” for that dimension.")

    selections = {
        "platform": set(platforms),
        "benchmark_type": set(benchmark_types),
        "model_id": set(models),
        "memory_type": set(memory),
        "status": set(statuses),
        "precision": set(precision),
        "source": set(sources),
    }
    filtered = []
    for entry in catalog.entries:
        row = rows[entry.key]
        if any(
            selected and str(row.get(field)) not in selected
            for field, selected in selections.items()
        ):
            continue
        if workload_query and workload_query not in str(row.get("workload", "")).lower():
            continue
        filtered.append(entry)
    return filtered


def _analysis_frame(pd: Any, entries: Sequence[CatalogEntry]) -> Any:
    rows = []
    for entry in entries:
        row = catalog_row(entry)
        row.update(
            {
                spec.field: _finite(entry.summary.get(spec.field))
                for spec in METRIC_SPECS
            }
        )
        peak = entry.summary.get("peak_gpu_memory_mib")
        if peak is None:
            peak = entry.summary.get("peak_cpu_memory_mib")
        row["peak_memory_gib"] = float(peak) / 1024 if _finite(peak) is not None else None
        row["hardware_name"] = hardware_name(entry.summary)
        rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "run_id",
            "timestamp",
            "status",
            "benchmark_type",
            "model_id",
            "model_scale",
            "platform",
            "hardware",
            "memory_type",
            "precision",
            "source",
            "workload",
            "input_length",
            "output_length",
            "number_of_requests",
            "request_rate",
            "maximum_concurrency",
            "batch_size",
            "thread_count",
            "numa_policy",
            "memory_binding",
            "peak_memory_gib",
            "hardware_name",
            *(spec.field for spec in METRIC_SPECS),
        ]
    )


def _catalog_table(pd: Any, entries: Sequence[CatalogEntry]) -> Any:
    table = pd.DataFrame([catalog_row(entry) for entry in entries])
    if table.empty:
        return table
    return table.rename(
        columns={
            "run_id": "Run",
            "timestamp": "Timestamp",
            "status": "Status",
            "benchmark_type": "Benchmark",
            "model_id": "Model",
            "model_scale": "Scale",
            "platform": "Platform",
            "hardware": "Hardware",
            "memory_type": "Memory",
            "precision": "Precision",
            "source": "Source",
            "workload": "Workload",
            "run_directory": "Result directory",
        }
    )


def _render_catalog_diagnostics(st: Any, pd: Any, catalog: ResultCatalog) -> None:
    if not catalog.diagnostics:
        return
    with st.expander(f"Catalog diagnostics ({len(catalog.diagnostics)})"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Path": str(item.path), "Message": item.message}
                    for item in catalog.diagnostics
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _render_overview(
    st: Any,
    pd: Any,
    px: Any,
    entries: Sequence[CatalogEntry],
    dataset: DashboardDataset,
) -> None:
    _page_header(
        st,
        "Campaign pulse",
        "Benchmark overview",
        "A high-level view of run health, platform coverage, performance trade-offs, "
        "and metric completeness across the active dataset.",
    )
    _simulated_notice(st, entries)
    if not entries:
        st.info("No runs match the active data source and filters.")
        _render_catalog_diagnostics(st, pd, dataset.catalog)
        return

    completed = sum(entry.summary.get("status") == "completed" for entry in entries)
    models = len({entry.summary.get("model_id") for entry in entries})
    platforms = len({hardware_platform(entry.summary) for entry in entries})
    memory_systems = {
        memory_system(entry.summary)
        for entry in entries
        if memory_system(entry.summary) != "Unspecified"
    }
    columns = st.columns(5)
    columns[0].metric("Visible runs", len(entries))
    columns[1].metric("Completion rate", f"{completed / len(entries):.0%}")
    columns[2].metric("Models", models)
    columns[3].metric("Platforms", platforms)
    columns[4].metric("Memory systems", len(memory_systems))

    frame = _analysis_frame(pd, entries)
    left, right = st.columns((1.65, 1))
    tradeoff = frame.dropna(
        subset=["output_throughput_tokens_per_second", "mean_e2e_latency_ms"]
    )
    with left:
        st.markdown(
            '<div class="bench-section">Performance trade-off</div>',
            unsafe_allow_html=True,
        )
        if tradeoff.empty:
            figure = px.scatter(
                pd.DataFrame({"Latency": [], "Throughput": []}),
                x="Latency",
                y="Throughput",
                title="Output throughput vs. mean end-to-end latency",
            )
            figure.add_annotation(
                text=(
                    "No runs contain both output throughput<br>"
                    "and mean end-to-end latency."
                ),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 15, "color": "#667085"},
            )
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
            st.plotly_chart(
                _apply_chart_style(figure, height=430),
                width="stretch",
                theme=None,
            )
            st.caption(
                "The trade-off needs both metrics; unavailable latency is not treated as zero."
            )
        else:
            size = "peak_memory_gib" if tradeoff["peak_memory_gib"].notna().any() else None
            figure = px.scatter(
                tradeoff,
                x="mean_e2e_latency_ms",
                y="output_throughput_tokens_per_second",
                color="platform",
                symbol="benchmark_type",
                size=size,
                size_max=34,
                custom_data=["run_id", "model_scale", "memory_type", "workload", "source"],
                color_discrete_map=PLATFORM_COLORS,
                title="Output throughput vs. mean end-to-end latency",
                labels={
                    "mean_e2e_latency_ms": "Mean end-to-end latency (ms)",
                    "output_throughput_tokens_per_second": "Output throughput (tokens/s)",
                    "platform": "Platform",
                },
            )
            figure.update_traces(
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Model: %{customdata[1]}<br>"
                    "Memory: %{customdata[2]}<br>"
                    "Workload: %{customdata[3]}<br>"
                    "Source: %{customdata[4]}<br>"
                    "Latency: %{x:.3g} ms<br>"
                    "Throughput: %{y:.4g} tokens/s<extra></extra>"
                )
            )
            st.plotly_chart(
                _apply_chart_style(figure, height=430),
                width="stretch",
                theme=None,
            )
            st.caption(
                "Upper-left observations combine higher throughput with lower latency. "
                "Bubble size represents peak memory when available."
            )

    with right:
        st.markdown('<div class="bench-section">Campaign coverage</div>', unsafe_allow_html=True)
        coverage = (
            frame.groupby(["platform", "benchmark_type"], dropna=False)
            .size()
            .reset_index(name="runs")
        )
        figure = px.bar(
            coverage,
            x="platform",
            y="runs",
            color="benchmark_type",
            barmode="stack",
            title="Runs by platform and benchmark",
            labels={
                "platform": "Platform",
                "runs": "Run count",
                "benchmark_type": "Benchmark",
            },
            color_discrete_sequence=["#7257E8", "#0E9F9A", "#E58B32"],
        )
        st.plotly_chart(
            _apply_chart_style(figure, height=430),
            width="stretch",
            theme=None,
        )

    st.markdown('<div class="bench-section">Metric readiness</div>', unsafe_allow_html=True)
    readiness_fields = (
        "output_throughput_tokens_per_second",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "energy_per_output_token_joules",
        "measured_memory_bandwidth_gbps",
        "average_gpu_utilization_percent",
        "average_cpu_utilization_percent",
    )
    readiness_rows = []
    for platform, group in frame.groupby("platform"):
        for field in readiness_fields:
            spec = _spec(field)
            if spec is None:
                continue
            readiness_rows.append(
                {
                    "Platform": platform,
                    "Metric": spec.label,
                    "Available": 100 * group[field].notna().mean(),
                }
            )
    readiness = pd.DataFrame(readiness_rows)
    if not readiness.empty:
        matrix = readiness.pivot(index="Platform", columns="Metric", values="Available")
        figure = px.imshow(
            matrix,
            color_continuous_scale=[
                [0.0, "#F2F4F7"],
                [0.45, "#D9D6FE"],
                [1.0, "#7257E8"],
            ],
            range_color=(0, 100),
            text_auto=".0f",
            aspect="auto",
            title="Metric availability by platform (%)",
            labels={"color": "Available (%)"},
        )
        figure.update_traces(texttemplate="%{z:.0f}%")
        st.plotly_chart(
            _apply_chart_style(figure, height=310),
            width="stretch",
            theme=None,
        )
        st.caption(
            "Completeness is shown explicitly so unavailable instrumentation is not mistaken "
            "for a zero measurement."
        )

    st.markdown('<div class="bench-section">Run catalog</div>', unsafe_allow_html=True)
    table = _catalog_table(pd, entries)
    visible = [
        "Run",
        "Timestamp",
        "Status",
        "Benchmark",
        "Scale",
        "Platform",
        "Hardware",
        "Memory",
        "Precision",
        "Source",
        "Workload",
    ]
    visible = [field for field in visible if field in table]
    st.dataframe(table[visible], width="stretch", hide_index=True, height=390)

    measured = table[table["Source"] == "Measured"] if "Source" in table else table
    st.download_button(
        "Download measured catalog CSV",
        measured.to_csv(index=False).encode("utf-8"),
        file_name="benchmark_catalog_measured.csv",
        mime="text/csv",
        disabled=measured.empty,
        help="Simulated preview runs are deliberately excluded.",
        key="overview_catalog_download",
    )
    _render_catalog_diagnostics(st, pd, dataset.catalog)


def _load_entry_telemetry(
    st: Any,
    entry: CatalogEntry,
    dataset: DashboardDataset,
    completed_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | None]]:
    simulated = telemetry_for_entry(dataset, entry)
    if simulated is not None:
        return simulated

    telemetry_records = [record for record in completed_records if record.get("telemetry_file")]
    if not telemetry_records:
        return []
    telemetry_record = st.selectbox(
        "Telemetry repetition",
        telemetry_records,
        format_func=lambda record: f"Repetition {record['repetition']}",
        key="run_telemetry_repetition",
    )
    telemetry_path = resolve_recorded_path(
        telemetry_record["telemetry_file"], entry.run_directory
    )
    if telemetry_path is None or not telemetry_path.is_file():
        st.warning("The recorded telemetry file is unavailable after resolving copied paths.")
        return []
    try:
        return read_telemetry_series(telemetry_path)
    except ResultError as exc:
        st.warning(str(exc))
        return []


def _render_repetitions(
    st: Any,
    pd: Any,
    px: Any,
    entry: CatalogEntry,
    dataset: DashboardDataset,
) -> None:
    measurements, legacy = measurements_for_entry(dataset, entry)
    if legacy and measurements.get("warnings"):
        st.warning(measurements["warnings"][0])
    elif measurements.get("source") == "simulated-preview":
        st.caption("Five deterministic preview repetitions exercise the variability UI.")

    completed = [
        record for record in measurements.get("records", []) if record.get("status") == "completed"
    ]
    specs = [
        spec
        for spec in METRIC_SPECS
        if any(
            _finite(record.get("metrics", {}).get(spec.field)) is not None
            for record in completed
        )
    ]
    if not specs:
        st.info("No repetition-level metrics are available for this run.")
        return

    benchmark_type = str(entry.summary.get("benchmark_type") or "")
    selected = st.selectbox(
        "Repetition metric",
        specs,
        index=_default_spec(specs, benchmark_type),
        format_func=lambda spec: f"{spec.category} · {spec.label}",
        key="run_repetition_metric",
    )
    rows = [
        {
            "Repetition": record["repetition"],
            "Value": record["metrics"][selected.field],
        }
        for record in completed
        if _finite(record.get("metrics", {}).get(selected.field)) is not None
    ]
    frame = pd.DataFrame(rows)
    figure = px.line(
        frame,
        x="Repetition",
        y="Value",
        markers=True,
        title=f"{selected.label} across measured repetitions",
        labels={"Value": f"{selected.label} ({selected.unit})"},
        color_discrete_sequence=["#7257E8"],
    )
    figure.update_traces(marker={"size": 10}, line={"width": 2.5})
    aggregate = _finite(entry.summary.get(selected.field))
    if aggregate is not None and selected.aggregation in {"mean", "maximum"}:
        label = "summary mean" if selected.aggregation == "mean" else "summary maximum"
        figure.add_hline(
            y=aggregate,
            line_dash="dash",
            line_color="#475467",
            annotation_text=label,
        )
    st.plotly_chart(
        _apply_chart_style(figure),
        width="stretch",
        theme=None,
        key="run_repetition_chart",
    )
    variability = repetition_variability(measurements, selected.field)
    if selected.aggregation == "mean" and variability:
        st.caption(
            "Sample standard deviation: "
            f"{variability['standard_deviation']:.4g} {selected.unit}; "
            f"range {variability['minimum']:.4g}–{variability['maximum']:.4g}. "
            "This is descriptive, not a confidence interval."
        )
    elif selected.aggregation in {"sum", "weighted"}:
        st.caption(
            "The run-level aggregate is not overlaid because totals and weighted metrics "
            "are not directly comparable with a single repetition."
        )

    series = _load_entry_telemetry(st, entry, dataset, completed)
    telemetry_frame = pd.DataFrame(series)
    if not series:
        st.info(
            "No resource telemetry timeline is available; empty panels make the missing "
            "instrumentation explicit."
        )
    st.markdown('<div class="bench-section">Resource telemetry</div>', unsafe_allow_html=True)
    if entry.summary.get("hardware_type") == "gpu":
        telemetry_fields = (
            "average_gpu_utilization_percent",
            "total_gpu_memory_mib",
            "total_gpu_power_watts",
            "memory_bandwidth_gbps",
        )
    else:
        telemetry_fields = (
            "average_cpu_utilization_percent",
            "total_cpu_memory_mib",
            "total_cpu_power_watts",
            "memory_bandwidth_gbps",
        )
    columns = st.columns(2)
    for index, field in enumerate(telemetry_fields):
        label, unit, _ = _TELEMETRY_SPECS[field]
        available = (
            not telemetry_frame.empty
            and field in telemetry_frame
            and telemetry_frame[field].notna().any()
        )
        if available:
            data = telemetry_frame.dropna(subset=[field])
            figure = px.line(
                data,
                x="elapsed_seconds",
                y=field,
                title=label,
                labels={
                    "elapsed_seconds": "Elapsed time (s)",
                    field: f"{label} ({unit})",
                },
                color_discrete_sequence=[
                    (
                        "#7257E8"
                        if "gpu" in field
                        else "#0E9F9A"
                        if "cpu" in field
                        else "#E58B32"
                    )
                ],
            )
            figure.update_traces(line={"width": 2.4})
        else:
            figure = px.scatter(
                pd.DataFrame({"Elapsed": [], "Value": []}),
                x="Elapsed",
                y="Value",
                title=label,
            )
            figure.add_annotation(
                text="Not recorded for this run",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 14, "color": "#667085"},
            )
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
        with columns[index % 2]:
            st.plotly_chart(
                _apply_chart_style(figure, height=310),
                width="stretch",
                theme=None,
                key=f"run_telemetry_slot_{index}",
            )
    if entry.summary.get("hardware_type") == "gpu":
        st.caption(
            "GPU utilization is averaged across visible devices; memory and power are summed "
            "across the allocation."
        )
    else:
        st.caption(
            "CPU telemetry is shown only when the result records an applicable package/node "
            "instrumentation boundary."
        )


def _status_renderer(st: Any, status: str) -> Any:
    return {
        "completed": st.success,
        "failed": st.error,
        "running": st.info,
        "partial": st.warning,
    }.get(status, st.warning)


def _headline_specs(summary: Mapping[str, Any]) -> list[MetricSpec]:
    fields = list(_HEADLINE_FIELDS.get(str(summary.get("benchmark_type")), ()))
    if summary.get("hardware_type") == "cpu" and fields:
        fields[-1] = "measured_memory_bandwidth_gbps"
    fields.append(
        "peak_gpu_memory_mib"
        if summary.get("hardware_type") == "gpu"
        else "peak_cpu_memory_mib"
    )
    return [spec for field in fields[:5] if (spec := _spec(field)) is not None]


def _control_rows(
    summary: Mapping[str, Any], fields: Sequence[tuple[str, str]]
) -> list[dict[str, Any]]:
    rows = []
    for field, label in fields:
        value = summary.get(field)
        if value is None:
            continue
        if isinstance(value, (Mapping, list, tuple)):
            display = json.dumps(value, sort_keys=True)
        else:
            display = str(value)
        rows.append({"Control": label, "Recorded value": display})
    return rows


def _render_run_detail(
    st: Any,
    pd: Any,
    px: Any,
    entries: Sequence[CatalogEntry],
    dataset: DashboardDataset,
) -> None:
    _page_header(
        st,
        "Run microscope",
        "Run detail",
        "Inspect one run from headline outcomes down to repetitions, telemetry, "
        "placement controls, warnings, and the complete normalized record.",
    )
    if not entries:
        st.info("No runs match the active filters.")
        return

    entry = st.selectbox("Run", entries, format_func=_entry_label, key="detail_run")
    if st.session_state.get("_detail_entry_key") != entry.key:
        st.session_state["_detail_entry_key"] = entry.key
        st.session_state.pop("run_repetition_metric", None)
        st.session_state.pop("run_telemetry_repetition", None)
    summary = entry.summary
    status = str(summary.get("status") or "unknown")
    source = dataset.source_label(entry)
    st.markdown(
        (
            '<div class="bench-run-card">'
            f"<strong>{summary.get('run_id')}</strong> &nbsp;·&nbsp; {source} "
            f"&nbsp;·&nbsp; {hardware_platform(summary)} "
            f"&nbsp;·&nbsp; {hardware_name(summary)} "
            f"&nbsp;·&nbsp; {memory_system(summary)}</div>"
        ),
        unsafe_allow_html=True,
    )
    _status_renderer(st, status)(f"Run status: {status}")
    _simulated_notice(st, [entry])
    if summary.get("benchmark_type") == "smoke":
        st.info(
            "This is a functional smoke test. Use offline or serving campaigns for formal "
            "performance conclusions."
        )
    else:
        st.info(
            "This is a measured performance workload. Interpret it only against runs with "
            "matching workload, software, hardware, and placement controls."
        )

    headline = _headline_specs(summary)
    if headline:
        columns = st.columns(len(headline))
        for column, spec in zip(columns, headline, strict=True):
            value, unit = _metric_card_value(summary.get(spec.field), spec.unit)
            column.metric(
                f"{spec.label} ({unit})",
                value,
                help=(
                    f"Run aggregation: {spec.aggregation}. "
                    f"Preferred direction: {spec.direction}."
                ),
            )

    performance_tab, repetitions_tab, controls_tab, evidence_tab = st.tabs(
        ("Metrics", "Repetitions & telemetry", "Experiment controls", "Evidence"),
        key="run_detail_tabs",
    )
    with performance_tab:
        rows = _metric_rows(summary)
        frame = pd.DataFrame(
            rows,
            columns=(
                "category",
                "metric",
                "value",
                "display",
                "unit",
                "aggregation",
                "direction",
                "scope",
            ),
        )
        for category in (
            "Execution",
            "Throughput",
            "Latency",
            "Resources",
            "Memory",
            "Energy",
        ):
            st.markdown(
                f'<div class="bench-section">{category}</div>',
                unsafe_allow_html=True,
            )
            subset = frame[frame["category"] == category][
                ["metric", "display", "aggregation", "direction"]
            ].rename(
                columns={
                    "metric": "Metric",
                    "display": "Value",
                    "aggregation": "Run aggregation",
                    "direction": "Preferred direction",
                }
            )
            st.dataframe(
                subset,
                width="stretch",
                hide_index=True,
                key=f"run_metric_table_{category.lower()}",
            )

    with repetitions_tab:
        _render_repetitions(st, pd, px, entry, dataset)

    with controls_tab:
        workload_fields = (
            ("benchmark_type", "Benchmark"),
            ("input_length", "Input length"),
            ("output_length", "Output length"),
            ("number_of_requests", "Requests per repetition"),
            ("request_rate", "Request rate"),
            ("maximum_concurrency", "Maximum concurrency"),
            ("batch_size", "Batch size"),
            ("warmup_runs", "Warm-up runs"),
            ("repetitions", "Measured repetitions configured"),
            ("seed", "Seed"),
        )
        hardware_fields = (
            ("hardware_type", "Hardware type"),
            ("accelerator_name", "Accelerator"),
            ("accelerator_count", "Accelerator count"),
            ("cpu_model", "CPU model"),
            ("socket_count", "Sockets"),
            ("numa_node_count", "NUMA nodes"),
            ("memory_type", "Memory type"),
            ("memory_mode", "Memory mode"),
            ("memory_capacity_gib", "Memory capacity (GiB)"),
            ("thread_count", "Thread count"),
            ("process_count", "Process count"),
            ("thread_affinity", "Thread affinity"),
            ("numa_policy", "NUMA policy"),
            ("memory_binding", "Memory binding"),
            ("cpu_isa", "CPU ISA"),
            ("memory_bandwidth_instrument", "Bandwidth instrument"),
            ("memory_bandwidth_scope", "Bandwidth scope"),
            ("power_instrument", "Power instrument"),
            ("power_scope", "Power/energy scope"),
        )
        left, right = st.columns(2)
        with left:
            st.subheader("Workload")
            st.dataframe(
                pd.DataFrame(_control_rows(summary, workload_fields)),
                width="stretch",
                hide_index=True,
            )
        with right:
            st.subheader("Hardware & placement")
            rows = _control_rows(summary, hardware_fields)
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("No hardware-placement controls were recorded.")

    with evidence_tab:
        warnings = summary.get("warnings", [])
        if warnings:
            st.subheader(f"Warnings ({len(warnings)})")
            for warning in warnings:
                st.warning(str(warning))
        else:
            st.success("No run warnings were recorded.")
        st.subheader("Complete normalized summary")
        st.json(summary)
        if not dataset.is_simulated(entry):
            st.caption(f"Read-only result directory: {entry.run_directory.resolve()}")


def _single_metric_explorer(
    st: Any,
    pd: Any,
    px: Any,
    entries: Sequence[CatalogEntry],
    benchmark_type: str,
) -> None:
    selected_entries = [
        entry for entry in entries if entry.summary.get("benchmark_type") == benchmark_type
    ]
    specs = _available_specs(selected_entries)
    if not specs:
        st.info("No numeric metrics are available for these runs.")
        return
    categories = list(dict.fromkeys(spec.category for spec in specs))
    category = st.selectbox("Metric family", categories, key="explorer_category")
    category_specs = [spec for spec in specs if spec.category == category]
    spec = st.selectbox(
        "Metric",
        category_specs,
        index=_default_spec(category_specs, benchmark_type),
        format_func=lambda item: item.label,
        key="explorer_metric",
    )
    dimensions = {
        "Input length": "input_length",
        "Output length": "output_length",
        "Concurrency": "maximum_concurrency",
        "Batch size": "batch_size",
        "Thread count": "thread_count",
        "Memory system": "memory_type",
        "Platform": "platform",
        "Model": "model_scale",
        "Timestamp": "timestamp",
        "Run": "run_id",
    }
    encodings = {
        "Platform": "platform",
        "Model": "model_scale",
        "Memory system": "memory_type",
        "Evidence source": "source",
    }
    controls = st.columns(2)
    dimension_label = controls[0].selectbox("X axis", list(dimensions), key="explorer_x")
    color_label = controls[1].selectbox(
        "Color", list(encodings), key="explorer_color"
    )
    dimension = dimensions[dimension_label]
    color = encodings[color_label]
    rows = []
    for entry in selected_entries:
        value = _finite(entry.summary.get(spec.field))
        catalog = catalog_row(entry)
        x_value = catalog.get(dimension)
        if value is not None and x_value is not None:
            rows.append(
                {
                    dimension_label: x_value,
                    "Value": value,
                    "Color": catalog.get(color) or "Unspecified",
                    "Run": entry.summary.get("run_id"),
                    "Model": _short_model(entry.summary),
                    "Platform": catalog["platform"],
                    "Memory": catalog["memory_type"],
                    "Workload": catalog["workload"],
                    "Source": catalog["source"],
                }
            )
    if not rows:
        st.info("The selected metric and axis have no overlapping observations.")
        return
    frame = pd.DataFrame(rows)
    numeric_axis = dimension in {
        "input_length",
        "output_length",
        "maximum_concurrency",
        "batch_size",
        "thread_count",
    }
    if dimension == "timestamp":
        frame[dimension_label] = pd.to_datetime(frame[dimension_label], errors="coerce")
    color_map = PLATFORM_COLORS if color == "platform" else None
    if numeric_axis or dimension == "timestamp":
        figure = px.scatter(
            frame,
            x=dimension_label,
            y="Value",
            color="Color",
            symbol="Model",
            hover_data=["Run", "Platform", "Memory", "Workload", "Source"],
            color_discrete_map=color_map,
            title=f"{spec.label} — {benchmark_type}",
            labels={"Value": f"{spec.label} ({spec.unit})", "Color": color_label},
        )
        figure.update_traces(marker={"size": 12, "line": {"width": 1, "color": "white"}})
    else:
        figure = px.strip(
            frame,
            x=dimension_label,
            y="Value",
            color="Color",
            hover_data=["Run", "Model", "Platform", "Memory", "Workload", "Source"],
            color_discrete_map=color_map,
            title=f"{spec.label} — {benchmark_type}",
            labels={"Value": f"{spec.label} ({spec.unit})", "Color": color_label},
        )
        figure.update_traces(marker={"size": 13, "line": {"width": 1, "color": "white"}})
    st.plotly_chart(
        _apply_chart_style(figure, height=470),
        width="stretch",
        theme=None,
    )
    st.caption(
        "Every point is one run. This view is descriptive; use Compare or the memory-study "
        "matched pairs before making directional claims."
    )
    with st.expander("View plotted observations"):
        st.dataframe(frame, width="stretch", hide_index=True)


def _tradeoff_explorer(
    st: Any, pd: Any, px: Any, entries: Sequence[CatalogEntry], benchmark_type: str
) -> None:
    selected_entries = [
        entry for entry in entries if entry.summary.get("benchmark_type") == benchmark_type
    ]
    specs = _available_specs(selected_entries)
    if len(specs) < 2:
        st.info("At least two overlapping metrics are required for a trade-off plot.")
        return
    x_default = next(
        (index for index, spec in enumerate(specs) if spec.field == "mean_e2e_latency_ms"),
        0,
    )
    y_default = next(
        (
            index
            for index, spec in enumerate(specs)
            if spec.field == "output_throughput_tokens_per_second"
        ),
        min(1, len(specs) - 1),
    )
    controls = st.columns(2)
    x_spec = controls[0].selectbox(
        "Horizontal metric",
        specs,
        index=x_default,
        format_func=lambda spec: spec.label,
        key="tradeoff_x",
    )
    y_spec = controls[1].selectbox(
        "Vertical metric",
        specs,
        index=y_default,
        format_func=lambda spec: spec.label,
        key="tradeoff_y",
    )
    if x_spec == y_spec:
        st.info("Choose two different metrics.")
        return
    rows = []
    for entry in selected_entries:
        x_value = _finite(entry.summary.get(x_spec.field))
        y_value = _finite(entry.summary.get(y_spec.field))
        if x_value is None or y_value is None:
            continue
        row = catalog_row(entry)
        rows.append(
            {
                "X": x_value,
                "Y": y_value,
                "Platform": row["platform"],
                "Model": row["model_scale"] or row["model_id"],
                "Memory": row["memory_type"],
                "Run": row["run_id"],
                "Workload": row["workload"],
                "Source": row["source"],
            }
        )
    if not rows:
        st.info("The selected metrics have no overlapping observations.")
        return
    frame = pd.DataFrame(rows)
    figure = px.scatter(
        frame,
        x="X",
        y="Y",
        color="Platform",
        symbol="Model",
        custom_data=["Run", "Memory", "Workload", "Source"],
        color_discrete_map=PLATFORM_COLORS,
        title=f"{y_spec.label} vs. {x_spec.label}",
        labels={
            "X": f"{x_spec.label} ({x_spec.unit})",
            "Y": f"{y_spec.label} ({y_spec.unit})",
        },
    )
    figure.update_traces(marker={"size": 14, "line": {"width": 1, "color": "white"}})
    st.plotly_chart(
        _apply_chart_style(figure, height=490),
        width="stretch",
        theme=None,
    )
    st.caption(
        f"{x_spec.label}: {x_spec.direction}; {y_spec.label}: {y_spec.direction}. "
        "Direction metadata is shown instead of collapsing unlike units into an overall score."
    )


def _render_explorer(
    st: Any, pd: Any, px: Any, entries: Sequence[CatalogEntry]
) -> None:
    _page_header(
        st,
        "Analysis canvas",
        "Explorer",
        "Slice one metric across a workload dimension or inspect a two-metric trade-off. "
        "Raw observations remain visible and are never automatically ranked.",
    )
    _simulated_notice(st, entries)
    if not entries:
        st.info("No runs match the active filters.")
        return
    types = sorted({str(entry.summary.get("benchmark_type")) for entry in entries})
    benchmark_type = st.selectbox("Benchmark type", types, key="explorer_benchmark")
    single_tab, tradeoff_tab = st.tabs(
        ("Single metric", "Trade-off"), key="explorer_tabs"
    )
    with single_tab:
        _single_metric_explorer(st, pd, px, entries, benchmark_type)
    with tradeoff_tab:
        _tradeoff_explorer(st, pd, px, entries, benchmark_type)
    if benchmark_type == "smoke":
        st.warning("Smoke measurements are validation evidence, not a formal performance campaign.")


def _memory_pair_key(entry: CatalogEntry) -> tuple[Any, ...]:
    summary = entry.summary
    return (
        summary.get("model_id"),
        summary.get("benchmark_type"),
        summary.get("backend"),
        summary.get("backend_version"),
        summary.get("cpu_model"),
        summary.get("model_precision"),
        summary.get("input_length"),
        summary.get("output_length"),
        summary.get("number_of_requests"),
        summary.get("request_rate"),
        summary.get("maximum_concurrency"),
        summary.get("batch_size"),
        summary.get("thread_count"),
        summary.get("process_count"),
        summary.get("thread_affinity"),
        summary.get("numa_policy"),
        summary.get("memory_bandwidth_instrument"),
        summary.get("memory_bandwidth_scope"),
        summary.get("power_instrument"),
        summary.get("power_scope"),
    )


def _matched_memory_pairs(
    entries: Sequence[CatalogEntry],
) -> list[tuple[CatalogEntry, CatalogEntry]]:
    grouped: dict[tuple[Any, ...], dict[str, CatalogEntry]] = defaultdict(dict)
    for entry in entries:
        summary = entry.summary
        if summary.get("hardware_type") != "cpu" or summary.get("status") != "completed":
            continue
        memory = memory_system(summary).lower()
        if "ddr" in memory:
            grouped[_memory_pair_key(entry)]["ddr"] = entry
        elif "hbm" in memory:
            grouped[_memory_pair_key(entry)]["hbm"] = entry
    return [
        (pair["ddr"], pair["hbm"])
        for pair in grouped.values()
        if "ddr" in pair and "hbm" in pair
    ]


def _render_memory_study(
    st: Any, pd: Any, px: Any, entries: Sequence[CatalogEntry]
) -> None:
    _page_header(
        st,
        "Future study",
        "CPU memory study",
        "A treatment-aware workspace for matched CPU DDR and HBM runs: bandwidth, "
        "phase behavior, placement evidence, and direction-aware paired effects.",
    )
    cpu_entries = [entry for entry in entries if entry.summary.get("hardware_type") == "cpu"]
    _simulated_notice(st, cpu_entries)
    if not cpu_entries:
        st.info(
            "No CPU results are visible yet. The dashboard is ready for CPU summaries that "
            "record memory type, placement controls, bandwidth, CPU resources, and shared "
            "latency/throughput metrics."
        )

    pairs = _matched_memory_pairs(cpu_entries)
    memory_types = {
        memory_system(entry.summary)
        for entry in cpu_entries
        if memory_system(entry.summary) != "Unspecified"
    }
    placed = sum(
        all(
            entry.summary.get(field) is not None
            for field in ("thread_affinity", "numa_policy", "memory_binding")
        )
        for entry in cpu_entries
    )
    bandwidth_values = [
        value
        for entry in cpu_entries
        if (value := _finite(entry.summary.get("measured_memory_bandwidth_gbps"))) is not None
    ]
    columns = st.columns(4)
    columns[0].metric("Visible CPU runs", len(cpu_entries))
    columns[1].metric("Matched DDR↔HBM pairs", len(pairs))
    columns[2].metric("Memory systems", len(memory_types))
    columns[3].metric(
        "Placement evidence",
        f"{placed / len(cpu_entries):.0%}" if cpu_entries else "—",
        help="Thread affinity, NUMA policy, and memory binding all recorded.",
    )

    frame = _analysis_frame(pd, cpu_entries)
    left, right = st.columns((1.15, 1))
    with left:
        bandwidth = frame.dropna(
            subset=[
                "measured_memory_bandwidth_gbps",
                "output_throughput_tokens_per_second",
            ]
        )
        if bandwidth.empty:
            figure = px.scatter(
                pd.DataFrame({"Bandwidth": [], "Throughput": []}),
                x="Bandwidth",
                y="Throughput",
                title="Achieved bandwidth vs. output throughput",
            )
            figure.add_annotation(
                text="No overlapping bandwidth and throughput observations",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 14, "color": "#667085"},
            )
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
        else:
            figure = px.scatter(
                bandwidth,
                x="measured_memory_bandwidth_gbps",
                y="output_throughput_tokens_per_second",
                color="memory_type",
                symbol="model_scale",
                size="peak_memory_gib",
                size_max=28,
                custom_data=["run_id", "workload", "numa_policy", "memory_binding", "source"],
                color_discrete_map={"HBM2e": "#0E9F9A", "DDR5": "#E58B32"},
                title="Achieved bandwidth vs. output throughput",
                labels={
                    "measured_memory_bandwidth_gbps": "Measured memory bandwidth (GB/s)",
                    "output_throughput_tokens_per_second": "Output throughput (tokens/s)",
                    "memory_type": "Memory",
                    "model_scale": "Model",
                },
            )
            figure.update_traces(marker={"line": {"width": 1, "color": "white"}})
        st.plotly_chart(
            _apply_chart_style(figure, height=440),
            width="stretch",
            theme=None,
            key="memory_primary_left",
        )
    with right:
        st.metric(
            "Highest observed bandwidth",
            (
                _format_number(max(bandwidth_values), "GB/s")
                if bandwidth_values
                else "—"
            ),
        )
        phase_rows = []
        for entry in cpu_entries:
            for field in (
                "prefill_throughput_tokens_per_second",
                "decode_throughput_tokens_per_second",
            ):
                value = _finite(entry.summary.get(field))
                if value is None:
                    continue
                phase_rows.append(
                    {
                        "Model": _short_model(entry.summary),
                        "Memory": memory_system(entry.summary),
                        "Phase": "Prefill" if field.startswith("prefill") else "Decode",
                        "Throughput": value,
                    }
                )
        if phase_rows:
            phase_frame = pd.DataFrame(phase_rows)
            figure = px.bar(
                phase_frame,
                x="Phase",
                y="Throughput",
                color="Memory",
                facet_col="Model",
                barmode="group",
                color_discrete_map={"HBM2e": "#0E9F9A", "DDR5": "#E58B32"},
                title="Phase-specific throughput",
                labels={"Throughput": "Throughput (tokens/s)"},
            )
            figure.for_each_annotation(lambda item: item.update(text=item.text.split("=")[-1]))
            figure.update_xaxes(title_text="")
        else:
            figure = px.bar(
                pd.DataFrame({"Phase": [], "Throughput": []}),
                x="Phase",
                y="Throughput",
                title="Phase-specific throughput",
            )
            figure.add_annotation(
                text="Prefill/decode phase metrics are not recorded",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 14, "color": "#667085"},
            )
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
        st.plotly_chart(
            _apply_chart_style(figure, height=380),
            width="stretch",
            theme=None,
            key="memory_primary_right",
        )

    st.markdown('<div class="bench-section">Matched-pair effect</div>', unsafe_allow_html=True)
    pair_specs = [
        spec
        for spec in METRIC_SPECS
        if spec.direction != "descriptive"
        and any(
            _finite(ddr.summary.get(spec.field)) is not None
            and _finite(hbm.summary.get(spec.field)) is not None
            for ddr, hbm in pairs
        )
    ]
    if not pair_specs:
        fallback = _spec("output_throughput_tokens_per_second")
        pair_specs = [fallback] if fallback is not None else []
    selected = st.selectbox(
        "Paired metric",
        pair_specs,
        index=_default_spec(pair_specs, "offline"),
        format_func=lambda spec: spec.label,
        key="memory_pair_metric",
        disabled=not pairs,
    )
    pair_rows = []
    for ddr, hbm in pairs:
        ddr_value = _finite(ddr.summary.get(selected.field))
        hbm_value = _finite(hbm.summary.get(selected.field))
        if ddr_value in (None, 0.0) or hbm_value is None:
            continue
        raw_change = (hbm_value / ddr_value - 1) * 100
        improvement = raw_change if selected.direction == "higher" else -raw_change
        pair_rows.append(
            {
                "Pair": (
                    f"{_short_model(ddr.summary)} · "
                    f"{ddr.summary.get('benchmark_type')} · "
                    f"in={ddr.summary.get('input_length')}"
                ),
                "HBM improvement": improvement,
                "DDR": ddr_value,
                "HBM": hbm_value,
                "Source": catalog_row(hbm)["source"],
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    if pair_frame.empty:
        st.info(
            "No DDR/HBM pairs share the same model, backend, CPU, workload, precision, "
            "threading, and NUMA controls."
        )
        figure = px.bar(
            pd.DataFrame({"Pair": [], "HBM improvement": []}),
            x="HBM improvement",
            y="Pair",
            orientation="h",
            title=f"Direction-aware HBM effect — {selected.label}",
        )
        figure.add_annotation(
            text="No controlled DDR/HBM pairs are available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": "#667085"},
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    else:
        figure = px.bar(
            pair_frame,
            x="HBM improvement",
            y="Pair",
            orientation="h",
            color="HBM improvement",
            color_continuous_scale=["#F04438", "#F2F4F7", "#12B76A"],
            color_continuous_midpoint=0,
            custom_data=["DDR", "HBM", "Source"],
            title=f"Direction-aware HBM effect — {selected.label}",
            labels={"HBM improvement": "HBM improvement vs. DDR (%)"},
        )
        figure.add_vline(x=0, line_color="#667085", line_width=1)
        figure.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>Improvement: %{x:.1f}%<br>"
                f"DDR: %{{customdata[0]:.4g}} {selected.unit}<br>"
                f"HBM: %{{customdata[1]:.4g}} {selected.unit}<br>"
                "Source: %{customdata[2]}<extra></extra>"
            )
        )
    st.plotly_chart(
        _apply_chart_style(figure, height=380),
        width="stretch",
        theme=None,
        key="memory_secondary",
    )
    st.caption(
        "Positive means HBM moved the metric in its preferred direction. "
        "Pairs match on the recorded model, CPU, backend, workload, precision, "
        "threading, and NUMA controls; memory is the intended treatment."
    )

    st.markdown('<div class="bench-section">Placement control matrix</div>', unsafe_allow_html=True)
    controls = []
    for entry in cpu_entries:
        summary = entry.summary
        controls.append(
            {
                "Run": summary.get("run_id"),
                "Model": _short_model(summary),
                "Benchmark": summary.get("benchmark_type"),
                "Memory": memory_system(summary),
                "Threads": summary.get("thread_count"),
                "Processes": summary.get("process_count"),
                "Affinity": summary.get("thread_affinity"),
                "NUMA policy": summary.get("numa_policy"),
                "Memory binding": summary.get("memory_binding"),
                "Memory mode": summary.get("memory_mode"),
                "Source": catalog_row(entry)["source"],
            }
        )
    st.dataframe(
        pd.DataFrame(
            controls,
            columns=(
                "Run",
                "Model",
                "Benchmark",
                "Memory",
                "Threads",
                "Processes",
                "Affinity",
                "NUMA policy",
                "Memory binding",
                "Memory mode",
                "Source",
            ),
        ),
        width="stretch",
        hide_index=True,
        key="memory_control_matrix",
    )


def _get_dotted(data: Mapping[str, Any], field: str) -> Any:
    value: Any = data
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _memory_treatment_compatibility(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    if left.get("hardware_type") != "cpu" or right.get("hardware_type") != "cpu":
        return {
            "status": "incompatible",
            "evidence": [
                {
                    "field": "hardware_type",
                    "left": left.get("hardware_type"),
                    "right": right.get("hardware_type"),
                    "reason": "The memory-treatment lens requires two CPU runs.",
                }
            ],
        }
    evidence = []
    control_fields = [
        "model_id",
        "model_revision",
        "resolved_model_revision",
        *COMPATIBILITY_FIELDS,
        *CPU_COMPATIBILITY_FIELDS,
        "batch_size",
        "memory_bandwidth_instrument",
        "memory_bandwidth_scope",
        "power_instrument",
        "power_scope",
    ]
    ignored = _MEMORY_TREATMENT_FIELDS | {
        "accelerator_name",
        "accelerator_count",
        "accelerator_visibility",
        "gpu_memory_utilization",
        "software_versions.cuda_runtime",
        "software_versions.nvidia_driver",
    }
    for field in dict.fromkeys(control_fields):
        if field in ignored:
            continue
        left_value = _get_dotted(left, field)
        right_value = _get_dotted(right, field)
        if left_value != right_value:
            evidence.append(
                {
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                    "reason": "Recorded control differs.",
                }
            )
    left_memory = memory_system(left)
    right_memory = memory_system(right)
    if left_memory == "Unspecified" or right_memory == "Unspecified":
        evidence.append(
            {
                "field": "memory_type",
                "left": left_memory,
                "right": right_memory,
                "reason": "Both memory treatments must be explicit.",
            }
        )
    elif left_memory == right_memory:
        evidence.append(
            {
                "field": "memory_type",
                "left": left_memory,
                "right": right_memory,
                "reason": "Choose distinct DDR and HBM treatments.",
            }
        )
    failed = [
        side
        for side, summary in (("left", left), ("right", right))
        if summary.get("status") != "completed"
    ]
    if failed:
        evidence.append(
            {
                "field": "status",
                "left": left.get("status"),
                "right": right.get("status"),
                "reason": "Both runs must be completed.",
            }
        )
    return {"status": "compatible" if not evidence else "incompatible", "evidence": evidence}


def _comparison_values(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    ratios_allowed: bool,
) -> list[dict[str, Any]]:
    rows = []
    for spec in METRIC_SPECS:
        left_value = _finite(left.get(spec.field))
        right_value = _finite(right.get(spec.field))
        if left_value is None and right_value is None:
            continue
        ratio = (
            right_value / left_value
            if ratios_allowed
            and left_value not in (None, 0.0)
            and right_value is not None
            else None
        )
        preferred_change = None
        if ratio is not None and spec.direction != "descriptive":
            raw_change = (ratio - 1) * 100
            preferred_change = raw_change if spec.direction == "higher" else -raw_change
        rows.append(
            {
                "Category": spec.category,
                "Metric": spec.label,
                "Unit": spec.unit,
                "Left": left_value,
                "Right": right_value,
                "Right / left": ratio,
                "Preferred change (%)": preferred_change,
                "Aggregation": spec.aggregation,
                "Direction": spec.direction,
            }
        )
    return rows


def _comparison_plot_rows(
    left: CatalogEntry,
    right: CatalogEntry,
    left_measurements: Mapping[str, Any],
    right_measurements: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for spec in METRIC_SPECS:
        for side, entry, measurements in (
            ("Left", left, left_measurements),
            ("Right", right, right_measurements),
        ):
            value = _finite(entry.summary.get(spec.field))
            if value is None:
                continue
            variability = (
                repetition_variability(measurements, spec.field)
                if spec.aggregation == "mean"
                else None
            )
            rows.append(
                {
                    "category": spec.category,
                    "unit": spec.unit,
                    "metric": spec.label,
                    "side": side,
                    "value": value,
                    "error": variability["standard_deviation"] if variability else 0.0,
                }
            )
    return rows


def _render_compare(
    st: Any,
    pd: Any,
    px: Any,
    entries: Sequence[CatalogEntry],
    dataset: DashboardDataset,
) -> None:
    _page_header(
        st,
        "Controlled contrast",
        "Compare explicit runs",
        "Compare two named runs with an explicit analytical lens. Ratios appear only when "
        "the applicable controls are established; absolute observations always remain visible.",
    )
    if len(entries) < 2:
        st.info("At least two visible runs are required. Adjust the filters if necessary.")
        return
    columns = st.columns(2)
    left = columns[0].selectbox(
        "Left run", entries, format_func=_entry_label, key="left_run"
    )
    right = columns[1].selectbox(
        "Right run", entries, index=1, format_func=_entry_label, key="right_run"
    )
    if left.key == right.key:
        st.warning("Choose two distinct result records.")
        return
    _simulated_notice(st, [left, right])

    cards = st.columns(2)
    for column, label, entry in (
        (cards[0], "LEFT", left),
        (cards[1], "RIGHT", right),
    ):
        summary = entry.summary
        column.markdown(
            (
                '<div class="bench-run-card">'
                f'<div class="bench-kicker">{label}</div>'
                f"<strong>{_short_model(summary)} · {summary.get('benchmark_type')}</strong><br>"
                f"{hardware_platform(summary)} · {hardware_name(summary)} · "
                f"{memory_system(summary)}<br>{summary.get('run_id')}</div>"
            ),
            unsafe_allow_html=True,
        )

    lens = st.selectbox(
        "Comparison lens",
        (
            "Like-for-like (formal)",
            "Memory-system treatment (CPU)",
            "Descriptive only",
        ),
        help=(
            "The treatment lens allows DDR/HBM to differ intentionally while requiring the "
            "model, CPU, backend, workload, precision, threading, and NUMA controls to match."
        ),
        key="comparison_lens",
    )
    comparison = compare_summaries(left.summary, right.summary)
    evidence: list[dict[str, Any]]
    if lens == "Like-for-like (formal)":
        compatibility = comparison["compatibility"]
        status = compatibility["status"]
        evidence = [
            {"kind": "mismatch", **item}
            for item in compatibility["mismatched_fields"]
        ] + [
            {"kind": "missing", **item}
            for item in compatibility["missing_fields"]
        ]
        ratios_allowed = status == "compatible"
    elif lens == "Memory-system treatment (CPU)":
        treatment = _memory_treatment_compatibility(left.summary, right.summary)
        status = treatment["status"]
        evidence = [{"kind": "control", **item} for item in treatment["evidence"]]
        ratios_allowed = status == "compatible"
    else:
        status = "descriptive"
        evidence = []
        ratios_allowed = False

    renderer = {
        "compatible": st.success,
        "partial": st.warning,
        "descriptive": st.info,
    }.get(status, st.error)
    messages = {
        "compatible": "Compatibility established for the selected lens; ratios are enabled.",
        "partial": "Compatibility is partial; ratios remain suppressed.",
        "incompatible": "Controls differ for the selected lens; ratios remain suppressed.",
        "descriptive": "Descriptive lens selected; only absolute observations are shown.",
    }
    renderer(messages.get(status, f"Compatibility: {status}"))
    if evidence:
        with st.expander(f"Compatibility evidence ({len(evidence)})", expanded=False):
            evidence_rows = [
                {
                    **item,
                    "left": repr(item.get("left")),
                    "right": repr(item.get("right")),
                }
                for item in evidence
            ]
            st.dataframe(pd.DataFrame(evidence_rows), width="stretch", hide_index=True)

    left_measurements, left_legacy = measurements_for_entry(dataset, left)
    right_measurements, right_legacy = measurements_for_entry(dataset, right)
    if left_legacy or right_legacy:
        st.warning(
            "One or both runs lack measurements.json; repetition values were derived in memory "
            "from preserved legacy files."
        )

    st.markdown('<div class="bench-section">Interpretation</div>', unsafe_allow_html=True)
    if lens == "Like-for-like (formal)":
        for statement in comparison_narrative(
            comparison,
            left.summary,
            right.summary,
            left_measurements=left_measurements,
            right_measurements=right_measurements,
        ):
            st.write(f"- {statement}")
    elif lens == "Memory-system treatment (CPU)":
        if ratios_allowed:
            st.write(
                "- The recorded CPU, model, backend, workload, precision, threading, and "
                "NUMA controls match; memory system is the declared treatment."
            )
        else:
            st.write(
                "- The recorded controls do not establish a clean memory-system treatment pair."
            )
        st.write(
            "- Effects remain descriptive observations; no statistical significance or "
            "model-answer quality is inferred."
        )
    else:
        st.write(
            "- Absolute values describe each run in its own recorded conditions. No fair-pair "
            "claim, directional ratio, statistical significance, or quality inference is made."
        )

    metric_rows = _comparison_values(
        left.summary, right.summary, ratios_allowed=ratios_allowed
    )
    metric_frame = pd.DataFrame(metric_rows)
    st.markdown('<div class="bench-section">Metric ledger</div>', unsafe_allow_html=True)
    if metric_frame.empty:
        st.info("Neither run contains recognized numeric metrics.")
    else:
        st.dataframe(metric_frame, width="stretch", hide_index=True)

    plot_rows = _comparison_plot_rows(
        left, right, left_measurements, right_measurements
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plot_rows:
        grouped[(row["category"], row["unit"])].append(row)
    if grouped:
        st.markdown(
            '<div class="bench-section">Absolute metric panels</div>',
            unsafe_allow_html=True,
        )
        for (category, unit), rows in grouped.items():
            frame = pd.DataFrame(rows)
            figure = px.bar(
                frame,
                x="metric",
                y="value",
                color="side",
                error_y="error",
                barmode="group",
                title=f"{category} ({unit})",
                labels={"value": f"Value ({unit})", "metric": "", "side": "Run"},
                color_discrete_map={"Left": "#7257E8", "Right": "#0E9F9A"},
            )
            st.plotly_chart(
                _apply_chart_style(figure, height=360),
                width="stretch",
                theme=None,
            )
        st.caption(
            "Error bars show one descriptive sample standard deviation only for mean-aggregated "
            "metrics with at least two observations."
        )

    if ratios_allowed and not metric_frame.empty:
        changes = metric_frame.dropna(subset=["Preferred change (%)"])
        if not changes.empty:
            figure = px.bar(
                changes,
                x="Preferred change (%)",
                y="Metric",
                color="Category",
                orientation="h",
                title="Direction-aware change from left to right",
                labels={"Preferred change (%)": "Preferred-direction change (%)"},
            )
            figure.add_vline(x=0, line_color="#667085", line_width=1)
            st.plotly_chart(
                _apply_chart_style(figure, height=max(360, 30 * len(changes))),
                width="stretch",
                theme=None,
            )
            st.caption(
                "Positive means the right run moved in the metric's preferred direction. "
                "Descriptive metrics are omitted and unlike units are never combined."
            )

    if not metric_frame.empty:
        export = {
            "comparison_lens": lens,
            "ratios_allowed": ratios_allowed,
            "left_run_id": left.summary.get("run_id"),
            "right_run_id": right.summary.get("run_id"),
            "compatibility_status": status,
            "compatibility_evidence": evidence,
            "metrics": metric_rows,
        }
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download visible comparison JSON",
            (json.dumps(export, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
            file_name="dashboard_comparison.json",
            mime="application/json",
            key="comparison_json",
        )
        download_columns[1].download_button(
            "Download visible comparison CSV",
            metric_frame.to_csv(index=False).encode("utf-8"),
            file_name="dashboard_comparison.csv",
            mime="text/csv",
            key="comparison_csv",
        )


def run_dashboard(results_root: str | Path) -> None:
    """Render the dashboard; optional dependencies are imported only when launched."""

    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.set_page_config(
        page_title="LLM benchmark results",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles(st)
    st.markdown(
        '<div class="bench-kicker">Inference systems observatory</div>',
        unsafe_allow_html=True,
    )
    st.title("LLM inference benchmark results")
    resolved_root = Path(results_root).expanduser().resolve()
    st.caption(f"Read-only results root · {resolved_root}")

    @st.cache_data(show_spinner=False)
    def cached_catalog(root: str) -> ResultCatalog:
        return discover_results(root)

    measured = dataset_from_catalog(cached_catalog(str(resolved_root)))
    demo = simulated_dataset()
    source_options = ("Measured results", "Demo study", "Measured + demo")
    default_source = "Measured results" if measured.catalog.entries else "Demo study"
    source = st.sidebar.selectbox(
        "Data source",
        source_options,
        index=source_options.index(default_source),
        help="Demo runs stay in memory and never modify the results root.",
        key="data_source",
    )
    if source == "Demo study":
        dataset = demo
    elif source == "Measured + demo":
        dataset = combine_datasets(measured, demo)
    else:
        dataset = measured

    if st.sidebar.button("Refresh measured catalog", width="stretch", key="refresh_catalog"):
        st.cache_data.clear()
        st.rerun()
    page = st.sidebar.radio(
        "View",
        ("Overview", "Run detail", "Explorer", "CPU memory study", "Compare"),
        key="page",
    )
    filter_namespace = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    entries = _filter_entries(st, dataset, namespace=filter_namespace)
    st.sidebar.caption(f"{len(entries)} of {len(dataset.catalog.entries)} runs visible")

    page_keys = {
        "Overview": "overview",
        "Run detail": "run_detail",
        "Explorer": "explorer",
        "CPU memory study": "cpu_memory_study",
        "Compare": "compare",
    }
    hidden_rules = "\n".join(
        f".st-key-dashboard_page_{key} {{ display: none; }}"
        for label, key in page_keys.items()
        if label != page
    )
    st.markdown(f"<style>{hidden_rules}</style>", unsafe_allow_html=True)

    with st.container(key="dashboard_page_overview"):
        _render_overview(st, pd, px, entries, dataset)
    with st.container(key="dashboard_page_run_detail"):
        _render_run_detail(st, pd, px, entries, dataset)
    with st.container(key="dashboard_page_explorer"):
        _render_explorer(st, pd, px, entries)
    with st.container(key="dashboard_page_cpu_memory_study"):
        _render_memory_study(st, pd, px, entries)
    with st.container(key="dashboard_page_compare"):
        _render_compare(st, pd, px, entries, dataset)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only benchmark results dashboard")
    parser.add_argument("--results-root", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    run_dashboard(args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
