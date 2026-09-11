"""Streamlit entry point for read-only benchmark result exploration."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
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
from llm_bench.interpretation import repetition_variability
from llm_bench.measurements import read_telemetry_series, resolve_recorded_path
from llm_bench.metrics import METRIC_SPECS, MetricSpec
from llm_bench.results import (
    COMPATIBILITY_FIELDS,
    CPU_COMPATIBILITY_FIELDS,
    ResultError,
    check_backend_treatment_compatibility,
    compare_summaries,
)

_THEME_TOKENS = {
    "light": {
        "background": "#F4F6FA",
        "surface": "#FFFFFF",
        "surface_raised": "#FFFFFF",
        "surface_soft": "#EEF1F7",
        "surface_accent": "#EEEDFF",
        "text": "#182033",
        "muted": "#667085",
        "subtle": "#8A94A6",
        "border": "#DDE3EC",
        "border_strong": "#C8D0DD",
        "primary": "#5B5BD6",
        "primary_strong": "#4545B8",
        "primary_text": "#39399D",
        "grid": "#E5EAF1",
        "positive": "#087F5B",
        "negative": "#C92A3A",
        "warning": "#B45F06",
        "info": "#2563A8",
        "shadow": "rgba(24, 32, 51, .08)",
        "gpu": "#6956E8",
        "hbm": "#008C7A",
        "ddr": "#D97706",
        "neutral": "#64748B",
    },
    "dark": {
        "background": "#0A0F1D",
        "surface": "#111827",
        "surface_raised": "#151E2F",
        "surface_soft": "#1B2538",
        "surface_accent": "#25244A",
        "text": "#F3F6FC",
        "muted": "#A8B2C7",
        "subtle": "#7F8BA3",
        "border": "#2A3549",
        "border_strong": "#3A4760",
        "primary": "#9A8CFF",
        "primary_strong": "#B8ADFF",
        "primary_text": "#C8C1FF",
        "grid": "#29354A",
        "positive": "#3ED6A1",
        "negative": "#FF7180",
        "warning": "#F6C15C",
        "info": "#64B5FF",
        "shadow": "rgba(0, 0, 0, .28)",
        "gpu": "#A78BFA",
        "hbm": "#2DD4BF",
        "ddr": "#FBBF24",
        "neutral": "#94A3B8",
    },
}

_ACTIVE_THEME: ContextVar[str] = ContextVar("dashboard_theme", default="light")

_VIEW_META = {
    "Overview": ("dashboard", "Campaign snapshot"),
    "Run detail": ("search_insights", "Inspect one run"),
    "Explorer": ("query_stats", "Explore metrics"),
    "Memory study": ("memory", "GPU, CPU, DDR, HBM"),
    "Compare": ("compare_arrows", "Compare selected runs"),
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
    "memory_mode_requested",
    "memory_mode_detected",
    "memory_mode_detection_method",
    "memory_mode_verified",
    "memory_binding",
    "memory_binding_resolved",
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


def _theme_tokens(theme: str | None = None) -> Mapping[str, str]:
    selected = (theme or _ACTIVE_THEME.get()).strip().lower()
    return _THEME_TOKENS.get(selected, _THEME_TOKENS["light"])


def _platform_colors() -> dict[str, str]:
    colors = _theme_tokens()
    return {
        "GPU": colors["gpu"],
        "CPU · HBM": colors["hbm"],
        "CPU · DDR": colors["ddr"],
        "CPU": colors["neutral"],
        "UNKNOWN": colors["subtle"],
    }


def _categorical_colors() -> list[str]:
    colors = _theme_tokens()
    return [
        colors["gpu"],
        colors["hbm"],
        colors["ddr"],
        colors["info"],
        colors["positive"],
        colors["primary_strong"],
        colors["neutral"],
    ]


def _apply_chart_style(figure: Any, *, height: int = 390) -> Any:
    colors = _theme_tokens()
    is_dark = _ACTIVE_THEME.get() == "dark"
    figure.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        paper_bgcolor=colors["surface"],
        plot_bgcolor=colors["surface"],
        height=height,
        margin={"l": 20, "r": 20, "t": 62, "b": 92},
        colorway=_categorical_colors(),
        font={
            "family": "Inter, ui-sans-serif, system-ui",
            "color": colors["muted"],
        },
        title_font={"size": 17, "color": colors["text"]},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "left",
            "x": 0,
            "title": None,
            "font": {"size": 11},
        },
        hoverlabel={
            "font_size": 13,
            "bgcolor": colors["surface_raised"],
            "bordercolor": colors["border_strong"],
            "font_color": colors["text"],
        },
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=colors["border"],
        tickfont={"color": colors["muted"]},
        title_font={"color": colors["muted"]},
    )
    figure.update_yaxes(
        gridcolor=colors["grid"],
        zerolinecolor=colors["border_strong"],
        tickfont={"color": colors["muted"]},
        title_font={"color": colors["muted"]},
    )
    return figure


def _inject_styles(st: Any, theme: str) -> None:
    colors = _theme_tokens(theme)
    color_scheme = "dark" if theme == "dark" else "light"
    st.markdown(
        f"""
        <style>
        :root {{
            --bench-bg: {colors["background"]};
            --bench-surface: {colors["surface"]};
            --bench-raised: {colors["surface_raised"]};
            --bench-soft: {colors["surface_soft"]};
            --bench-accent-soft: {colors["surface_accent"]};
            --bench-ink: {colors["text"]};
            --bench-muted: {colors["muted"]};
            --bench-subtle: {colors["subtle"]};
            --bench-border: {colors["border"]};
            --bench-border-strong: {colors["border_strong"]};
            --bench-primary: {colors["primary"]};
            --bench-primary-strong: {colors["primary_strong"]};
            --bench-primary-text: {colors["primary_text"]};
            --bench-positive: {colors["positive"]};
            --bench-negative: {colors["negative"]};
            --bench-warning: {colors["warning"]};
            --bench-info: {colors["info"]};
            --bench-shadow: {colors["shadow"]};
            color-scheme: {color_scheme};
        }}

        html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{
            color-scheme: {color_scheme};
            background: var(--bench-bg);
            color: var(--bench-ink);
        }}
        [data-testid="stAppViewContainer"] {{
            --text-color: var(--bench-ink);
            --background-color: var(--bench-bg);
            --secondary-background-color: var(--bench-soft);
            --primary-color: var(--bench-primary);
            background:
                radial-gradient(
                    circle at 82% -8%,
                    color-mix(in srgb, var(--bench-primary) 12%, transparent),
                    transparent 29rem
                ),
                radial-gradient(
                    circle at 50% 120%,
                    color-mix(in srgb, var(--bench-info) 7%, transparent),
                    transparent 36rem
                ),
                var(--bench-bg);
        }}
        [data-testid="stHeader"] {{
            height: 0 !important;
            min-height: 0 !important;
            border: 0 !important;
            background: transparent !important;
        }}
        [data-testid="stToolbar"] {{ display: none !important; }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 1540px;
            padding-top: 1.35rem;
            padding-bottom: 4rem;
        }}
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] label {{ color: var(--bench-ink); }}
        [data-testid="stAppViewContainer"] h1 {{
            font-size: clamp(2rem, 3vw, 2.7rem);
            line-height: 1.08;
            letter-spacing: -.045em;
            margin-bottom: .4rem;
        }}
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {{
            color: var(--bench-muted);
        }}

        [data-testid="stSidebar"] {{
            min-width: 336px !important;
            width: 336px !important;
            border-right: 1px solid var(--bench-border);
            background: color-mix(in srgb, var(--bench-surface) 94%, transparent);
            box-shadow: 10px 0 34px color-mix(in srgb, var(--bench-shadow) 45%, transparent);
        }}
        [data-testid="stSidebar"] > div:first-child {{ width: 336px !important; }}
        [data-testid="stSidebarContent"] {{ padding-top: .85rem; }}
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [role="radiogroup"] p,
        [data-testid="stSidebar"] details > summary p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
            color: var(--bench-muted);
        }}
        .bench-brand {{
            display: flex;
            align-items: center;
            gap: .75rem;
            margin: .1rem 0 .85rem;
            padding: .2rem .15rem .75rem;
            border-bottom: 1px solid var(--bench-border);
        }}
        .bench-brand-mark {{
            display: grid;
            place-items: center;
            width: 2.25rem;
            height: 2.25rem;
            border-radius: 11px;
            color: white;
            background: linear-gradient(145deg, var(--bench-primary-strong), var(--bench-primary));
            box-shadow: 0 8px 20px color-mix(in srgb, var(--bench-primary) 28%, transparent);
            font-size: 1.05rem;
            font-weight: 800;
        }}
        .bench-brand-copy strong {{
            display: block;
            color: var(--bench-ink);
            font-size: .98rem;
            letter-spacing: -.015em;
        }}
        .bench-brand-copy span {{
            color: var(--bench-muted);
            font-size: .76rem;
        }}
        .bench-sidebar-label {{
            color: var(--bench-subtle);
            font-size: .67rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin: 1rem 0 .38rem;
        }}
        .bench-sidebar-summary {{
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            gap: .3rem .8rem;
            margin: .7rem 0 .25rem;
            padding: .78rem .85rem;
            border: 1px solid var(--bench-border);
            border-radius: 13px;
            background: var(--bench-soft);
            color: var(--bench-muted);
            font-size: .76rem;
        }}
        .bench-sidebar-summary strong {{ color: var(--bench-ink); font-size: .84rem; }}
        .bench-sidebar-summary .bench-live {{ color: var(--bench-positive); }}

        .st-key-page [role="radiogroup"] {{ gap: .32rem; }}
        .st-key-page [role="radiogroup"] label {{
            width: 100%;
            min-height: 2.55rem;
            padding: .56rem .68rem;
            border: 1px solid transparent;
            border-radius: 11px;
            background: transparent;
            transition: background .16s ease, border-color .16s ease, transform .16s ease;
        }}
        .st-key-page [role="radiogroup"] label:hover {{
            background: var(--bench-soft);
            border-color: var(--bench-border);
            transform: translateX(2px);
        }}
        .st-key-page [role="radiogroup"] label:has(input:checked) {{
            background: var(--bench-accent-soft);
            border-color: color-mix(in srgb, var(--bench-primary) 42%, var(--bench-border));
            box-shadow: inset 3px 0 0 var(--bench-primary);
        }}
        .st-key-page [data-testid="stRadioOption"] > div > div > div:first-child {{
            display: none;
        }}
        .st-key-page [role="radiogroup"] label:has(input:checked) p {{
            color: var(--bench-primary-text) !important;
            font-weight: 720;
        }}

        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"],
        input, textarea {{
            color: var(--bench-ink) !important;
            background: var(--bench-raised) !important;
            border-color: var(--bench-border-strong) !important;
        }}
        [data-baseweb="select"] span,
        [data-baseweb="select"] div,
        [data-baseweb="input"] input {{ color: var(--bench-ink) !important; }}
        [data-baseweb="select"] > div > div:last-child {{
            background: var(--bench-raised) !important;
        }}
        [data-baseweb="select"] svg {{
            color: var(--bench-muted) !important;
            fill: var(--bench-muted) !important;
        }}
        .react-aria-ComboBox [role="group"] {{
            border: 1px solid var(--bench-border-strong) !important;
            border-radius: 9px !important;
            background: var(--bench-raised) !important;
            overflow: hidden;
        }}
        .react-aria-ComboBox [role="group"] input {{
            border: 0 !important;
            background: var(--bench-raised) !important;
        }}
        .react-aria-ComboBox [role="group"] button {{
            color: var(--bench-muted) !important;
            border: 0 !important;
            background: var(--bench-raised) !important;
        }}
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="menu"],
        [role="listbox"] {{
            color: var(--bench-ink) !important;
            background: var(--bench-raised) !important;
            border: 1px solid var(--bench-border);
            box-shadow: 0 14px 36px var(--bench-shadow);
        }}
        [role="option"] {{
            color: var(--bench-ink) !important;
            background: var(--bench-raised) !important;
        }}
        [role="option"]:hover {{ background: var(--bench-soft) !important; }}
        [role="option"][aria-selected="true"] {{
            color: var(--bench-primary-text) !important;
            background: var(--bench-accent-soft) !important;
        }}
        [data-baseweb="tag"] {{
            color: var(--bench-primary-text) !important;
            background: var(--bench-accent-soft) !important;
            border: 0 !important;
            box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--bench-primary) 35%, transparent);
        }}
        [data-testid="stButtonGroup"] [role="radiogroup"] {{
            width: 100%;
            padding: .18rem;
            border: 1px solid var(--bench-border);
            border-radius: 11px;
            background: var(--bench-soft);
        }}
        [data-testid="stButtonGroup"] button[data-variant="segmented_control"] {{
            flex: 1;
            color: var(--bench-muted) !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            background: transparent !important;
        }}
        [data-testid="stButtonGroup"] button[data-variant="segmented_control"] p {{
            color: var(--bench-muted) !important;
        }}
        [data-testid="stButtonGroup"] button[aria-checked="true"] {{
            color: var(--bench-ink) !important;
            background: var(--bench-raised) !important;
            box-shadow: 0 1px 3px var(--bench-shadow);
        }}
        [data-testid="stButtonGroup"] button[aria-checked="true"] p {{
            color: var(--bench-ink) !important;
        }}

        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {{
            color: var(--bench-ink);
            border-color: var(--bench-border-strong);
            border-radius: 10px;
            background: var(--bench-raised);
        }}
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {{
            color: var(--bench-primary-text);
            border-color: var(--bench-primary);
            background: var(--bench-accent-soft);
        }}
        [data-testid="stExpander"] {{
            border: 1px solid var(--bench-border) !important;
            border-radius: 12px !important;
            background: color-mix(in srgb, var(--bench-raised) 78%, transparent);
            overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{
            color: var(--bench-ink) !important;
            background: var(--bench-soft) !important;
        }}
        [data-testid="stExpander"] summary p {{ color: var(--bench-ink) !important; }}
        [data-testid="stExpanderDetails"] {{
            background: var(--bench-raised);
            border-top: 1px solid var(--bench-border);
        }}

        [data-testid="stMetric"] {{
            min-height: 112px;
            padding: 1rem 1.05rem;
            border: 1px solid var(--bench-border);
            border-radius: 15px;
            background: linear-gradient(
                145deg,
                var(--bench-raised),
                color-mix(in srgb, var(--bench-soft) 45%, var(--bench-raised))
            );
            box-shadow: 0 5px 18px color-mix(in srgb, var(--bench-shadow) 50%, transparent);
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
            color: var(--bench-muted) !important;
        }}
        [data-testid="stMetricLabel"] p {{ white-space: normal; line-height: 1.2; }}
        [data-testid="stMetricValue"] {{
            color: var(--bench-ink);
            letter-spacing: -.035em;
        }}
        [data-testid="stMetricDelta"] {{ color: var(--bench-positive); }}

        div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {{
            border: 1px solid var(--bench-border);
            border-radius: 15px;
            overflow: hidden;
            background: var(--bench-surface);
            box-shadow: 0 5px 20px color-mix(in srgb, var(--bench-shadow) 44%, transparent);
        }}
        [role="tablist"] {{
            width: fit-content;
            gap: .25rem;
            padding: .25rem;
            border: 1px solid var(--bench-border);
            border-radius: 12px;
            background: var(--bench-soft);
        }}
        [data-testid="stTab"] {{
            height: 2.45rem;
            padding: 0 .85rem;
            color: var(--bench-muted) !important;
            border-radius: 9px;
        }}
        [data-testid="stTab"] p {{ color: var(--bench-muted) !important; }}
        [data-testid="stTab"][aria-selected="true"] {{
            color: var(--bench-ink) !important;
            background: var(--bench-raised);
            box-shadow: 0 1px 4px var(--bench-shadow);
        }}
        [data-testid="stTab"][aria-selected="true"] p {{
            color: var(--bench-ink) !important;
        }}
        .react-aria-SelectionIndicator {{ display: none; }}

        .bench-kicker {{
            color: var(--bench-primary-text);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .15em;
            text-transform: uppercase;
            margin-bottom: -.35rem;
        }}
        .bench-topline {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .1rem;
        }}
        .bench-context-chip {{
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            padding: .35rem .6rem;
            border: 1px solid var(--bench-border);
            border-radius: 999px;
            color: var(--bench-muted);
            background: var(--bench-raised);
            font-size: .72rem;
            white-space: nowrap;
        }}
        .bench-context-chip::before {{
            content: '';
            width: .42rem;
            height: .42rem;
            border-radius: 50%;
            background: var(--bench-positive);
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--bench-positive) 15%, transparent);
        }}
        .bench-page {{
            position: relative;
            overflow: hidden;
            margin: .8rem 0 1rem;
            padding: 1.2rem 1.3rem;
            border: 1px solid var(--bench-border);
            border-radius: 17px;
            background: linear-gradient(
                120deg,
                var(--bench-raised),
                color-mix(in srgb, var(--bench-accent-soft) 42%, var(--bench-raised))
            );
            box-shadow: 0 5px 24px color-mix(in srgb, var(--bench-shadow) 40%, transparent);
        }}
        .bench-page::after {{
            content: '';
            position: absolute;
            width: 10rem;
            height: 10rem;
            right: -4rem;
            top: -6rem;
            border-radius: 50%;
            background: color-mix(in srgb, var(--bench-primary) 12%, transparent);
            pointer-events: none;
        }}
        .bench-page h2 {{
            color: var(--bench-ink);
            font-size: 1.5rem;
            letter-spacing: -.03em;
            margin: 0 0 .25rem;
        }}
        .bench-page p {{ color: var(--bench-muted); margin: 0; max-width: 82ch; }}
        .bench-run-card {{
            padding: .9rem 1rem;
            border: 1px solid var(--bench-border);
            border-left: 4px solid var(--bench-primary);
            border-radius: 12px;
            background: var(--bench-raised);
            color: var(--bench-muted);
            margin: .25rem 0 .75rem;
            box-shadow: 0 4px 14px color-mix(in srgb, var(--bench-shadow) 35%, transparent);
        }}
        .bench-run-card strong {{ color: var(--bench-ink); }}
        .bench-sim, .bench-measured {{
            padding: .78rem .95rem;
            border-radius: 11px;
            margin: .25rem 0 1rem;
        }}
        .bench-sim {{
            border: 1px solid color-mix(in srgb, var(--bench-warning) 42%, transparent);
            background: color-mix(in srgb, var(--bench-warning) 10%, var(--bench-raised));
            color: var(--bench-warning);
        }}
        .bench-measured {{
            border: 1px solid color-mix(in srgb, var(--bench-info) 40%, transparent);
            background: color-mix(in srgb, var(--bench-info) 9%, var(--bench-raised));
            color: var(--bench-info);
        }}
        .bench-section {{
            display: flex;
            align-items: center;
            gap: .5rem;
            color: var(--bench-muted);
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin: 1.35rem 0 .5rem;
        }}
        .bench-section::before {{
            content: '';
            width: .38rem;
            height: .38rem;
            border-radius: 50%;
            background: var(--bench-primary);
        }}
        [data-testid="stAlert"] {{
            color: var(--bench-ink);
            border-radius: 11px;
        }}

        @media (max-width: 900px) {{
            [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {{
                min-width: min(88vw, 336px) !important;
                width: min(88vw, 336px) !important;
            }}
            [data-testid="stMainBlockContainer"] {{ padding: 1rem 1rem 3rem; }}
            .bench-topline {{ align-items: flex-start; flex-direction: column; gap: .45rem; }}
            .bench-page {{ padding: 1rem; }}
            [role="tablist"] {{ width: 100%; overflow-x: auto; }}
        }}
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
                '<div class="bench-sim"><strong>Demo data</strong> · '
                f"{simulated} simulated run(s) · excluded from measured exports</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            (
                '<div class="bench-measured"><strong>Measured data</strong> · '
                "read-only filesystem evidence</div>"
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

    def choose(label: str, field: str) -> list[str]:
        options = values(field)
        if not options:
            return []
        placeholders = {
            "Platform": "All platforms",
            "Benchmark type": "All benchmark types",
            "Model": "All models",
            "Memory system": "All memory systems",
            "Status": "All statuses",
            "Precision": "All precisions",
            "Evidence source": "All evidence sources",
            "Backend": "All backends",
            "Execution profile": "All execution profiles",
            "Provider": "All providers",
            "Artifact variant": "All artifact variants",
            "Measurement method": "All measurement methods",
        }
        return st.multiselect(
            label,
            options,
            default=[],
            placeholder=placeholders[label],
            key=widget_key(f"filter_{field}"),
        )

    with st.sidebar.expander("Scope", expanded=True):
        st.caption("Choose the runs that feed every workspace view.")
        platforms = choose("Platform", "platform")
        benchmark_types = choose("Benchmark type", "benchmark_type")
        models = choose("Model", "model_id")

    with st.sidebar.expander("Advanced filters", expanded=False):
        memory = choose("Memory system", "memory_type")
        statuses = choose("Status", "status")
        precision = choose("Precision", "precision")
        sources = choose("Evidence source", "source")
        backends = choose("Backend", "backend")
        profiles = choose("Execution profile", "backend_profile")
        providers = choose("Provider", "provider")
        variants = choose("Artifact variant", "artifact_variant")
        methods = choose("Measurement method", "measurement_method")
        workload_query = (
            st.text_input(
                "Workload contains",
                help="Examples: input=256, concurrency=8, or batch=8",
                key=widget_key("filter_workload"),
            )
            .strip()
            .lower()
        )
        st.caption("An empty selection means all values in that dimension.")
        if st.button(
            "Reset all filters",
            width="stretch",
            key=widget_key("reset_filters"),
        ):
            for field in (
                "platform",
                "benchmark_type",
                "model_id",
                "memory_type",
                "status",
                "precision",
                "source",
                "backend",
                "backend_profile",
                "provider",
                "artifact_variant",
                "measurement_method",
                "workload",
            ):
                st.session_state.pop(widget_key(f"filter_{field}"), None)
            st.rerun()

    selections = {
        "platform": set(platforms),
        "benchmark_type": set(benchmark_types),
        "model_id": set(models),
        "memory_type": set(memory),
        "status": set(statuses),
        "precision": set(precision),
        "source": set(sources),
        "backend": set(backends),
        "backend_profile": set(profiles),
        "provider": set(providers),
        "artifact_variant": set(variants),
        "measurement_method": set(methods),
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
        "Runs, coverage, performance, and metric readiness.",
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
                font={"size": 15, "color": _theme_tokens()["muted"]},
            )
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
            st.plotly_chart(
                _apply_chart_style(figure, height=430),
                width="stretch",
                theme=None,
            )
            st.caption("Requires both latency and throughput.")
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
                color_discrete_map=_platform_colors(),
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
            st.caption("Upper-left is better · bubble size = peak memory.")

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
            color_discrete_sequence=_categorical_colors()[:3],
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
                [0.0, _theme_tokens()["surface_soft"]],
                [0.45, _theme_tokens()["surface_accent"]],
                [1.0, _theme_tokens()["primary"]],
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
        st.caption("Blank means unavailable, not zero.")

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
        color_discrete_sequence=[_theme_tokens()["primary"]],
    )
    figure.update_traces(marker={"size": 10}, line={"width": 2.5})
    aggregate = _finite(entry.summary.get(selected.field))
    if aggregate is not None and selected.aggregation in {"mean", "maximum"}:
        label = "summary mean" if selected.aggregation == "mean" else "summary maximum"
        figure.add_hline(
            y=aggregate,
            line_dash="dash",
            line_color=_theme_tokens()["muted"],
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
            "SD "
            f"{variability['standard_deviation']:.4g} {selected.unit}; "
            f"range {variability['minimum']:.4g}–{variability['maximum']:.4g}."
        )
    elif selected.aggregation in {"sum", "weighted"}:
        st.caption("Run aggregate omitted for non-mean metrics.")

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
                        _theme_tokens()["gpu"]
                        if "gpu" in field
                        else _theme_tokens()["hbm"]
                        if "cpu" in field
                        else _theme_tokens()["ddr"]
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
                font={"size": 14, "color": _theme_tokens()["muted"]},
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
        st.caption("GPU utilization = mean · memory/power = allocation total.")
    else:
        st.caption("CPU telemetry follows the recorded package/node scope.")


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
        "Metrics, telemetry, controls, and evidence for one run.",
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
        st.info("Smoke validation only.")
    else:
        st.info("Compare only runs with matching controls.")

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
            ("measurement_method", "Measurement method"),
            ("measurement_scope", "Measurement scope"),
            ("workload_manifest_sha256", "Workload manifest SHA-256"),
            ("input_length", "Input length"),
            ("output_length", "Output length"),
            ("number_of_requests", "Requests per repetition"),
            ("request_rate", "Request rate"),
            ("maximum_concurrency", "Maximum concurrency"),
            ("batch_size", "Batch size"),
            ("warmup_runs", "Warm-up runs"),
            ("backend_internal_warmup", "Backend-internal warm-up"),
            ("repetitions", "Measured repetitions configured"),
            ("seed", "Seed"),
        )
        hardware_fields = (
            ("backend", "Backend"),
            ("backend_version", "Backend version"),
            ("backend_profile", "Execution profile"),
            ("execution_provider", "Provider"),
            ("container_image_digest", "Container digest"),
            ("container_id", "Container ID"),
            ("runtime_environment_variable", "Runtime selector variable"),
            ("native_binary_path", "Native binary path"),
            ("model_artifact_format", "Artifact format"),
            ("model_artifact_variant", "Artifact variant"),
            ("model_artifact_sha256", "Artifact SHA-256"),
            ("hardware_type", "Hardware type"),
            ("accelerator_name", "Accelerator"),
            ("accelerator_count", "Accelerator count"),
            ("cpu_model", "CPU model"),
            ("socket_count", "Sockets"),
            ("numa_node_count", "NUMA nodes"),
            ("memory_type", "Memory type"),
            ("memory_mode", "Requested memory mode"),
            ("memory_mode_detected", "Detected memory mode"),
            ("memory_mode_detection_method", "Memory-mode evidence"),
            ("memory_mode_verified", "Memory mode verified"),
            ("memory_capacity_gib", "Memory capacity (GiB)"),
            ("thread_count", "Thread count"),
            ("thread_count_batch", "Batch thread count"),
            ("cpu_mask", "CPU mask"),
            ("process_count", "Process count"),
            ("thread_affinity", "Thread affinity"),
            ("numa_policy", "NUMA policy"),
            ("memory_binding", "Memory binding"),
            ("memory_binding_resolved", "Resolved memory binding"),
            ("vllm_cpu_kvcache_space_gib", "vLLM CPU KV cache (GiB)"),
            ("vllm_cpu_omp_threads_bind", "vLLM CPU OpenMP binding"),
            ("vllm_cpu_num_reserved_cpu", "vLLM CPU reserved cores/rank"),
            ("gpu_layers", "GPU-offloaded layers"),
            ("ubatch_size", "Physical batch size"),
            ("parallel_slots", "Parallel server slots"),
            ("cpu_isa_target", "CPU ISA target"),
            ("cpu_isa", "Detected CPU ISA"),
            ("cpu_features_required", "Required CPU features"),
            ("cpu_features_detected", "Detected CPU features"),
            ("cpu_isa_verified", "CPU ISA verified"),
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
        provenance_fields = (
            ("model_id", "Canonical model"),
            ("model_revision_policy", "Revision policy"),
            ("resolved_model_revision", "Resolved model revision"),
            ("tokenizer_id", "Canonical tokenizer"),
            ("resolved_tokenizer_revision", "Resolved tokenizer revision"),
            ("model_artifact_source_revision", "Artifact source revision"),
            ("model_artifact_sha256", "Artifact SHA-256"),
            ("container_image_digest", "Container/image SHA-256"),
            ("native_binary_version", "Native binary version"),
            ("workload_manifest_sha256", "Workload manifest SHA-256"),
            ("git_commit", "Runner Git commit"),
            ("configuration_hashes", "Source-config hashes"),
        )
        st.subheader("Provenance")
        provenance = _control_rows(summary, provenance_fields)
        if provenance:
            st.dataframe(pd.DataFrame(provenance), width="stretch", hide_index=True)
        else:
            st.info("This legacy result does not contain structured provenance fields.")
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
    color_map = _platform_colors() if color == "platform" else None
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
    st.caption("Each point is one run.")
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
        color_discrete_map=_platform_colors(),
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
    st.caption(f"X: {x_spec.direction} · Y: {y_spec.direction} · no composite score.")


def _render_explorer(
    st: Any, pd: Any, px: Any, entries: Sequence[CatalogEntry]
) -> None:
    _page_header(
        st,
        "Analysis canvas",
        "Explorer",
        "Slice a metric or inspect a two-metric trade-off.",
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


def _render_cpu_memory_study(
    st: Any, pd: Any, px: Any, entries: Sequence[CatalogEntry]
) -> None:
    cpu_entries = [entry for entry in entries if entry.summary.get("hardware_type") == "cpu"]
    if not cpu_entries:
        st.info("No CPU memory results in the current scope.")

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
                font={"size": 14, "color": _theme_tokens()["muted"]},
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
                color_discrete_map={
                    "HBM2e": _theme_tokens()["hbm"],
                    "DDR5": _theme_tokens()["ddr"],
                },
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
                color_discrete_map={
                    "HBM2e": _theme_tokens()["hbm"],
                    "DDR5": _theme_tokens()["ddr"],
                },
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
                font={"size": 14, "color": _theme_tokens()["muted"]},
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
        st.info("No control-matched DDR/HBM pairs.")
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
            font={"size": 14, "color": _theme_tokens()["muted"]},
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
                color_continuous_scale=[
                    _theme_tokens()["negative"],
                    _theme_tokens()["surface_soft"],
                    _theme_tokens()["positive"],
                ],
            color_continuous_midpoint=0,
            custom_data=["DDR", "HBM", "Source"],
            title=f"Direction-aware HBM effect — {selected.label}",
            labels={"HBM improvement": "HBM improvement vs. DDR (%)"},
        )
        figure.add_vline(x=0, line_color=_theme_tokens()["muted"], line_width=1)
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
    st.caption("Positive = HBM moved in the preferred direction.")

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
                "Resolved binding": summary.get("memory_binding_resolved"),
                "Requested mode": summary.get("memory_mode"),
                "Detected mode": summary.get("memory_mode_detected"),
                "Mode verified": summary.get("memory_mode_verified"),
                "ISA target": summary.get("cpu_isa_target"),
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
                "Resolved binding",
                "Requested mode",
                "Detected mode",
                "Mode verified",
                "ISA target",
                "Source",
            ),
        ),
        width="stretch",
        hide_index=True,
        key="memory_control_matrix",
    )


def _platform_study_key(entry: CatalogEntry) -> tuple[Any, ...]:
    summary = entry.summary
    return (
        summary.get("model_id"),
        summary.get("benchmark_type"),
        summary.get("dtype"),
        summary.get("input_length"),
        summary.get("output_length"),
        summary.get("batch_size"),
    )


def _matched_platform_keys(entries: Sequence[CatalogEntry]) -> set[tuple[Any, ...]]:
    grouped: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for entry in entries:
        if entry.summary.get("status") != "completed":
            continue
        hardware_type = str(entry.summary.get("hardware_type") or "").lower()
        if hardware_type in {"gpu", "cpu"}:
            grouped[_platform_study_key(entry)].add(hardware_type)
    return {key for key, platforms in grouped.items() if platforms == {"gpu", "cpu"}}


def _render_gpu_cpu_study(
    st: Any, pd: Any, px: Any, entries: Sequence[CatalogEntry]
) -> None:
    study_entries = [
        entry
        for entry in entries
        if entry.summary.get("status") == "completed"
        and entry.summary.get("hardware_type") in {"gpu", "cpu"}
    ]
    gpu_entries = [
        entry for entry in study_entries if entry.summary.get("hardware_type") == "gpu"
    ]
    cpu_entries = [
        entry for entry in study_entries if entry.summary.get("hardware_type") == "cpu"
    ]
    matched_keys = _matched_platform_keys(study_entries)

    columns = st.columns(4)
    columns[0].metric("GPU runs", len(gpu_entries))
    columns[1].metric("CPU runs", len(cpu_entries))
    columns[2].metric("Aligned shapes", len(matched_keys))
    columns[3].metric(
        "Platforms",
        len({hardware_platform(entry.summary) for entry in study_entries}),
    )

    if not gpu_entries or not cpu_entries:
        st.info("GPU and CPU results are both required for this study.")
        return

    matched_only = st.toggle(
        "Aligned shapes only",
        value=bool(matched_keys),
        help="Same model, benchmark, precision, input/output shape, and batch size.",
        key="platform_study_matched_only",
    )
    plotted_entries = (
        [entry for entry in study_entries if _platform_study_key(entry) in matched_keys]
        if matched_only
        else study_entries
    )
    frame = _analysis_frame(pd, plotted_entries)
    shared_specs = [
        spec
        for spec in _available_specs(plotted_entries)
        if spec.scope == "shared"
        and any(_finite(entry.summary.get(spec.field)) is not None for entry in gpu_entries)
        and any(_finite(entry.summary.get(spec.field)) is not None for entry in cpu_entries)
    ]
    selected = st.selectbox(
        "Metric",
        shared_specs,
        index=_default_spec(shared_specs, "offline"),
        format_func=lambda spec: spec.label,
        key="platform_study_metric",
    )

    chart_frame = frame.dropna(subset=[selected.field]).copy()
    chart_frame["Run"] = chart_frame.apply(
        lambda row: (
            f"{row['model_scale']} · {row['benchmark_type']} · "
            f"{row['platform']} · in={row['input_length']}"
        ),
        axis=1,
    )
    figure = px.bar(
        chart_frame,
        x="Run",
        y=selected.field,
        color="platform",
        barmode="group",
        color_discrete_map=_platform_colors(),
        title=f"{selected.label} · GPU vs CPU",
        labels={selected.field: f"{selected.label} ({selected.unit})", "platform": "Platform"},
        custom_data=["run_id", "workload", "source"],
    )
    figure.update_xaxes(tickangle=-24, title_text="")
    figure.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>Value: %{y:.4g}<br>Run: %{customdata[0]}<br>"
            "Workload: %{customdata[1]}<br>Source: %{customdata[2]}<extra></extra>"
        )
    )
    st.plotly_chart(
        _apply_chart_style(figure, height=430),
        width="stretch",
        theme=None,
        key="platform_study_metric_chart",
    )

    left, right = st.columns(2)
    with left:
        tradeoff = frame.dropna(
            subset=["mean_e2e_latency_ms", "output_throughput_tokens_per_second"]
        )
        figure = px.scatter(
            tradeoff,
            x="mean_e2e_latency_ms",
            y="output_throughput_tokens_per_second",
            color="platform",
            symbol="model_scale",
            color_discrete_map=_platform_colors(),
            title="Latency vs throughput",
            labels={
                "mean_e2e_latency_ms": "Mean latency (ms)",
                "output_throughput_tokens_per_second": "Output throughput (tokens/s)",
                "platform": "Platform",
                "model_scale": "Model",
            },
        )
        st.plotly_chart(
            _apply_chart_style(figure, height=390),
            width="stretch",
            theme=None,
            key="platform_study_tradeoff",
        )
    with right:
        memory_frame = frame.dropna(
            subset=["peak_memory_gib", "output_throughput_tokens_per_second"]
        )
        figure = px.scatter(
            memory_frame,
            x="peak_memory_gib",
            y="output_throughput_tokens_per_second",
            color="platform",
            symbol="model_scale",
            color_discrete_map=_platform_colors(),
            title="Memory footprint vs throughput",
            labels={
                "peak_memory_gib": "Peak memory (GiB)",
                "output_throughput_tokens_per_second": "Output throughput (tokens/s)",
                "platform": "Platform",
                "model_scale": "Model",
            },
        )
        st.plotly_chart(
            _apply_chart_style(figure, height=390),
            width="stretch",
            theme=None,
            key="platform_study_memory",
        )


def _render_memory_study(
    st: Any, pd: Any, px: Any, entries: Sequence[CatalogEntry]
) -> None:
    _page_header(
        st,
        "Hardware lens",
        "Memory & platform study",
        "GPU vs CPU performance and CPU DDR vs HBM effects.",
    )
    _simulated_notice(st, entries)
    mode = st.segmented_control(
        "Study",
        ("GPU vs CPU", "CPU DDR vs HBM"),
        default="GPU vs CPU",
        format_func=lambda value: (
            f":material/{'developer_board' if value == 'GPU vs CPU' else 'memory'}: {value}"
        ),
        key="memory_study_mode",
        label_visibility="collapsed",
        width="stretch",
    )
    if mode == "CPU DDR vs HBM":
        _render_cpu_memory_study(st, pd, px, entries)
    else:
        _render_gpu_cpu_study(st, pd, px, entries)


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


def _comparison_compatibility(
    baseline: CatalogEntry,
    candidate: CatalogEntry,
    lens: str,
) -> dict[str, Any]:
    if baseline.key == candidate.key:
        return {
            "status": "baseline",
            "allowed": lens != "Absolute values",
            "evidence": [],
        }
    if lens == "Absolute values":
        return {"status": "descriptive", "allowed": False, "evidence": []}
    if lens == "CPU memory treatment":
        result = _memory_treatment_compatibility(
            baseline.summary, candidate.summary
        )
        return {
            "status": result["status"],
            "allowed": result["status"] == "compatible",
            "evidence": result["evidence"],
        }
    if lens == "Backend treatment":
        result = check_backend_treatment_compatibility(
            baseline.summary, candidate.summary
        )
        return {
            "status": result["status"],
            "allowed": result["status"] == "compatible",
            "evidence": result["evidence"],
        }

    comparison = compare_summaries(baseline.summary, candidate.summary)
    compatibility = comparison["compatibility"]
    evidence = [
        {"kind": "mismatch", **item}
        for item in compatibility["mismatched_fields"]
    ] + [
        {"kind": "missing", **item}
        for item in compatibility["missing_fields"]
    ]
    return {
        "status": compatibility["status"],
        "allowed": compatibility["status"] == "compatible",
        "evidence": evidence,
    }


def _comparison_metric_rows(
    entries: Sequence[CatalogEntry],
    baseline: CatalogEntry,
    compatibility: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for spec in METRIC_SPECS:
        baseline_value = _finite(baseline.summary.get(spec.field))
        for entry in entries:
            value = _finite(entry.summary.get(spec.field))
            if value is None:
                continue
            allowed = bool(compatibility[entry.key]["allowed"])
            ratio = (
                value / baseline_value
                if allowed and baseline_value not in (None, 0.0)
                else None
            )
            preferred_change = None
            if ratio is not None and spec.direction != "descriptive":
                raw_change = (ratio - 1) * 100
                preferred_change = (
                    raw_change if spec.direction == "higher" else -raw_change
                )
            rows.append(
                {
                    "Category": spec.category,
                    "Metric": spec.label,
                    "Unit": spec.unit,
                    "Run": entry.summary.get("run_id"),
                    "Platform": hardware_platform(entry.summary),
                    "Value": value,
                    "Ratio vs baseline": ratio,
                    "Preferred change (%)": preferred_change,
                    "Compatibility": compatibility[entry.key]["status"],
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
        "Multi-run contrast",
        "Compare runs",
        "Select 2–6 runs, choose a baseline, and visualize one metric.",
    )
    if len(entries) < 2:
        st.info("At least two runs are required.")
        return

    namespace = hashlib.sha1(
        "|".join(entry.key for entry in entries).encode("utf-8")
    ).hexdigest()[:8]
    selected = st.multiselect(
        "Runs",
        entries,
        default=list(entries[: min(3, len(entries))]),
        max_selections=min(6, len(entries)),
        format_func=_entry_label,
        key=f"comparison_runs_{namespace}",
        placeholder="Choose 2–6 runs",
    )
    if len(selected) < 2:
        st.warning("Select at least two runs.")
        return
    _simulated_notice(st, selected)

    controls = st.columns(2)
    lens = controls[0].selectbox(
        "Lens",
        ("Absolute values", "Like-for-like", "Backend treatment", "CPU memory treatment"),
        key="comparison_lens",
    )
    baseline = controls[1].selectbox(
        "Baseline",
        selected,
        format_func=_entry_label,
        key=f"comparison_baseline_{namespace}",
    )
    compatibility = {
        entry.key: _comparison_compatibility(baseline, entry, lens)
        for entry in selected
    }
    comparable = sum(
        item["allowed"]
        for key, item in compatibility.items()
        if key != baseline.key
    )
    available_specs = _available_specs(selected)
    platforms = {hardware_platform(entry.summary) for entry in selected}
    columns = st.columns(4)
    columns[0].metric("Runs", len(selected))
    columns[1].metric("Platforms", len(platforms))
    columns[2].metric("Metrics", len(available_specs))
    columns[3].metric("Comparable", comparable if lens != "Absolute values" else "—")

    if lens != "Absolute values":
        status_rows = [
            {
                "Run": entry.summary.get("run_id"),
                "Platform": hardware_platform(entry.summary),
                "Status": compatibility[entry.key]["status"],
                "Issues": len(compatibility[entry.key]["evidence"]),
            }
            for entry in selected
        ]
        st.dataframe(
            pd.DataFrame(status_rows),
            width="stretch",
            hide_index=True,
            key="comparison_compatibility_matrix",
        )

    if not available_specs:
        st.info("No shared numeric metrics are available.")
        return
    categories = list(dict.fromkeys(spec.category for spec in available_specs))
    preferred_field = (
        "request_throughput_requests_per_second"
        if baseline.summary.get("benchmark_type") == "serving"
        else "output_throughput_tokens_per_second"
    )
    preferred_spec = next(
        (item for item in available_specs if item.field == preferred_field),
        available_specs[0],
    )
    metric_controls = st.columns(2)
    category = metric_controls[0].selectbox(
        "Metric family",
        categories,
        index=categories.index(preferred_spec.category),
        key="comparison_metric_category",
    )
    category_specs = [spec for spec in available_specs if spec.category == category]
    spec = metric_controls[1].selectbox(
        "Metric",
        category_specs,
        index=_default_spec(category_specs, str(baseline.summary.get("benchmark_type"))),
        format_func=lambda item: item.label,
        key="comparison_metric",
    )

    plot_rows = []
    legacy_runs = 0
    for entry in selected:
        value = _finite(entry.summary.get(spec.field))
        if value is None:
            continue
        measurements, legacy = measurements_for_entry(dataset, entry)
        legacy_runs += int(legacy)
        variability = (
            repetition_variability(measurements, spec.field)
            if spec.aggregation == "mean"
            else None
        )
        plot_rows.append(
            {
                "Run": (
                    f"{_short_model(entry.summary)} · "
                    f"{entry.summary.get('benchmark_type')} · "
                    f"{hardware_platform(entry.summary)} · "
                    f"in={entry.summary.get('input_length')}"
                ),
                "Platform": hardware_platform(entry.summary),
                "Value": value,
                "Error": variability["standard_deviation"] if variability else 0.0,
                "Run ID": entry.summary.get("run_id"),
                "Baseline": entry.key == baseline.key,
            }
        )
    plot_frame = pd.DataFrame(plot_rows)
    if plot_frame.empty:
        st.info("The selected runs do not contain this metric.")
        return

    figure = px.bar(
        plot_frame.sort_values("Value"),
        x="Value",
        y="Run",
        color="Platform",
        error_x="Error",
        orientation="h",
        color_discrete_map=_platform_colors(),
        title=spec.label,
        labels={"Value": f"{spec.label} ({spec.unit})"},
        custom_data=["Run ID", "Baseline"],
    )
    figure.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>Value: %{x:.4g}<br>Run: %{customdata[0]}"
            "<br>Baseline: %{customdata[1]}<extra></extra>"
        )
    )
    st.plotly_chart(
        _apply_chart_style(figure, height=max(340, 58 * len(plot_rows))),
        width="stretch",
        theme=None,
        key="comparison_metric_chart",
    )
    if legacy_runs:
        st.caption(f"{legacy_runs} run(s) use derived repetition data.")

    baseline_value = _finite(baseline.summary.get(spec.field))
    change_rows = []
    if spec.direction != "descriptive" and baseline_value not in (None, 0.0):
        for entry in selected:
            if entry.key == baseline.key or not compatibility[entry.key]["allowed"]:
                continue
            value = _finite(entry.summary.get(spec.field))
            if value is None:
                continue
            raw_change = (value / baseline_value - 1) * 100
            change_rows.append(
                {
                    "Run": f"{_short_model(entry.summary)} · {hardware_platform(entry.summary)}",
                    "Preferred change": (
                        raw_change if spec.direction == "higher" else -raw_change
                    ),
                    "Platform": hardware_platform(entry.summary),
                }
            )
    if change_rows:
        change_frame = pd.DataFrame(change_rows)
        figure = px.bar(
            change_frame,
            x="Preferred change",
            y="Run",
            color="Platform",
            orientation="h",
            color_discrete_map=_platform_colors(),
            title="Change vs baseline",
            labels={"Preferred change": "Preferred-direction change (%)"},
        )
        figure.add_vline(x=0, line_color=_theme_tokens()["muted"], line_width=1)
        st.plotly_chart(
            _apply_chart_style(figure, height=max(300, 54 * len(change_rows))),
            width="stretch",
            theme=None,
            key="comparison_change_chart",
        )

    evidence_rows = []
    for entry in selected:
        for item in compatibility[entry.key]["evidence"]:
            evidence_rows.append(
                {
                    "Run": entry.summary.get("run_id"),
                    "Field": item.get("field"),
                    "Baseline": repr(item.get("left")),
                    "Candidate": repr(item.get("right")),
                    "Reason": item.get("reason"),
                }
            )
    metric_rows = _comparison_metric_rows(selected, baseline, compatibility)
    metric_frame = pd.DataFrame(metric_rows)
    with st.expander("Data & compatibility", expanded=False):
        if evidence_rows:
            st.dataframe(pd.DataFrame(evidence_rows), width="stretch", hide_index=True)
        st.dataframe(metric_frame, width="stretch", hide_index=True)

    export = {
        "comparison_lens": lens,
        "baseline_run_id": baseline.summary.get("run_id"),
        "selected_run_ids": [entry.summary.get("run_id") for entry in selected],
        "compatibility": compatibility,
        "metrics": metric_rows,
    }
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "JSON",
        (json.dumps(export, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
        file_name="dashboard_comparison.json",
        mime="application/json",
        key="comparison_json",
        icon=":material/download:",
    )
    download_columns[1].download_button(
        "CSV",
        metric_frame.to_csv(index=False).encode("utf-8"),
        file_name="dashboard_comparison.csv",
        mime="text/csv",
        key="comparison_csv",
        icon=":material/download:",
    )


def run_dashboard(results_root: str | Path | None = None) -> None:
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
    resolved_root = (
        Path(results_root).expanduser().resolve() if results_root is not None else None
    )

    st.sidebar.markdown(
        (
            '<div class="bench-brand">'
            '<div class="bench-brand-mark">◈</div>'
            '<div class="bench-brand-copy"><strong>Benchmark Studio</strong>'
            '<span>GPU · CPU · HBM</span></div></div>'
        ),
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="bench-sidebar-label">Appearance</div>',
        unsafe_allow_html=True,
    )
    appearance = st.sidebar.segmented_control(
        "Appearance",
        ("Light", "Dark"),
        default="Light",
        format_func=lambda value: (
            f":material/{'light_mode' if value == 'Light' else 'dark_mode'}: {value}"
        ),
        key="ui_theme",
        label_visibility="collapsed",
        width="stretch",
    )
    theme = str(appearance or "Light").lower()
    _ACTIVE_THEME.set(theme)
    _inject_styles(st, theme)

    @st.cache_data(show_spinner=False)
    def cached_catalog(root: str) -> ResultCatalog:
        return discover_results(root)

    measured_catalog = (
        cached_catalog(str(resolved_root))
        if resolved_root is not None
        else ResultCatalog(Path.cwd(), (), ())
    )
    measured = dataset_from_catalog(measured_catalog)
    demo = simulated_dataset()
    source_options = (
        ("Measured results", "Demo study", "Measured + demo")
        if resolved_root is not None
        else ("Demo study",)
    )
    default_source = (
        "Measured results" if measured.catalog.entries else "Demo study"
    )
    st.sidebar.markdown(
        '<div class="bench-sidebar-label">Dataset</div>',
        unsafe_allow_html=True,
    )
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

    if resolved_root is not None:
        if st.sidebar.button(
            "Refresh results", width="stretch", key="refresh_catalog", icon=":material/refresh:"
        ):
            st.cache_data.clear()
            st.rerun()
    st.sidebar.markdown(
        '<div class="bench-sidebar-label">Workspace</div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Workspace view",
        tuple(_VIEW_META),
        format_func=lambda value: f":material/{_VIEW_META[value][0]}: {value}",
        key="page",
        label_visibility="collapsed",
    )
    st.sidebar.caption(_VIEW_META[page][1])
    filter_namespace = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    st.sidebar.markdown(
        '<div class="bench-sidebar-label">Filters</div>',
        unsafe_allow_html=True,
    )
    entries = _filter_entries(st, dataset, namespace=filter_namespace)
    simulated_visible = sum(
        bool(entry.summary.get("is_simulated")) for entry in entries
    )
    evidence_label = (
        "simulated preview"
        if entries and simulated_visible == len(entries)
        else "mixed evidence"
        if simulated_visible
        else "measured evidence"
    )
    st.sidebar.markdown(
        (
            '<div class="bench-sidebar-summary">'
            '<span>Visible runs</span>'
            f'<strong>{len(entries)} / {len(dataset.catalog.entries)}</strong>'
            '<span>Evidence mode</span>'
            f'<strong class="bench-live">{html.escape(evidence_label)}</strong>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="bench-topline">'
            '<div class="bench-kicker">Inference systems observatory</div>'
            f'<div class="bench-context-chip">{html.escape(source)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    st.title("LLM benchmark studio")
    st.caption(
        f"Results · {resolved_root}"
        if resolved_root is not None
        else "In-memory demo · no results directory selected"
    )

    page_keys = {
        "Overview": "overview",
        "Run detail": "run_detail",
        "Explorer": "explorer",
        "Memory study": "memory_study",
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
    with st.container(key="dashboard_page_memory_study"):
        _render_memory_study(st, pd, px, entries)
    with st.container(key="dashboard_page_compare"):
        _render_compare(st, pd, px, entries, dataset)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only benchmark results dashboard")
    parser.add_argument("--results-root", type=Path)
    args, _ = parser.parse_known_args(argv)
    run_dashboard(args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
