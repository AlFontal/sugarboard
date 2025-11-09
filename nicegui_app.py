"""NiceGUI dashboard for SugarBoard.

Full-featured CGM dashboard with minute-level hero refresh, TIR analytics,
recent glucose chart, and daily pattern visualization.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from pandas.api.types import is_datetime64_any_dtype
from nicegui import ui

# Import HbA1c calculation
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from utils import mean_glucose_to_hba1c

# Configuration
SITE = "cgm-monitor-alfontal.herokuapp.com"
LINEPLOT_HOURS = 4
RECENT_POINTS = LINEPLOT_HOURS * 75
RECENT_REQUEST_TIMEOUT = 20
TARGET_SEVERE_LOW = 50
TARGET_LOW = 70
TARGET_MILD_HIGH = 150
TARGET_HIGH = 180
TARGET_SEVERE_HIGH = 250
BG_CATEGORIES = [
    f"<{TARGET_SEVERE_LOW}",
    f"{TARGET_SEVERE_LOW}-{TARGET_LOW - 1}",
    f"{TARGET_LOW}-{TARGET_MILD_HIGH}",
    f"{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}",
    f"{TARGET_HIGH + 1}-{TARGET_SEVERE_HIGH}",
    f">{TARGET_SEVERE_HIGH}",
]
DIRECTIONS = {
    "DoubleDown": "⇊",
    "SingleDown": "↓",
    "FortyFiveDown": "↘",
    "Flat": "→",
    "FortyFiveUp": "↗",
    "SingleUp": "↑",
    "DoubleUp": "⇈",
}

# Colors
STRONG_RED = "#960200"
LIGHT_RED = "#CE6C47"
MILD_YELLOW = "#FFD046"
LIGHT_GREEN = "#49D49D"

# Caching
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
RECENT_CACHE = CACHE_DIR / "nicegui_recent.pkl"
HISTORICAL_CACHE = CACHE_DIR / "nicegui_historical.pkl"


@dataclass
class DataState:
    """Global application state."""
    last_value: Optional[Dict[str, Any]] = None
    previous_value: Optional[Dict[str, Any]] = None
    df_recent: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_3months: pd.DataFrame = field(default_factory=pd.DataFrame)
    fetched_at: Optional[float] = None
    historical_cached_at: Optional[float] = None


STATE = DataState()


# ============================================================================
# Cache helpers
# ============================================================================

def _load_pickle(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save_pickle(path: Path, data: Any) -> None:
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def load_historical_cache() -> Optional[pd.DataFrame]:
    cached = _load_pickle(HISTORICAL_CACHE)
    if not cached:
        return None
    STATE.historical_cached_at = cached.get("cached_at")
    return cached.get("df_3months")


def save_historical_cache(df: pd.DataFrame) -> None:
    _save_pickle(HISTORICAL_CACHE, {"df_3months": df, "cached_at": time.time()})


def load_recent_cache() -> None:
    cached = _load_pickle(RECENT_CACHE)
    if not cached:
        return
    STATE.last_value = cached.get("last_value")
    STATE.previous_value = cached.get("previous_value")
    STATE.df_recent = cached.get("df_recent", pd.DataFrame())
    STATE.fetched_at = cached.get("fetched_at")


def save_recent_cache() -> None:
    if STATE.df_recent.empty or STATE.last_value is None:
        return
    _save_pickle(
        RECENT_CACHE,
        {
            "last_value": STATE.last_value,
            "previous_value": STATE.previous_value,
            "df_recent": STATE.df_recent,
            "fetched_at": STATE.fetched_at,
        },
    )


# ============================================================================
# Data fetching
# ============================================================================

def fetch_recent_data() -> tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Fetch the most recent CGM entries and return structured data."""
    response = requests.get(
        f"https://{SITE}/api/v1/entries/sgv.json",
        params={"count": RECENT_POINTS},
        timeout=RECENT_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    entries = response.json()
    if not entries:
        raise ValueError("Nightscout returned no recent entries.")

    # Filter out xDrip-NSFollower device
    entries = [e for e in entries if e.get("device") != "xDrip-NSFollower"]
    
    if not entries:
        raise ValueError("No valid entries after filtering devices.")

    df_recent = pd.DataFrame(entries)
    print(df_recent.device.unique())
    if "dateString" not in df_recent:
        raise ValueError("Recent data payload is missing dateString.")

    df_recent["date"] = pd.to_datetime(df_recent["dateString"], utc=True)
    df_recent = (
        df_recent[["date", "sgv", "device"]]
        .drop_duplicates()
        .set_index("date")
        .sort_index()
    )

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=LINEPLOT_HOURS)
    df_recent = df_recent.loc[df_recent.index >= cutoff].reset_index()

    last_value = entries[0]
    previous_value = entries[1] if len(entries) > 1 else entries[0]
    return last_value, previous_value, df_recent


def fetch_latest_entry() -> Dict[str, Any]:
    """Fetch only the most recent CGM entry for minute-level refresh."""
    response = requests.get(
        f"https://{SITE}/api/v1/entries/sgv.json",
        params={"count": 10},  # Fetch a few entries in case top ones are filtered
        timeout=RECENT_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    entries = response.json()
    if not entries:
        raise ValueError("Nightscout returned no latest entry.")
    
    # Filter out xDrip-NSFollower device
    entries = [e for e in entries if e.get("device") != "xDrip-NSFollower"]
    
    if not entries:
        raise ValueError("No valid entries after filtering devices.")
    
    return entries[0]


def fetch_historical_data(days: int = 90, progress_callback: Optional[callable] = None) -> pd.DataFrame:
    """Fetch 90 days of historical data in manageable chunks."""
    chunk_size = 10_000
    delay_between_requests = 3
    request_timeout = 120

    estimated_records = days * 1440
    num_chunks = (estimated_records + chunk_size - 1) // chunk_size

    all_data: list[Dict[str, Any]] = []
    oldest_date: Optional[pd.Timestamp] = None

    for index in range(num_chunks):
        # Report progress
        if progress_callback:
            progress_callback(index + 1, num_chunks)
        
        if oldest_date is not None:
            url = (
                f"https://{SITE}/api/v1/entries/sgv.json?"
                f"find[dateString][$lt]={oldest_date.isoformat()}&count={chunk_size}"
            )
        else:
            url = f"https://{SITE}/api/v1/entries/sgv.json?count={chunk_size}"

        if index > 0:
            time.sleep(delay_between_requests)

        response = requests.get(url, timeout=request_timeout)
        response.raise_for_status()
        chunk_data = response.json()

        if not chunk_data:
            break

        # Filter out xDrip-NSFollower device
        chunk_data = [e for e in chunk_data if e.get("device") != "xDrip-NSFollower"]

        all_data.extend(chunk_data)

        if not chunk_data:
            # If all data was filtered out, we need to get the last timestamp from the original chunk
            # to continue pagination, so we need to re-fetch without filtering for pagination purposes
            response = requests.get(url, timeout=request_timeout)
            original_chunk = response.json()
            if original_chunk:
                newest_timestamp = pd.to_datetime(original_chunk[-1]["dateString"])
                if isinstance(newest_timestamp, pd.Timestamp) and newest_timestamp.tzinfo is not None:
                    newest_timestamp = newest_timestamp.tz_localize(None)
                oldest_date = newest_timestamp if isinstance(newest_timestamp, pd.Timestamp) else None
            continue

        newest_timestamp = pd.to_datetime(chunk_data[-1]["dateString"])
        if isinstance(newest_timestamp, pd.Timestamp) and newest_timestamp.tzinfo is not None:
            newest_timestamp = newest_timestamp.tz_localize(None)

        oldest_date = newest_timestamp if isinstance(newest_timestamp, pd.Timestamp) else None

        target_start = pd.Timestamp.now() - pd.Timedelta(days=days)
        if oldest_date is not None and oldest_date < target_start:
            break

    if not all_data:
        raise ValueError("Nightscout returned no historical data.")

    df_raw = pd.DataFrame(all_data)
    df_raw["date"] = pd.to_datetime(df_raw["dateString"])
    df_raw = (
        df_raw[["date", "sgv", "device"]]
        .drop_duplicates()
        .set_index("date")
        .sort_index()
    )

    df_3months = (
        df_raw.resample("5 min")["sgv"]
        .mean()
        .reset_index()
        .assign(
            cat_glucose=lambda frame: pd.cut(
                frame["sgv"],
                bins=[
                    0,
                    TARGET_SEVERE_LOW,
                    TARGET_LOW,
                    TARGET_MILD_HIGH,
                    TARGET_HIGH,
                    TARGET_SEVERE_HIGH,
                    np.inf,
                ],
                labels=BG_CATEGORIES,
            )
        )
    )

    return df_3months


# ============================================================================
# Data helpers
# ============================================================================

def parse_entry_timestamp(entry: Optional[Dict[str, Any]]) -> Optional[pd.Timestamp]:
    if not entry:
        return None
    ts_raw = entry.get("dateString") or entry.get("date")
    if ts_raw is None:
        return None
    return pd.to_datetime(ts_raw, utc=True)


def ensure_timezone_aware(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "date" not in df.columns:
        raise ValueError("Dataframe missing 'date' column.")
    if not is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


async def ensure_historical_data(refs: Optional[Any] = None) -> None:
    """Load or fetch the 90-day historical dataset."""
    cached_df = load_historical_cache()
    if cached_df is not None:
        print(f"✓ Loaded {len(cached_df)} historical records from cache")
        STATE.df_3months = cached_df
        return

    print("⏳ Fetching 90 days of historical data from API (this may take a while)...")
    
    def progress_update(current: int, total: int):
        """Update progress in UI if refs provided."""
        if refs and hasattr(refs, 'status_label'):
            percentage = int((current / total) * 100)
            # Terminal-style progress bar (instant update, no typing effect for smooth progress)
            bar_width = 20
            filled = int((current / total) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_text = f"$ fetching --chunks [{bar}] {current}/{total} ({percentage}%) "
            # Direct content update
            refs.status_label.content = f'<span id="terminal-status" class="terminal-cursor">{progress_text}</span>'
    
    df_3months = await asyncio.to_thread(fetch_historical_data, 90, progress_update)
    STATE.df_3months = df_3months
    save_historical_cache(df_3months)
    print(f"✓ Fetched and cached {len(df_3months)} historical records")


async def refresh_recent_data(full_refresh: bool = False) -> None:
    """Refresh recent data, optionally doing a full fetch."""
    if full_refresh or STATE.df_recent.empty or STATE.last_value is None:
        last_value, previous_value, df_recent = await asyncio.to_thread(fetch_recent_data)
        df_recent = ensure_timezone_aware(df_recent)
        STATE.last_value = last_value
        STATE.previous_value = previous_value
        STATE.df_recent = df_recent
        STATE.fetched_at = time.time()
        save_recent_cache()
        return

    latest_entry = await asyncio.to_thread(fetch_latest_entry)
    latest_ts = parse_entry_timestamp(latest_entry)
    cached_ts = parse_entry_timestamp(STATE.last_value)

    if latest_ts is None or (cached_ts is not None and latest_ts <= cached_ts):
        return

    previous_value = STATE.last_value
    df_recent = ensure_timezone_aware(STATE.df_recent)

    latest_row = pd.DataFrame(
        [
            {
                "date": latest_ts,
                "sgv": latest_entry.get("sgv"),
                "device": latest_entry.get("device", "Unknown") or "Unknown",
            }
        ]
    )

    df_recent = cast(
        pd.DataFrame,
        pd.concat([df_recent, latest_row], ignore_index=True)
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date"),
    )

    recent_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=LINEPLOT_HOURS)
    df_recent = cast(
        pd.DataFrame, df_recent.loc[df_recent["date"] >= recent_cutoff].reset_index(drop=True)
    )

    STATE.last_value = latest_entry
    STATE.previous_value = previous_value
    STATE.df_recent = df_recent
    STATE.fetched_at = time.time()
    save_recent_cache()


# ============================================================================
# Chart builders
# ============================================================================

def create_placeholder_chart(title: str = "Loading...", height: int = 360) -> go.Figure:
    """Create a styled placeholder chart matching the loaded chart aesthetics."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="JetBrains Mono, Consolas, monospace", size=11),
        title=dict(text=title, font=dict(size=14, color="#94a3b8")),
        xaxis=dict(showgrid=True, gridcolor="#334155", showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="#334155", showticklabels=False),
        height=height,
        margin=dict(t=60, r=20, b=40, l=50),
    )
    return fig


def build_recent_chart(df: pd.DataFrame) -> go.Figure:
    """Build recent glucose line chart."""
    if df.empty:
        return create_placeholder_chart("No data yet")

    # Convert timestamps to local timezone then to strings for JSON serialization
    df_plot = df.copy()
    # Convert to Europe/Madrid timezone before converting to string
    df_plot["date"] = df_plot["date"].dt.tz_convert("Europe/Madrid").astype(str)
    
    fig = go.Figure()
    
    # Add line trace (without markers)
    fig.add_trace(go.Scatter(
        x=df_plot["date"],
        y=df_plot["sgv"],
        mode='lines',
        line=dict(color="#8b5cf6", width=3),
        name="Glucose",
        hovertemplate="<b>%{y:.0f} mg/dL</b><br>%{x}<extra></extra>",
    ))
    
    # Add marker only for the last point
    if not df_plot.empty:
        last_row = df_plot.iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_row["date"]],
            y=[last_row["sgv"]],
            mode='markers',
            marker=dict(size=10, color="#8b5cf6", line=dict(width=2, color="#0f172a")),
            name="Current",
            hovertemplate="<b>Current: %{y:.0f} mg/dL</b><extra></extra>",
        ))
    
    # Add target range bands
    fig.add_hrect(
        y0=TARGET_LOW, y1=TARGET_MILD_HIGH,
        fillcolor=LIGHT_GREEN, opacity=0.1,
        layer="below", line_width=0,
    )
    fig.add_hrect(
        y0=TARGET_MILD_HIGH, y1=TARGET_HIGH,
        fillcolor=MILD_YELLOW, opacity=0.1,
        layer="below", line_width=0,
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="JetBrains Mono, Consolas, monospace", size=11),
        showlegend=False,
        hovermode="x unified",
        margin=dict(t=40, r=20, b=40, l=50),
        height=280,
        title_font=dict(size=14, color="#94a3b8"),
    )
    fig.update_yaxes(title="Glucose [mg/dL]", gridcolor="#334155", title_font=dict(color="#94a3b8"))
    fig.update_xaxes(title=None, gridcolor="#334155")
    return fig


def build_tir_chart(selected_df: pd.DataFrame):
    """Build TIR bar chart."""
    if selected_df.empty:
        return create_placeholder_chart("No data yet")
    
    tir_counts = selected_df["cat_glucose"].value_counts(normalize=True)
    tir_data = pd.DataFrame(
        {
            "cat_glucose": BG_CATEGORIES,
            "value": [tir_counts.get(cat, 0) for cat in BG_CATEGORIES],
            "percent_label": [f"{(tir_counts.get(cat, 0) * 100):.0f}%" for cat in BG_CATEGORIES],
        }
    )
    value_max = max(tir_data["value"].max(), 0.0001)

    fig = px.bar(
        tir_data,
        x="cat_glucose",
        y="value",
        color="cat_glucose",
        text="percent_label",
        category_orders={"cat_glucose": BG_CATEGORIES},
        color_discrete_map={
            BG_CATEGORIES[0]: STRONG_RED,
            BG_CATEGORIES[1]: LIGHT_RED,
            BG_CATEGORIES[2]: LIGHT_GREEN,
            BG_CATEGORIES[3]: MILD_YELLOW,
            BG_CATEGORIES[4]: LIGHT_RED,
            BG_CATEGORIES[5]: STRONG_RED,
        },
    )

    fig.update_traces(textposition="outside", cliponaxis=False, width=0.8, textfont=dict(size=12, color="#e2e8f0"))
    fig.update_yaxes(tickformat=".0%", title=None, range=[0, value_max * 1.15], gridcolor="#334155", title_font=dict(color="#94a3b8"))
    fig.update_xaxes(title=None, tickangle=-45, gridcolor="#334155")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="JetBrains Mono, Consolas, monospace", size=11),
        showlegend=False,
        height=360,
        margin=dict(t=50, r=20, b=70, l=40),
        title_font=dict(size=14, color="#94a3b8"),
    )
    return fig


def build_histogram_chart(selected_df: pd.DataFrame) -> go.Figure:
    """Build glucose histogram with TIR color mapping."""
    if selected_df.empty:
        return create_placeholder_chart("No data yet")
    
    # Filter out extreme values and cap at 300
    df_hist = selected_df.copy()
    df_hist['sgv_capped'] = df_hist['sgv'].clip(upper=300)
    
    # Create bins with 5 mg/dL width, last bin is ">300"
    bins = list(range(0, 305, 5))
    
    # Calculate histogram
    hist_data, bin_edges = np.histogram(df_hist['sgv_capped'], bins=bins)
    
    # Convert to percentages
    total_count = hist_data.sum()
    hist_pct = (hist_data / total_count * 100) if total_count > 0 else hist_data
    
    # Create bin centers and labels
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    
    # Assign colors based on TIR categories
    def get_color_for_bin(bin_center):
        if bin_center < TARGET_SEVERE_LOW:
            return STRONG_RED
        elif bin_center < TARGET_LOW:
            return LIGHT_RED
        elif bin_center <= TARGET_MILD_HIGH:
            return LIGHT_GREEN
        elif bin_center <= TARGET_HIGH:
            return MILD_YELLOW
        elif bin_center <= TARGET_SEVERE_HIGH:
            return LIGHT_RED
        else:
            return STRONG_RED
    
    colors = [get_color_for_bin(center) for center in bin_centers]
    
    # Create the histogram figure
    fig = go.Figure(data=[
        go.Bar(
            x=bin_centers,
            y=hist_pct,
            marker_color=colors,
            width=4.5,  # Slightly less than bin width for visual separation
            hovertemplate="<b>%{customdata}</b><br>%{y:.1f}%<extra></extra>",
            customdata=bin_labels,
        )
    ])
    
    # Add target lines
    for target, color, name in [
        (TARGET_LOW, "#10b981", "Target Low"),
        (TARGET_MILD_HIGH, "#10b981", "Target Mild High"),
        (TARGET_HIGH, "#f59e0b", "High"),
    ]:
        fig.add_vline(
            x=target,
            line_dash="dash",
            line_color=color,
            opacity=0.7,
            line_width=2,
        )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="JetBrains Mono, Consolas, monospace", size=11),
        showlegend=False,
        height=360,
        margin=dict(t=50, r=20, b=70, l=40),
        title_font=dict(size=14, color="#94a3b8"),
        xaxis=dict(
            title="Glucose [mg/dL]",
            gridcolor="#334155",
            title_font=dict(color="#94a3b8"),
            range=[0, 305],
        ),
        yaxis=dict(
            title="Percentage",
            gridcolor="#334155",
            title_font=dict(color="#94a3b8"),
            ticksuffix="%",
        ),
    )
    
    return fig


def build_pattern_chart(sel_df: pd.DataFrame) -> tuple[go.Figure, str, int, int]:
    """Build glucose pattern chart with quantile bands."""
    if sel_df.empty:
        fig = go.Figure(layout=dict(title="No data in range"))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0", family="JetBrains Mono, Consolas, monospace"),
        )
        return fig, "No data in range", 0, 0

    window_text = f"{sel_df.date.min().date()} → {sel_df.date.max().date()}"
    valid_sgv_count = int(sel_df.sgv.notna().sum())

    quantiles = (
        sel_df.dropna(subset=["sgv"])
        .assign(hour=lambda frame: frame.date.dt.hour + frame.date.dt.minute / 60)
        .groupby("hour", as_index=False)
        .agg(
            {
                "sgv": [
                    ("median", "median"),
                    ("q90", lambda x: x.quantile(0.9)),
                    ("q10", lambda x: x.quantile(0.1)),
                    ("q25", lambda x: x.quantile(0.25)),
                    ("q75", lambda x: x.quantile(0.75)),
                ]
            }
        )
    )
    quantiles.columns = ["hour", "median", "q90", "q10", "q25", "q75"]

    if len(quantiles) >= 10:
        quantiles[["median", "q90", "q10", "q25", "q75"]] = (
            quantiles[["median", "q90", "q10", "q25", "q75"]]
            .rolling(window=10, center=True, min_periods=1)
            .mean()
        )

    quantiles["hour_formatted"] = quantiles["hour"].apply(
        lambda value: f"{int(value):02d}:{int((value % 1) * 60):02d}"
    )

    fig = go.Figure()

    # 10-90th percentile band
    fig.add_trace(
        go.Scatter(
            x=quantiles["hour"],
            y=quantiles["q90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=quantiles["hour"],
            y=quantiles["q10"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(139, 92, 246, 0.15)",
            line=dict(color="rgba(139, 92, 246, 0.4)", width=1),
            name="10-90th percentile",
            hovertemplate="<b>10-90th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>",
            customdata=quantiles[["hour_formatted", "q10", "q90"]].values,
        )
    )

    # 25-75th percentile band
    fig.add_trace(
        go.Scatter(
            x=quantiles["hour"],
            y=quantiles["q75"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=quantiles["hour"],
            y=quantiles["q25"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(139, 92, 246, 0.35)",
            line=dict(color="rgba(139, 92, 246, 0.7)", width=1.5),
            name="25-75th percentile",
            hovertemplate="<b>25-75th:</b> %{customdata[1]:.0f}-%{customdata[2]:.0f} mg/dl<extra></extra>",
            customdata=quantiles[["hour_formatted", "q25", "q75"]].values,
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=quantiles["hour"],
            y=quantiles["median"],
            mode="lines",
            line=dict(color="#a78bfa", width=3, shape="spline"),
            name="Median",
            hovertemplate="<b>Median:</b> %{y:.0f} mg/dl<extra></extra>",
        )
    )

    # Target lines
    for target, text, color in [
        (TARGET_LOW, "Target Low", "#10b981"),
        (TARGET_MILD_HIGH, "Target Mild High", "#10b981"),
        (TARGET_HIGH, "High", "#f59e0b"),
    ]:
        fig.add_hline(
            y=target,
            line_dash="dash",
            line_color=color,
            opacity=0.7,
            line_width=2,
            annotation_text=text,
            annotation_position="right",
            annotation=dict(font_size=10, font_color=color, font_family="JetBrains Mono, Consolas, monospace"),
        )

    fig.update_xaxes(
        title=None,
        tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
        ticktext=["0h", "3h", "6h", "9h", "12h", "15h", "18h", "21h"],
        showgrid=True,
        gridwidth=1,
        gridcolor="#334155",
    )
    fig.update_yaxes(
        title="Glucose [mg/dL]",
        range=[40, None],
        showgrid=True,
        gridwidth=1,
        gridcolor="#334155",
        title_font=dict(color="#94a3b8"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="JetBrains Mono, Consolas, monospace", size=11),
        margin=dict(t=60, r=30, b=40, l=60),
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(30, 41, 59, 0.8)"),
        title_font=dict(size=14, color="#94a3b8"),
    )

    return fig, window_text, valid_sgv_count, len(quantiles)


# ============================================================================
# UI update functions
# ============================================================================

@dataclass
class UIRefs:
    """References to all UI components for updates."""
    last_value_label: Any
    delta_label: Any
    delta_rate_label: Any
    updated_label: Any
    device_label: Any
    tir_value_label: Any
    tir_caption_label: Any
    avg_label: Any
    mmol_label: Any
    hba1c_label: Any
    dataset_label: Any
    hypo_label: Any
    tir_select: Any
    tir_chart: Any
    histogram_chart: Any
    recent_chart: Any
    pattern_chart: Any
    pattern_status: Any
    pattern_start_input: Any
    pattern_end_input: Any
    status_label: Any
    status_card: Any
    loading_spinner: Any


def update_hero(refs: UIRefs) -> None:
    """Update hero metrics with latest reading."""
    if STATE.last_value is None:
        # Status card will be hidden by this point, no need to update
        return

    last = STATE.last_value
    prev = STATE.previous_value or last

    last_ts = parse_entry_timestamp(last) or pd.Timestamp.utcnow().tz_localize("UTC")
    prev_ts = parse_entry_timestamp(prev) or last_ts
    last_local = last_ts.tz_convert("Europe/Madrid")
    
    delta = (last.get("sgv") or 0) - (prev.get("sgv") or 0)
    delta_class = "text-green-400" if delta > 0 else "text-red-400" if delta < 0 else "text-slate-400"
    trend_raw = last.get("direction") or "Flat"
    curr_dir = DIRECTIONS.get(trend_raw, "→")

    # Calculate delta per minute
    time_diff_minutes = (last_ts - prev_ts).total_seconds() / 60
    if time_diff_minutes > 0:
        delta_per_min = delta / time_diff_minutes
        refs.delta_rate_label.text = f"{delta_per_min:+.2f} mg/dL/min"
    else:
        refs.delta_rate_label.text = ""

    refs.last_value_label.text = f"{last.get('sgv', '--')} mg/dL {curr_dir}"
    refs.delta_label.text = f"{delta:+.0f} mg/dL"
    refs.delta_label.classes(replace="text-xl font-bold", add=delta_class)
    # Update timestamp with minutes since and formatted string
    minutes_since = int((pd.Timestamp.utcnow() - last_ts).total_seconds() / 60)
    refs.updated_label.text = f"{last_local.strftime('%d %b %Y · %H:%M')}\n{minutes_since} min ago"
    refs.device_label.text = last.get("device") or "—"


def update_recent_chart(refs: UIRefs) -> None:
    """Update the recent glucose chart."""
    refs.recent_chart.update_figure(build_recent_chart(STATE.df_recent))


def update_summary_cards(refs: UIRefs) -> None:
    """Update TIR and summary metrics."""
    if STATE.df_3months.empty:
        refs.tir_value_label.content = "<div class='text-3xl font-bold text-slate-100'>--</div>"
        refs.tir_caption_label.text = "Time in Range window"
        refs.avg_label.text = "--"
        refs.mmol_label.text = "--"
        refs.hba1c_label.text = "--"
        refs.dataset_label.text = "--"
        refs.hypo_label.text = "--"
        refs.tir_chart.update_figure(px.bar(title="No historical data"))
        return

    period_days = {
        "Last Day": 1,
        "Last Week": 7,
        "Last Month": 30,
        "Last 3 Months": 90,
    }
    selected_days = period_days.get(refs.tir_select.value, 7)

    cutoff = STATE.df_3months.date.max() - pd.to_timedelta(f"{selected_days} days")
    selected_df = STATE.df_3months.loc[STATE.df_3months.date > cutoff].copy()

    tir_counts = selected_df["cat_glucose"].value_counts(normalize=True)
    tir_core_pct = tir_counts.get(f"{TARGET_LOW}-{TARGET_MILD_HIGH}", 0) * 100
    tir_extended_pct = tir_counts.get(f"{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}", 0) * 100
    tir_in_range_pct = tir_core_pct + tir_extended_pct

    average_glucose = selected_df["sgv"].mean()
    hypo_events = int((selected_df["sgv"] < TARGET_LOW).sum())
    records_selected = len(selected_df)

    # Create colored TIR display: Total% (Core% + Extended%) - white total, colored values
    refs.tir_value_label.content = f"""
    <div class="text-3xl font-bold text-slate-100">
        {tir_in_range_pct:.0f}%
        <span class="text-base text-slate-100">
            (<span style="color: {LIGHT_GREEN}">{tir_core_pct:.0f}%</span> + <span style="color: {MILD_YELLOW}">{tir_extended_pct:.0f}%</span>)
        </span>
    </div>
    """
    
    # Add colored range numbers to caption
    refs.tir_caption_label.content = f"""
    <div class="text-xs text-slate-500 font-mono">
        {refs.tir_select.value} · 
        <span style="color: {LIGHT_GREEN}">{TARGET_LOW}-{TARGET_MILD_HIGH}</span> ·  
        <span style="color: {MILD_YELLOW}">{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}</span>
    </div>
    """

    if pd.isna(average_glucose):
        refs.avg_label.text = "--"
        refs.mmol_label.text = "--"
        refs.hba1c_label.text = "--"
    else:
        refs.avg_label.text = f"{average_glucose:.0f} mg/dL"
        refs.mmol_label.text = f"{average_glucose * 0.0555:.1f} mmol/L"
        hba1c_value = mean_glucose_to_hba1c(average_glucose)
        refs.hba1c_label.text = f"{hba1c_value:.1f}%"

    refs.dataset_label.text = f"{records_selected:,} records"
    refs.hypo_label.text = f"Hypo events: {hypo_events}"

    refs.tir_chart.update_figure(build_tir_chart(selected_df))
    refs.histogram_chart.update_figure(build_histogram_chart(selected_df))



def update_pattern_section(refs: UIRefs) -> None:
    """Update the glucose patterns chart."""
    if STATE.df_3months.empty:
        refs.pattern_status.text = "✗ Historical data unavailable"
        refs.pattern_chart.update_figure(go.Figure(layout=dict(title="No data")))
        return

    start_value = refs.pattern_start_input.value
    end_value = refs.pattern_end_input.value
    
    if not start_value or not end_value:
        refs.pattern_status.text = "⚠ Select a valid date range"
        refs.pattern_chart.update_figure(go.Figure(layout=dict(title="No data")))
        return

    # Parse dates as timezone-naive timestamps
    try:
        start_dt = pd.Timestamp(start_value)
        end_dt = pd.Timestamp(end_value)
        
        # Remove timezone if present
        if hasattr(start_dt, 'tz') and start_dt.tz is not None:
            start_dt = start_dt.tz_localize(None)
        if hasattr(end_dt, 'tz') and end_dt.tz is not None:
            end_dt = end_dt.tz_localize(None)
            
    except Exception as e:
        refs.pattern_status.text = f"✗ Invalid date: {e}"
        refs.pattern_chart.update_figure(go.Figure(layout=dict(title="No data")))
        return

    if start_dt > end_dt:
        refs.pattern_status.text = "✗ Start date must be ≤ end date"
        refs.pattern_chart.update_figure(go.Figure(layout=dict(title="No data")))
        return
    
    end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Filter using .dt accessor to handle timezone-aware Series
    try:
        df_copy = STATE.df_3months.copy()
        # Normalize dates to timezone-naive if needed
        if hasattr(df_copy['date'].dtype, 'tz') and df_copy['date'].dtype.tz is not None:
            df_copy['date'] = df_copy['date'].dt.tz_localize(None)
        
        mask = (df_copy['date'] >= start_dt) & (df_copy['date'] <= end_dt)
        df_filtered = df_copy[mask]
    except Exception as e:
        refs.pattern_status.text = f"✗ Filter error: {str(e)[:50]}"
        refs.pattern_chart.update_figure(go.Figure(layout=dict(title="No data")))
        return

    fig, window_text, valid_sgv, points = build_pattern_chart(df_filtered)
    refs.pattern_chart.update_figure(fig)
    refs.pattern_status.text = f"✓ {window_text} · {valid_sgv:,} SGVs · {points:,} points"


def update_dashboard(refs: UIRefs) -> None:
    """Full dashboard update."""
    update_hero(refs)
    update_recent_chart(refs)
    update_summary_cards(refs)
    update_pattern_section(refs)


async def periodic_refresh(refs: UIRefs) -> None:
    """Background task for minute-level hero refresh."""
    try:
        await refresh_recent_data(full_refresh=False)
        update_hero(refs)
        update_recent_chart(refs)
    except Exception as exc:
        # Status card is hidden by this point, just log the error
        print(f"✗ Refresh failed: {exc}")


# ============================================================================
# Main UI
# ============================================================================

@ui.page("/")
async def index_page() -> None:
    """Main dashboard page."""
    ui.query("body").classes("bg-slate-950 text-slate-100")
    ui.page_title("SugarBoard · NiceGUI Dashboard")
    
    # Add external CSS and JavaScript
    ui.add_head_html('<link rel="stylesheet" href="/static/style.css">')
    ui.add_head_html('<script src="/static/script.js"></script>')

    with ui.column().classes("w-full max-w-6xl mx-auto py-10 gap-6"):
        # Header with terminal aesthetic
        with ui.row().classes("items-center gap-3 mb-2"):
            ui.label("❯").classes("text-4xl font-bold text-violet-400")
            ui.label("SugarBoard").classes("text-3xl font-bold text-violet-300 tracking-tight")
            # Loading indicator
            loading_spinner = ui.spinner(size="lg", color="violet").classes("ml-auto")
            loading_spinner.set_visibility(False)
        ui.label("Real-time CGM monitoring // live refresh every 60s").classes("text-sm text-slate-500 font-mono")
        
        # Status banner - terminal-style output
        with ui.card().classes("w-full bg-black border-2 border-green-500 shadow-lg") as status_card:
            with ui.row().classes("items-center gap-2 px-2 py-1"):
                ui.label("❯").classes("text-green-400 text-base font-bold")
                with ui.element('div').classes("text-sm text-green-300 font-mono flex-1") as status_container:
                    status_label = ui.html('<span id="terminal-status"></span>')
        status_card.set_visibility(True)

        # Hero metrics
        with ui.row().classes("w-full gap-4 items-stretch"):
            with ui.card().classes("flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("LAST_READING").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                last_value_label = ui.label("--").classes("text-3xl font-bold text-slate-100")
            with ui.card().classes("flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("DELTA").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                delta_label = ui.label("--").classes("text-xl font-bold")
                delta_rate_label = ui.label("").classes("text-xs text-slate-500 font-mono")
            with ui.card().classes("flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("LAST_UPDATED").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                updated_label = ui.label("--").classes("text-sm font-semibold text-slate-300")
            with ui.card().classes("flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("DEVICE").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                device_label = ui.label("--").classes("text-sm font-semibold text-slate-300")

        # Summary cards
        with ui.row().classes("w-full gap-4 flex-wrap items-stretch"):
            with ui.card().classes("flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("TIME_IN_RANGE").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                tir_value_label = ui.html("<div class='text-3xl font-bold text-slate-100'>--</div>")
                tir_caption_label = ui.html("<div class='text-xs text-slate-500 font-mono'>Time in Range window</div>")
            with ui.card().classes("flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("AVG_GLUCOSE").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                avg_label = ui.label("--").classes("text-2xl font-bold text-slate-100")
                mmol_label = ui.label("--").classes("text-xs text-slate-500 font-mono")
            with ui.card().classes("flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("EST_HbA1c").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                hba1c_label = ui.label("--").classes("text-2xl font-bold text-slate-100")
            with ui.card().classes("flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"):
                ui.label("CURRENT_DATASET").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
                dataset_label = ui.label("--").classes("text-2xl font-bold text-slate-100")
                hypo_label = ui.label("--").classes("text-xs text-slate-500 font-mono")

        # Recent glucose - full width (most immediate info)
        with ui.card().classes("w-full bg-slate-900 border border-slate-700 shadow-lg"):
            ui.label("RECENT GLUCOSE · Last 4 Hours").classes("text-xs uppercase tracking-widest text-slate-400 font-bold mb-0")
            recent_chart = ui.plotly(create_placeholder_chart("Loading...", height=280)).classes("w-full -mt-2")

        # TIR Window Controls
        with ui.row().classes("w-full gap-4 items-center"):
            ui.label("⚙").classes("text-xl text-violet-400")
            tir_select = ui.select(
                ["Last Day", "Last Week", "Last Month", "Last 3 Months"],
                value="Last Week",
                label="TIR Window",
                on_change=lambda _: update_summary_cards(refs),
            ).classes("w-64 text-sm").props('dark outlined dense color="violet"')

        # TIR and Distribution charts
        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("flex-1 min-w-[300px] bg-slate-900 border border-slate-700 shadow-lg"):
                ui.label("TIME IN RANGE").classes("text-xs uppercase tracking-widest text-slate-400 font-bold mb-0")
                tir_chart = ui.plotly(create_placeholder_chart("Loading...", height=360)).classes("w-full -mt-2")
            with ui.card().classes("flex-1 min-w-[300px] bg-slate-900 border border-slate-700 shadow-lg"):
                ui.label("DISTRIBUTION").classes("text-xs uppercase tracking-widest text-slate-400 font-bold mb-0")
                histogram_chart = ui.plotly(create_placeholder_chart("Loading...", height=360)).classes("w-full -mt-2")

        # Patterns section - centered and constrained width
        with ui.row().classes("w-full justify-center"):
            with ui.card().classes("w-full max-w-4xl bg-slate-900 border border-slate-700 shadow-xl"):
                with ui.row().classes("items-center gap-3 mb-4"):
                    ui.label("📊").classes("text-2xl")
                    ui.label("DAILY PATTERNS").classes("text-sm uppercase tracking-widest text-slate-400 font-bold")
                    pattern_status = ui.label("Select a window to explore patterns.").classes(
                        "text-xs text-slate-500 font-mono ml-auto"
                    )
                
                with ui.row().classes("gap-3 mb-4 items-center"):
                    ui.label("⏱").classes("text-lg text-cyan-400")
                    
                    # FROM date with popup
                    with ui.input(label="FROM", placeholder="Select date").classes("w-40").props('dark outlined dense readonly color="cyan"') as pattern_start_input:
                        with ui.menu().props('no-parent-event') as start_menu:
                            with ui.date().bind_value(pattern_start_input).props('dark color="cyan"') as start_date:
                                with ui.row().classes('justify-end gap-2 mt-2'):
                                    ui.button('Close', on_click=start_menu.close).props('flat dense color="cyan"')
                        with pattern_start_input.add_slot('append'):
                            ui.icon('edit_calendar').on('click', start_menu.open).classes('cursor-pointer text-cyan-400')
                    
                    # TO date with popup
                    with ui.input(label="TO", placeholder="Select date").classes("w-40").props('dark outlined dense readonly color="violet"') as pattern_end_input:
                        with ui.menu().props('no-parent-event') as end_menu:
                            with ui.date().bind_value(pattern_end_input).props('dark color="violet"') as end_date:
                                with ui.row().classes('justify-end gap-2 mt-2'):
                                    ui.button('Close', on_click=end_menu.close).props('flat dense color="violet"')
                        with pattern_end_input.add_slot('append'):
                            ui.icon('edit_calendar').on('click', end_menu.open).classes('cursor-pointer text-violet-400')
                
                # Wire up the change handlers to the actual date pickers
                start_date.on('update:model-value', lambda _: update_pattern_section(refs))
                end_date.on('update:model-value', lambda _: update_pattern_section(refs))
                
                pattern_chart = ui.plotly(create_placeholder_chart("Select a date range above", height=400))

    # Build refs object
    refs = UIRefs(
        last_value_label=last_value_label,
        delta_label=delta_label,
        delta_rate_label=delta_rate_label,
        updated_label=updated_label,
        device_label=device_label,
        tir_value_label=tir_value_label,
        tir_caption_label=tir_caption_label,
        avg_label=avg_label,
        mmol_label=mmol_label,
        hba1c_label=hba1c_label,
        dataset_label=dataset_label,
        hypo_label=hypo_label,
        tir_select=tir_select,
        tir_chart=tir_chart,
        histogram_chart=histogram_chart,
        recent_chart=recent_chart,
        pattern_chart=pattern_chart,
        pattern_status=pattern_status,
        pattern_start_input=pattern_start_input,
        pattern_end_input=pattern_end_input,
        status_label=status_label,
        status_card=status_card,
        loading_spinner=loading_spinner,
    )

    # Start periodic refresh timer
    ui.timer(60.0, lambda: asyncio.create_task(periodic_refresh(refs)))

    # Helper function to type text with typewriter effect
    async def type_status(text: str, speed: int = 50):
        """Type text with typewriter effect by updating content progressively."""
        # Simple progressive text update (simulating typing)
        status_label.content = '<span id="terminal-status" class="terminal-cursor"></span>'
        await asyncio.sleep(0.1)
        
        for i in range(len(text) + 1):
            status_label.content = f'<span id="terminal-status" class="terminal-cursor">{text[:i]}</span>'
            await asyncio.sleep(speed / 1000)
        
        # Remove cursor after typing
        status_label.content = f'<span id="terminal-status">{text}</span>'
        await asyncio.sleep(0.1)

    # Load and fetch data in the background (after page renders)
    async def load_initial_data():
        # Show loading spinner
        refs.loading_spinner.set_visibility(True)
        
        # Initial message
        await type_status("$ init system...")
        await asyncio.sleep(0.3)
        
        # Load cached data
        load_recent_cache()

        # Fetch historical data with status updates
        await type_status("$ fetch --historical --days=90")
        refs.pattern_status.text = "⏳ Loading from cache/API..."
        
        await ensure_historical_data(refs)
        
        # Update status based on whether data was cached or fetched
        if not STATE.df_3months.empty:
            cache_path = Path(".cache/nicegui_historical.pkl")
            if cache_path.exists():
                cache_age = time.time() - cache_path.stat().st_mtime
                refs.pattern_status.text = f"✓ Cached ({int(cache_age/60)}m old) · {len(STATE.df_3months):,} records"
            else:
                refs.pattern_status.text = f"✓ Fetched from API · {len(STATE.df_3months):,} records"
        
        # Fetch recent data
        await type_status("$ fetch --recent --hours=4")
        await refresh_recent_data(full_refresh=STATE.df_recent.empty)

        # Initialize pattern date inputs
        if not STATE.df_3months.empty:
            data_min = STATE.df_3months.date.min().date()
            data_max = STATE.df_3months.date.max().date()
            refs.pattern_start_input.value = str(data_min)
            refs.pattern_end_input.value = str(data_max)

        # Initial render
        update_dashboard(refs)
        await type_status("$ system ready [OK] · refresh_interval=60s")
        
        # Hide loading spinner and transition to idle state
        refs.loading_spinner.set_visibility(False)
        await asyncio.sleep(1.5)
        
        # Fade to semi-transparent idle message with blinking cursor
        status_label.content = '<span id="terminal-status" class="terminal-cursor">$ Monitoring live data · Listening for device updates...</span>'
        refs.status_card.classes(remove="border-green-500", add="border-green-500/30")
        status_container.classes(remove="text-green-300", add="text-green-300/50")

    # Trigger background data load
    asyncio.create_task(load_initial_data())


@ui.page("/health")
def healthcheck() -> None:
    """Health check endpoint."""
    ui.label("ok")


if __name__ in {"__main__", "__mp_main__"}:
    # Add static files route for assets
    from nicegui import app

    app.add_static_files('/static', str(Path(__file__).parent / 'static'))

    port = int(os.environ.get("PORT", "8080"))
    reload_enabled = os.environ.get("NICEGUI_RELOAD", "false").lower() in {"1", "true", "yes"}

    ui.run(
        title="SugarBoard NiceGUI",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
        favicon='static/favicon.png',
    )
