from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import (
    LIGHT_GREEN,
    LIGHT_RED,
    MILD_YELLOW,
    STRONG_RED,
    TARGET_HIGH,
    TARGET_LOW,
    TARGET_MILD_HIGH,
    TARGET_SEVERE_HIGH,
    TARGET_SEVERE_LOW,
)


def create_placeholder_chart(title: str = "Loading...", height: int = 360) -> go.Figure:
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
    if df.empty:
        return create_placeholder_chart("No data yet")

    df_plot = df.copy()
    df_plot["date"] = df_plot["date"].dt.tz_convert("Europe/Madrid").astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["date"],
            y=df_plot["sgv"],
            mode="lines",
            line=dict(color="#8b5cf6", width=3),
            name="Glucose",
            hovertemplate="<b>%{y:.0f} mg/dL</b><br>%{x}<extra></extra>",
        )
    )

    if not df_plot.empty:
        last_row = df_plot.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_row["date"]],
                y=[last_row["sgv"]],
                mode="markers",
                marker=dict(size=10, color="#8b5cf6", line=dict(width=2, color="#0f172a")),
                name="Current",
                hovertemplate="<b>Current: %{y:.0f} mg/dL</b><extra></extra>",
            )
        )

    fig.add_hrect(
        y0=TARGET_LOW,
        y1=TARGET_MILD_HIGH,
        fillcolor=LIGHT_GREEN,
        opacity=0.1,
        layer="below",
        line_width=0,
    )
    fig.add_hrect(
        y0=TARGET_MILD_HIGH,
        y1=TARGET_HIGH,
        fillcolor=MILD_YELLOW,
        opacity=0.1,
        layer="below",
        line_width=0,
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


def build_tir_chart(selected_df: pd.DataFrame) -> go.Figure:
    if selected_df.empty:
        return create_placeholder_chart("No data yet")

    bg_categories = [
        f"<{TARGET_SEVERE_LOW}",
        f"{TARGET_SEVERE_LOW}-{TARGET_LOW - 1}",
        f"{TARGET_LOW}-{TARGET_MILD_HIGH}",
        f"{TARGET_MILD_HIGH + 1}-{TARGET_HIGH}",
        f"{TARGET_HIGH + 1}-{TARGET_SEVERE_HIGH}",
        f">{TARGET_SEVERE_HIGH}",
    ]
    tir_counts = selected_df["cat_glucose"].value_counts(normalize=True)
    tir_data = pd.DataFrame(
        {
            "cat_glucose": bg_categories,
            "value": [tir_counts.get(cat, 0) for cat in bg_categories],
            "percent_label": [f"{(tir_counts.get(cat, 0) * 100):.0f}%" for cat in bg_categories],
        }
    )
    value_max = max(tir_data["value"].max(), 0.0001)

    fig = px.bar(
        tir_data,
        x="cat_glucose",
        y="value",
        color="cat_glucose",
        text="percent_label",
        category_orders={"cat_glucose": bg_categories},
        color_discrete_map={
            bg_categories[0]: STRONG_RED,
            bg_categories[1]: LIGHT_RED,
            bg_categories[2]: LIGHT_GREEN,
            bg_categories[3]: MILD_YELLOW,
            bg_categories[4]: LIGHT_RED,
            bg_categories[5]: STRONG_RED,
        },
    )

    fig.update_traces(textposition="outside", cliponaxis=False, width=0.8, textfont=dict(size=12, color="#e2e8f0"))
    fig.update_yaxes(
        tickformat=".0%",
        title=None,
        range=[0, value_max * 1.15],
        gridcolor="#334155",
        title_font=dict(color="#94a3b8"),
    )
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
    if selected_df.empty:
        return create_placeholder_chart("No data yet")

    df_hist = selected_df.copy()
    df_hist["sgv_capped"] = df_hist["sgv"].clip(upper=300)
    bins = list(range(0, 305, 5))
    hist_data, bin_edges = np.histogram(df_hist["sgv_capped"], bins=bins)

    total_count = hist_data.sum()
    hist_pct = (hist_data / total_count * 100) if total_count > 0 else hist_data
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(len(bin_edges) - 1)]

    def get_color_for_bin(bin_center):
        if bin_center < TARGET_SEVERE_LOW:
            return STRONG_RED
        if bin_center < TARGET_LOW:
            return LIGHT_RED
        if bin_center <= TARGET_MILD_HIGH:
            return LIGHT_GREEN
        if bin_center <= TARGET_HIGH:
            return MILD_YELLOW
        if bin_center <= TARGET_SEVERE_HIGH:
            return LIGHT_RED
        return STRONG_RED

    colors = [get_color_for_bin(center) for center in bin_centers]

    fig = go.Figure(
        data=[
            go.Bar(
                x=bin_centers,
                y=hist_pct,
                marker_color=colors,
                width=4.5,
                hovertemplate="<b>%{customdata}</b><br>%{y:.1f}%<extra></extra>",
                customdata=bin_labels,
            )
        ]
    )

    for target, color in [
        (TARGET_LOW, "#10b981"),
        (TARGET_MILD_HIGH, "#10b981"),
        (TARGET_HIGH, "#f59e0b"),
    ]:
        fig.add_vline(x=target, line_dash="dash", line_color=color, opacity=0.7, line_width=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="JetBrains Mono, Consolas, monospace", size=11),
        showlegend=False,
        height=360,
        margin=dict(t=50, r=20, b=70, l=40),
        title_font=dict(size=14, color="#94a3b8"),
        xaxis=dict(title="Glucose [mg/dL]", gridcolor="#334155", title_font=dict(color="#94a3b8"), range=[0, 305]),
        yaxis=dict(title="Percentage", gridcolor="#334155", title_font=dict(color="#94a3b8"), ticksuffix="%"),
    )
    return fig


def build_pattern_chart(sel_df: pd.DataFrame) -> Tuple[go.Figure, str, int, int]:
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
    fig.add_trace(
        go.Scatter(x=quantiles["hour"], y=quantiles["q90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
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
    fig.add_trace(
        go.Scatter(x=quantiles["hour"], y=quantiles["q75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
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


__all__ = [
    "build_histogram_chart",
    "build_pattern_chart",
    "build_recent_chart",
    "build_tir_chart",
    "create_placeholder_chart",
]
