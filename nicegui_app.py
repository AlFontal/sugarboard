"""NiceGUI dashboard for SugarBoard.

Full-featured CGM dashboard with minute-level hero refresh, TIR analytics,
recent glucose chart, and daily pattern visualization.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast

import pandas as pd
from nicegui import app, ui

from src.cache import load_historical_cache, load_recent_cache, save_historical_cache, save_recent_cache
from src.charts import (
    build_heatmap_chart,
    build_histogram_chart,
    build_pattern_chart,
    build_recent_chart,
    build_tir_chart,
    create_placeholder_chart,
)
from src.config import (
    DEFAULT_NIGHTSCOUT_URL,
    DIRECTIONS,
    LIGHT_GREEN,
    LINEPLOT_HOURS,
    MILD_YELLOW,
    RECENT_REQUEST_TIMEOUT,
    STORAGE_SECRET,
    STORAGE_SECRET_FROM_ENV,
    TARGET_HIGH,
    TARGET_LOW,
    TARGET_MILD_HIGH,
)
from src.data_services import (
    ensure_timezone_aware,
    fetch_historical_async,
    fetch_latest_entry,
    fetch_recent_data,
    parse_entry_timestamp,
)
from src.nightscout_client import NightscoutClient
from src.state import DataState
from src.utils import mean_glucose_to_hba1c

STATE = DataState()
CACHED_CREDENTIAL_PLACEHOLDER = "[saved credential]"
DEFAULT_THEME = "dark"
THEME_STORAGE_KEY = "ui_theme"
DARK_BODY_CLASSES = "dark-theme text-slate-100"
LIGHT_BODY_CLASSES = "light-theme text-slate-900"
THEME_CLASS_RESET = f"{DARK_BODY_CLASSES} {LIGHT_BODY_CLASSES} bg-slate-950 bg-slate-50"


def apply_theme_classes(theme: str) -> None:
    """Apply the selected theme classes to the document body."""
    body = ui.query("body")
    body.classes(
        remove=THEME_CLASS_RESET,
        add=LIGHT_BODY_CLASSES if theme == "light" else DARK_BODY_CLASSES,
    )


def set_active_theme(theme: str, storage: Optional[dict[str, Any]] = None) -> None:
    """Persist and apply the active theme."""
    STATE.theme = theme if theme in {"dark", "light"} else DEFAULT_THEME
    if storage is not None:
        storage[THEME_STORAGE_KEY] = STATE.theme
    apply_theme_classes(STATE.theme)


def _sanitize_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def get_client_from_storage() -> Optional[NightscoutClient]:
    storage = app.storage.user
    base_url = storage.get("ns_base_url")
    if not base_url:
        return None
    token = storage.get("ns_token") or None
    api_secret = storage.get("ns_api_secret") or None
    return NightscoutClient(
        base_url=base_url,
        token=token,
        api_secret=api_secret,
        timeout=RECENT_REQUEST_TIMEOUT,
    )


def render_nightscout_settings_card(
    on_saved: Optional[Callable[[], None]] = None, on_verify: Optional[Callable[[], None]] = None
) -> NightscoutRefs:
    storage = app.storage.user
    stored_base = storage.get("ns_base_url") or ""
    stored_token = storage.get("ns_token") or ""
    stored_secret = storage.get("ns_api_secret") or ""
    base_prefill = stored_base or DEFAULT_NIGHTSCOUT_URL or ""
    token_prefilled = bool(stored_token)
    secret_prefilled = bool(stored_secret)

    expansion = ui.expansion(value=True).classes("w-full bg-transparent text-slate-100 ns-expansion")
    with expansion.add_slot("header"):
        with ui.row().classes("items-center justify-between w-full gap-3 pr-2"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("link").classes("text-cyan-300")
                ui.label("Nightscout Connection").classes("text-xs uppercase tracking-[0.5em] text-cyan-200")
                status_dot = ui.icon("fiber_manual_record").classes("connection-dot hidden ml-2")

    with expansion:
        with ui.column().classes(
            "ns-settings-card w-full bg-[#0d1629]/95 border border-cyan-900/40 shadow-2xl shadow-black/40 "
            "rounded-2xl px-6 py-5 text-slate-100 backdrop-blur"
        ):
            status_label = ui.label(
                f"Current site: {stored_base or 'Not configured'}"
            ).classes("text-xs text-slate-400 mb-3")

            base_input = ui.input(
                label="Base URL",
                placeholder="https://mysite.herokuapp.com",
                value=base_prefill,
            ).props('type=url dark outlined dense color="cyan" label-color="cyan" input-class="night-input-text"').classes(
                "night-input night-input-cyan w-full mb-3"
            )

            token_input = ui.input(
                label="Read token (preferred)",
                placeholder="Optional",
                password=True,
                password_toggle_button=True,
                value=CACHED_CREDENTIAL_PLACEHOLDER if token_prefilled else "",
            ).props('dark outlined dense color="violet" label-color="violet" input-class="night-input-text"').classes(
                "night-input night-input-violet w-full mb-2"
            )

            secret_input = ui.input(
                label="API secret (fallback)",
                placeholder="Optional",
                password=True,
                password_toggle_button=True,
                value=CACHED_CREDENTIAL_PLACEHOLDER if secret_prefilled else "",
            ).props('dark outlined dense color="violet" label-color="violet" input-class="night-input-text"').classes(
                "night-input night-input-violet w-full mb-4"
            )

            ui.label(
                "Use a Nightscout read-only token from Settings → API whenever possible. "
                "Only fall back to the API secret if tokens are disabled; we hash it locally and send it via the api-secret header."
            ).classes("text-xs text-slate-400 mb-5 leading-relaxed")

        def save_settings() -> None:
            base = _sanitize_base_url(base_input.value or "")
            raw_token = (token_input.value or "").strip()
            raw_secret = (secret_input.value or "").strip()

            def resolve_secret(raw_value: str, stored_value: str) -> tuple[str, bool]:
                if raw_value:
                    if raw_value == CACHED_CREDENTIAL_PLACEHOLDER and stored_value:
                        return stored_value, True
                    return raw_value, True
                return "", False

            token, has_token = resolve_secret(raw_token, stored_token)
            secret, has_secret = resolve_secret(raw_secret, stored_secret)

            if not base:
                ui.notify("Nightscout base URL is required.", type="warning")
                return
            if not has_token and not has_secret:
                ui.notify("Provide a read token or API secret.", type="warning")
                return

            storage["ns_base_url"] = base
            if token:
                storage["ns_token"] = token
            else:
                storage.pop("ns_token", None)
            if secret:
                storage["ns_api_secret"] = secret
            else:
                storage.pop("ns_api_secret", None)

            token_input.value = ""
            secret_input.value = ""
            status_label.text = f"Current site: {base}"
            ui.notify("Nightscout settings saved.", type="positive")

            if on_saved:
                on_saved()
            if on_verify:
                on_verify()

        ui.button("Save Nightscout settings", on_click=save_settings).classes(
            "bg-gradient-to-r from-cyan-500 to-blue-500 text-slate-50 font-mono uppercase tracking-[0.4em] "
            "py-2 px-4 rounded-xl shadow-lg shadow-cyan-900/40 hover:opacity-90 transition"
        )

    return NightscoutRefs(callout=None, expansion=expansion, status_label=status_label, status_dot=status_dot)


def render_storage_secret_callout() -> Optional[Any]:
    if STORAGE_SECRET_FROM_ENV:
        return None
    with ui.card().classes(
        "w-full bg-amber-100 text-amber-950 border border-amber-500 shadow-lg "
        "shadow-amber-900/20 ring-1 ring-amber-600 font-mono px-4 py-3"
    ) as callout_card:
        with ui.row().classes("items-start gap-3"):
            ui.icon("warning_amber").classes("text-amber-600 text-3xl")
            with ui.column().classes("gap-1"):
                ui.label("Heads up: ephemeral storage secret").classes(
                    "font-semibold text-sm uppercase tracking-[0.3em]"
                )
                ui.label(
                    "Define STORAGE_SECRET in your container or shell to keep user Nightscout settings after a restart."
                ).classes("text-xs leading-relaxed text-amber-900")
    return callout_card

async def ensure_historical_data(client: NightscoutClient, refs: Optional[Any] = None) -> None:
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
    
    df_3months = await fetch_historical_async(client, 90, progress_update)
    STATE.df_3months = df_3months
    save_historical_cache(df_3months)
    print(f"✓ Fetched and cached {len(df_3months)} historical records")


async def refresh_recent_data(client: NightscoutClient, full_refresh: bool = False) -> None:
    """Refresh recent data, optionally doing a full fetch."""
    if full_refresh or STATE.df_recent.empty or STATE.last_value is None:
        last_value, previous_value, df_recent = await asyncio.to_thread(fetch_recent_data, client)
        df_recent = ensure_timezone_aware(df_recent)
        STATE.last_value = last_value
        STATE.previous_value = previous_value
        STATE.df_recent = df_recent
        STATE.fetched_at = time.time()
        save_recent_cache(STATE)
        return

    latest_entry = await asyncio.to_thread(fetch_latest_entry, client)
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
    save_recent_cache(STATE)


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
    pattern_heatmap: Any
    pattern_status: Any
    pattern_start_input: Any
    pattern_end_input: Any
    status_label: Any
    status_card: Any
    loading_spinner: Any


@dataclass
class NightscoutRefs:
    callout: Optional[Any]
    expansion: Any
    status_label: Any
    status_dot: Any


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
    refs.recent_chart.update_figure(build_recent_chart(STATE.df_recent, STATE.theme))


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
        refs.tir_chart.update_figure(create_placeholder_chart("No historical data", theme=STATE.theme))
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

    refs.tir_chart.update_figure(build_tir_chart(selected_df, STATE.theme))
    refs.histogram_chart.update_figure(build_histogram_chart(selected_df, STATE.theme))



def update_pattern_section(refs: UIRefs) -> None:
    """Update the glucose patterns chart."""
    if STATE.df_3months.empty:
        refs.pattern_status.text = "✗ Historical data unavailable"
        refs.pattern_chart.update_figure(create_placeholder_chart("No historical data", height=400, theme=STATE.theme))
        refs.pattern_heatmap.update_figure(create_placeholder_chart("No historical data", height=400, theme=STATE.theme))
        return

    try:
        refs.pattern_heatmap.update_figure(build_heatmap_chart(STATE.df_3months, STATE.theme))
    except Exception as e:
        refs.pattern_heatmap.update_figure(create_placeholder_chart("Heatmap unavailable", height=400, theme=STATE.theme))
        logging.error(f"✗ Failed to build heatmap chart: {e}")

    start_value = refs.pattern_start_input.value
    end_value = refs.pattern_end_input.value
    
    if not start_value or not end_value:
        refs.pattern_status.text = "⚠ Select a valid date range"
        refs.pattern_chart.update_figure(create_placeholder_chart("Pick start/end dates", height=400, theme=STATE.theme))
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
        refs.pattern_chart.update_figure(create_placeholder_chart("Invalid dates", height=400, theme=STATE.theme))
        return

    if start_dt > end_dt:
        refs.pattern_status.text = "✗ Start date must be ≤ end date"
        refs.pattern_chart.update_figure(create_placeholder_chart("Invalid range", height=400, theme=STATE.theme))
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
        refs.pattern_chart.update_figure(create_placeholder_chart("Filter error", height=400, theme=STATE.theme))
        return

    fig, window_text, valid_sgv, points = build_pattern_chart(df_filtered, STATE.theme)
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
        client = get_client_from_storage()
        if client is None:
            return
        await refresh_recent_data(client, full_refresh=False)
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
    storage = app.storage.user
    initial_theme = storage.get(THEME_STORAGE_KEY) or DEFAULT_THEME
    set_active_theme(initial_theme, storage)
    ui.page_title("SugarBoard · NiceGUI Dashboard")
    
    # Add external CSS and JavaScript
    ui.add_head_html('<link rel="stylesheet" href="/static/style.css">')
    ui.add_head_html('<script src="/static/script.js"></script>')

    load_task: Optional[asyncio.Task] = None
    refs: Optional[UIRefs] = None

    def schedule_initial_load() -> None:
        nonlocal load_task
        if load_task and not load_task.done():
            return
        load_task = asyncio.create_task(load_initial_data())

    def on_theme_toggle(value: str) -> None:
        set_active_theme(value or DEFAULT_THEME, storage)
        if refs:
            update_recent_chart(refs)
            update_summary_cards(refs)
            update_pattern_section(refs)

    with ui.column().classes("w-full max-w-6xl mx-auto py-10 gap-6"):
        # Header with terminal aesthetic
        with ui.row().classes("items-center gap-3 mb-2 w-full flex-wrap"):
            ui.label("❯").classes("text-4xl font-bold text-violet-400 terminal-arrow")
            ui.label("SugarBoard").classes("text-3xl font-bold text-violet-300 tracking-tight terminal-title")
            with ui.row().classes("items-center gap-3 ml-auto shrink-0"):
                loading_spinner = ui.spinner(size="lg", color="violet")
                loading_spinner.set_visibility(False)
                ui.label("Theme").classes("text-xs uppercase tracking-widest text-slate-400")
                toggle = ui.switch(
                    value=STATE.theme == "light",
                    on_change=lambda event: on_theme_toggle("light" if event.value else "dark"),
                ).props('dense color="purple" keep-color')
                toggle.classes("theme-toggle-simple")
        ui.label("Real-time CGM monitoring // live refresh every 60s").classes("text-sm text-slate-500 font-mono")

        callout_card = render_storage_secret_callout()
        has_saved_auth = bool(storage.get("ns_base_url") and (storage.get("ns_token") or storage.get("ns_api_secret")))
        prefill_verification_pending = has_saved_auth

        def handle_connection_verified() -> None:
            nonlocal prefill_verification_pending
            base = storage.get("ns_base_url") or "Nightscout"
            if connection_refs.callout:
                connection_refs.callout.set_visibility(False)
                connection_refs.callout = None
            suffix = " (verified from saved credentials)" if prefill_verification_pending else ""
            connection_refs.status_label.text = f"Connected · {base}{suffix}"
            connection_refs.expansion.value = False
            connection_refs.status_dot.classes(
                remove="hidden connection-dot-error connection-dot-pending",
                add="connection-dot-active",
            )
            connection_refs.status_dot.set_visibility(True)
            prefill_verification_pending = False

        async def verify_connection_settings() -> None:
            nonlocal prefill_verification_pending
            client = get_client_from_storage()
            if client is None:
                prefill_verification_pending = False
                return
            try:
                await asyncio.to_thread(lambda: client.get_sgv(count=1))
            except Exception as exc:
                prefill_verification_pending = False
                connection_refs.status_label.text = f"Connection failed: {exc}"
                connection_refs.status_dot.set_visibility(True)
                connection_refs.status_dot.classes(
                    remove="hidden connection-dot-active connection-dot-pending",
                    add="connection-dot-error",
                )
            else:
                handle_connection_verified()

        def on_settings_saved() -> None:
            nonlocal prefill_verification_pending
            prefill_verification_pending = False
            base = storage.get("ns_base_url") or "Nightscout"
            show_connection_pending(f"Testing connection to {base}...")
            schedule_initial_load()
            asyncio.create_task(verify_connection_settings())

        connection_refs = render_nightscout_settings_card(on_saved=on_settings_saved)
        connection_refs.callout = callout_card

        def show_connection_pending(message: str) -> None:
            connection_refs.status_label.text = message
            connection_refs.status_dot.set_visibility(True)
            connection_refs.status_dot.classes(
                remove="hidden connection-dot-active connection-dot-error",
                add="connection-dot-pending",
            )

        if has_saved_auth:
            base = storage.get("ns_base_url") or "Nightscout"
            show_connection_pending(f"Testing saved credentials for {base}...")
        
        # Status banner - terminal-style output
        with ui.card().classes("status-card w-full bg-black border-2 border-green-500 shadow-lg") as status_card:
            with ui.row().classes("items-center gap-2 px-2 py-1"):
                ui.label("❯").classes("text-green-400 text-base font-bold")
                with ui.element('div').classes("status-terminal-text text-sm font-mono flex-1") as status_container:
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
            recent_chart = ui.plotly(create_placeholder_chart("Loading...", height=280, theme=STATE.theme)).classes("w-full -mt-2")

        # TIR Window Controls
        with ui.row().classes("w-full gap-4 items-center"):
            ui.label("⚙").classes("text-xl text-violet-400")
            tir_select = ui.select(
                ["Last Day", "Last Week", "Last Month", "Last 3 Months"],
                value="Last Week",
                label="TIR Window",
                on_change=lambda _: update_summary_cards(refs),
            ).classes("tir-select w-64 text-sm").props('dark outlined dense color="violet"')

        # TIR and Distribution charts
        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("flex-1 min-w-[300px] bg-slate-900 border border-slate-700 shadow-lg"):
                ui.label("TIME IN RANGE").classes("text-xs uppercase tracking-widest text-slate-400 font-bold mb-0")
                tir_chart = ui.plotly(create_placeholder_chart("Loading...", height=360, theme=STATE.theme)).classes("w-full -mt-2")
            with ui.card().classes("flex-1 min-w-[300px] bg-slate-900 border border-slate-700 shadow-lg"):
                ui.label("DISTRIBUTION").classes("text-xs uppercase tracking-widest text-slate-400 font-bold mb-0")
                histogram_chart = ui.plotly(create_placeholder_chart("Loading...", height=360, theme=STATE.theme)).classes("w-full -mt-2")

        # Patterns section - full width to match other rows
        with ui.row().classes("w-full"):
            with ui.card().classes("pattern-card w-full bg-slate-900 border border-slate-700 shadow-xl"):
                with ui.row().classes("items-center gap-3 mb-4"):
                    ui.label("📊").classes("text-2xl")
                    ui.label("DAILY PATTERNS").classes("text-sm uppercase tracking-widest text-slate-400 font-bold")
                    pattern_status = ui.label("Select a window to explore patterns.").classes(
                        "text-xs text-slate-500 font-mono ml-auto"
                    )
                
                with ui.row().classes("gap-3 mb-4 items-center"):
                    ui.label("⏱").classes("text-lg text-cyan-400")
                    
                    # FROM date with popup
                    with ui.input(label="FROM", placeholder="Select date").classes("pattern-date-input w-40").props('dark outlined dense readonly color="cyan"') as pattern_start_input:
                        with ui.menu().props('no-parent-event') as start_menu:
                            with ui.date().bind_value(pattern_start_input).props('dark color="cyan"') as start_date:
                                with ui.row().classes('justify-end gap-2 mt-2'):
                                    ui.button('Close', on_click=start_menu.close).props('flat dense color="cyan"')
                        with pattern_start_input.add_slot('append'):
                            ui.icon('edit_calendar').on('click', start_menu.open).classes('cursor-pointer text-cyan-400')
                    
                    # TO date with popup
                    with ui.input(label="TO", placeholder="Select date").classes("pattern-date-input w-40").props('dark outlined dense readonly color="violet"') as pattern_end_input:
                        with ui.menu().props('no-parent-event') as end_menu:
                            with ui.date().bind_value(pattern_end_input).props('dark color="violet"') as end_date:
                                with ui.row().classes('justify-end gap-2 mt-2'):
                                    ui.button('Close', on_click=end_menu.close).props('flat dense color="violet"')
                        with pattern_end_input.add_slot('append'):
                            ui.icon('edit_calendar').on('click', end_menu.open).classes('cursor-pointer text-violet-400')
                
                # Wire up the change handlers to the actual date pickers
                start_date.on('update:model-value', lambda _: update_pattern_section(refs))
                end_date.on('update:model-value', lambda _: update_pattern_section(refs))

                with ui.row().classes("w-full gap-4 flex-wrap"):
                    pattern_chart = (
                        ui.plotly(create_placeholder_chart("Select a date range above", height=400, theme=STATE.theme))
                        .classes("flex-1 min-w-[320px]")
                    )
                    pattern_heatmap = (
                        ui.plotly(create_placeholder_chart("Heatmap will load automatically", height=400, theme=STATE.theme))
                        .classes("flex-1 min-w-[320px]")
                    )

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
        pattern_heatmap=pattern_heatmap,
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
        nonlocal prefill_verification_pending
        refs.loading_spinner.set_visibility(True)
        try:
            await type_status("$ init system...")
            await asyncio.sleep(0.3)

            load_recent_cache(STATE)

            client = get_client_from_storage()
            if client is None:
                prefill_verification_pending = False
                refs.pattern_status.text = "✗ Configure Nightscout settings above to load data"
                await type_status("$ waiting --nightscout-config")
                return

            await type_status("$ fetch --historical --days=90")
            refs.pattern_status.text = "⏳ Loading from cache/API..."
            await ensure_historical_data(client, refs)

            if not STATE.df_3months.empty:
                cache_path = Path(".cache/nicegui_historical.pkl")
                if cache_path.exists():
                    cache_age = time.time() - cache_path.stat().st_mtime
                    refs.pattern_status.text = f"✓ Cached ({int(cache_age/60)}m old) · {len(STATE.df_3months):,} records"
                else:
                    refs.pattern_status.text = f"✓ Fetched from API · {len(STATE.df_3months):,} records"

            await type_status("$ fetch --recent --hours=4")

            client = get_client_from_storage()
            if client is None:
                refs.pattern_status.text = "✗ Nightscout settings removed; re-enter to continue."
                await type_status("$ waiting --nightscout-config")
                return

            await refresh_recent_data(client, full_refresh=True)

            if not STATE.df_3months.empty:
                data_min = STATE.df_3months.date.min().date()
                data_max = STATE.df_3months.date.max().date()
                refs.pattern_start_input.value = str(data_min)
                refs.pattern_end_input.value = str(data_max)

            update_dashboard(refs)
            handle_connection_verified()
            await type_status("$ system ready [OK] · refresh_interval=60s")

            refs.loading_spinner.set_visibility(False)
            await asyncio.sleep(1.5)

            status_label.content = '<span id="terminal-status" class="terminal-cursor">$ Monitoring live data · Listening for device updates...</span>'
            refs.status_card.classes(remove="border-green-500", add="border-green-500/30")
            status_container.classes(remove="status-terminal-text", add="status-terminal-muted")
        except Exception as exc:
            prefill_verification_pending = False
            refs.pattern_status.text = f"✗ Error: {exc}"
            await type_status(f"$ error -- {exc}")
            connection_refs.status_label.text = f"Connection failed: {exc}"
            connection_refs.status_dot.set_visibility(True)
            connection_refs.status_dot.classes(
                remove="hidden connection-dot-active connection-dot-pending",
                add="connection-dot-error",
            )
        finally:
            refs.loading_spinner.set_visibility(False)

    # Trigger background data load
    schedule_initial_load()


@ui.page("/health")
def healthcheck() -> None:
    """Health check endpoint."""
    ui.label("ok")


if __name__ in {"__main__", "__mp_main__"}:
    # Add static files route for assets
    app.add_static_files('/static', str(Path(__file__).parent / 'static'))

    port = int(os.environ.get("PORT", "8080"))
    reload_enabled = os.environ.get("NICEGUI_RELOAD", "false").lower() in {"1", "true", "yes"}

    ui.run(
        title="SugarBoard NiceGUI",
        host="0.0.0.0",
        port=port,
        reload=True,
        storage_secret=STORAGE_SECRET,
        favicon='static/favicon.png',
    )
