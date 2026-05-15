from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

import pandas as pd
from nicegui import ui

from src.cache import load_historical_cache, save_historical_cache, save_recent_cache
from src.charts import (
    build_heatmap_chart,
    build_histogram_chart,
    build_pattern_chart,
    build_recent_chart,
    build_tir_chart,
    create_placeholder_chart,
)
from src.config import (
    LIGHT_GREEN,
    LINEPLOT_HOURS,
    MILD_YELLOW,
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
from src.hero import calculate_hero_metrics
from src.nightscout_client import NightscoutClient
from src.state import DataState
from src.ui.components import (
    get_client_from_storage,
    render_nightscout_settings_card,
    render_storage_secret_callout,
)
from src.utils import mean_glucose_to_gmi, strip_timezone

STATE = DataState()


@dataclass
class UIRefs:
    """References to all UI components for updates."""

    last_value_label: Any
    last_subtitle_label: Any
    delta_label: Any
    delta_rate_label: Any
    updated_label: Any
    updated_device_label: Any
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
    status_container: Any
    loading_spinner: Any
    streak_label: Any


async def ensure_historical_data(client: NightscoutClient, refs: Optional[Any] = None) -> None:
    """Load or fetch the 90-day historical dataset."""
    cached_df = load_historical_cache(source_url=client.base_url)
    if cached_df is not None:
        logging.info(f"✓ Loaded {len(cached_df)} historical records from cache")
        STATE.df_3months = cached_df
        return

    logging.info("⏳ Fetching 90 days of historical data from API (this may take a while)...")

    def progress_update(current: int, total: int):
        """Update progress in UI if refs provided."""
        if refs and hasattr(refs, "status_label"):
            percentage = int((current / total) * 100)
            # Terminal-style progress bar (instant update, no typing effect for smooth progress)
            bar_width = 20
            filled = int((current / total) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_text = f"$ fetching --chunks [{bar}] {current}/{total} ({percentage}%) "
            # Direct content update
            refs.status_label.content = (
                f'<span id="terminal-status" class="terminal-cursor">{progress_text}</span>'
            )

    df_3months = await fetch_historical_async(client, 90, progress_update)
    STATE.df_3months = df_3months
    save_historical_cache(df_3months, source_url=client.base_url)
    logging.info(f"✓ Fetched and cached {len(df_3months)} historical records")


async def refresh_recent_data(client: NightscoutClient, full_refresh: bool = False) -> None:
    """Refresh recent data, optionally doing a full fetch."""
    if full_refresh or STATE.df_recent.empty or STATE.last_value is None:
        last_value, previous_value, df_recent = await asyncio.to_thread(fetch_recent_data, client)
        df_recent = ensure_timezone_aware(df_recent)
        STATE.last_value = last_value
        STATE.previous_value = previous_value
        STATE.df_recent = df_recent
        STATE.fetched_at = time.time()
        save_recent_cache(STATE, source_url=client.base_url)
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
        pd.DataFrame,
        df_recent.loc[df_recent["date"] >= recent_cutoff].reset_index(drop=True),
    )

    STATE.last_value = latest_entry
    STATE.previous_value = previous_value
    STATE.df_recent = df_recent
    STATE.fetched_at = time.time()
    save_recent_cache(STATE, source_url=client.base_url)


def update_hero(refs: UIRefs) -> None:
    """Update hero metrics with latest reading."""
    if STATE.last_value is None:
        # Status card will be hidden by this point, no need to update
        return
    metrics = calculate_hero_metrics(
        STATE.last_value, STATE.previous_value or STATE.last_value, STATE.df_recent
    )
    if metrics is None:
        return

    refs.last_value_label.text = metrics.last_value_text
    if metrics.delta_value > 0:
        delta_class = "text-green-400"
    elif metrics.delta_value < 0:
        delta_class = "text-red-400"
    else:
        delta_class = "text-slate-400"
    refs.delta_label.text = metrics.delta_text
    refs.delta_label.classes(remove="text-green-400 text-red-400 text-slate-400", add=delta_class)
    refs.delta_rate_label.text = metrics.delta_rate_text
    refs.last_subtitle_label.text = metrics.last_subtitle_text
    refs.updated_label.text = metrics.updated_text
    refs.updated_device_label.text = metrics.device_text
    refs.streak_label.text = f"{metrics.streak_minutes} min"


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
        refs.tir_chart.update_figure(
            create_placeholder_chart("No historical data", theme=STATE.theme)
        )
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
        gmi_value = mean_glucose_to_gmi(average_glucose)
        refs.hba1c_label.text = f"{gmi_value:.1f}%"

    refs.dataset_label.text = f"{records_selected:,} records"
    refs.hypo_label.text = f"Hypo events: {hypo_events}"

    refs.tir_chart.update_figure(build_tir_chart(selected_df, STATE.theme))
    refs.histogram_chart.update_figure(build_histogram_chart(selected_df, STATE.theme))


def update_pattern_section(refs: UIRefs) -> None:
    """Update the glucose patterns chart."""
    if STATE.df_3months.empty:
        refs.pattern_status.text = "✗ Historical data unavailable"
        refs.pattern_chart.update_figure(
            create_placeholder_chart("No historical data", height=400, theme=STATE.theme)
        )
        refs.pattern_heatmap.update_figure(
            create_placeholder_chart("No historical data", height=400, theme=STATE.theme)
        )
        return

    try:
        refs.pattern_heatmap.update_figure(build_heatmap_chart(STATE.df_3months, STATE.theme))
    except Exception as e:
        refs.pattern_heatmap.update_figure(
            create_placeholder_chart("Heatmap unavailable", height=400, theme=STATE.theme)
        )
        logging.error(f"✗ Failed to build heatmap chart: {e}")

    start_value = refs.pattern_start_input.value
    end_value = refs.pattern_end_input.value

    if not start_value or not end_value:
        refs.pattern_status.text = "⚠ Select a valid date range"
        refs.pattern_chart.update_figure(
            create_placeholder_chart("Pick start/end dates", height=400, theme=STATE.theme)
        )
        return

    # Parse dates as timezone-naive timestamps
    try:
        start_dt = strip_timezone(pd.Timestamp(start_value))
        end_dt = strip_timezone(pd.Timestamp(end_value))

    except Exception as e:
        refs.pattern_status.text = f"✗ Invalid date: {e}"
        refs.pattern_chart.update_figure(
            create_placeholder_chart("Invalid dates", height=400, theme=STATE.theme)
        )
        return

    if start_dt > end_dt:
        refs.pattern_status.text = "✗ Start date must be ≤ end date"
        refs.pattern_chart.update_figure(
            create_placeholder_chart("Invalid range", height=400, theme=STATE.theme)
        )
        return

    end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Filter using .dt accessor to handle timezone-aware Series
    try:
        df_copy = STATE.df_3months.copy()
        # Normalize dates to timezone-naive if needed
        if hasattr(df_copy["date"].dtype, "tz") and df_copy["date"].dtype.tz is not None:
            df_copy["date"] = df_copy["date"].dt.tz_localize(None)

        mask = (df_copy["date"] >= start_dt) & (df_copy["date"] <= end_dt)
        df_filtered = df_copy[mask]
    except Exception as e:
        refs.pattern_status.text = f"✗ Filter error: {str(e)[:50]}"
        refs.pattern_chart.update_figure(
            create_placeholder_chart("Filter error", height=400, theme=STATE.theme)
        )
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
        logging.error(f"✗ Refresh failed: {exc}")


def build_dashboard_ui(
    storage: Any,
    on_theme_toggle_callback: Callable[[str], None],
    on_settings_saved_callback: Callable[[], None],
) -> tuple[UIRefs, Any]:
    """Build the dashboard UI and return references."""

    # Header with terminal aesthetic
    with ui.row().classes("items-center gap-3 mb-2 w-full flex-wrap"):
        ui.label("❯").classes("text-4xl font-bold text-violet-400 terminal-arrow")
        ui.label("SugarBoard").classes(
            "text-3xl font-bold text-violet-300 tracking-tight terminal-title"
        )
        with ui.row().classes("items-center gap-3 ml-auto shrink-0"):
            loading_spinner = ui.spinner(size="lg", color="violet")
            loading_spinner.set_visibility(False)
            ui.label("Theme").classes("text-xs uppercase tracking-widest text-slate-400")
            toggle = ui.switch(
                value=STATE.theme == "light",
                on_change=lambda event: on_theme_toggle_callback(
                    "light" if event.value else "dark"
                ),
            ).props('dense color="purple" keep-color')
            toggle.classes("theme-toggle-simple")
    ui.label("Real-time CGM monitoring // live refresh every 60s").classes(
        "text-sm text-slate-500 font-mono"
    )

    callout_card = render_storage_secret_callout()

    connection_refs = render_nightscout_settings_card(on_saved=on_settings_saved_callback)
    connection_refs.callout = callout_card

    # Status banner - terminal-style output
    with ui.card().classes(
        "status-card w-full bg-black border-2 border-green-500 shadow-lg"
    ) as status_card:
        with ui.row().classes("items-center gap-2 px-2 py-1"):
            ui.label("❯").classes("text-green-400 text-base font-bold")
            with ui.element("div").classes(
                "status-terminal-text text-sm font-mono flex-1"
            ) as status_container:
                status_label = ui.html('<span id="terminal-status"></span>')
    status_card.set_visibility(True)

    # Hero metrics
    with ui.row().classes("w-full gap-4 items-stretch"):
        with ui.card().classes(
            "flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("LAST_READING").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold"
            )
            last_value_label = ui.label("--").classes("text-3xl font-bold text-slate-100")
            last_subtitle_label = ui.label("--").classes("text-xs text-slate-400 font-mono")
        with ui.card().classes(
            "flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("DELTA").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
            delta_label = ui.label("--").classes("text-xl font-bold text-slate-100")
            delta_rate_label = ui.label("").classes("text-xs text-slate-500 font-mono")
        with ui.card().classes(
            "flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("LAST_UPDATED").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold"
            )
            updated_label = ui.label("--").classes("text-sm font-semibold text-slate-300")
            updated_device_label = ui.label("from device: --").classes(
                "text-xs text-slate-400 font-mono"
            )
        with ui.card().classes(
            "flex-1 bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("IN_RANGE_STREAK").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold"
            )
            streak_label = ui.label("0 min").classes("text-2xl font-bold text-slate-100")

    # Summary cards
    with ui.row().classes("w-full gap-4 flex-wrap items-stretch"):
        with ui.card().classes(
            "flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("TIME_IN_RANGE").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold"
            )
            tir_value_label = ui.html("<div class='text-3xl font-bold text-slate-100'>--</div>")
            tir_caption_label = ui.html(
                "<div class='text-xs text-slate-500 font-mono'>Time in Range window</div>"
            )
        with ui.card().classes(
            "flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("AVG_GLUCOSE").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold"
            )
            avg_label = ui.label("--").classes("text-2xl font-bold text-slate-100")
            mmol_label = ui.label("--").classes("text-xs text-slate-500 font-mono")
        with ui.card().classes(
            "flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("GMI").classes("text-xs uppercase tracking-widest text-slate-400 font-bold")
            hba1c_label = ui.label("--").classes("text-2xl font-bold text-slate-100")
        with ui.card().classes(
            "flex-1 min-w-[200px] bg-slate-900 border border-slate-700 shadow-lg flex flex-col"
        ):
            ui.label("CURRENT_DATASET").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold"
            )
            dataset_label = ui.label("--").classes("text-2xl font-bold text-slate-100")
            hypo_label = ui.label("--").classes("text-xs text-slate-500 font-mono")

    # Recent glucose - full width
    with ui.card().classes("w-full bg-slate-900 border border-slate-700 shadow-lg"):
        ui.label(f"RECENT GLUCOSE · Last {LINEPLOT_HOURS} Hours").classes(
            "text-xs uppercase tracking-widest text-slate-400 font-bold mb-0"
        )
        recent_chart = ui.plotly(
            create_placeholder_chart("Loading...", height=280, theme=STATE.theme)
        ).classes("w-full -mt-2")

    # TIR Window Controls
    with ui.row().classes("w-full gap-4 items-center"):
        ui.label("⚙").classes("text-xl text-violet-400")
        tir_select = (
            ui.select(
                ["Last Day", "Last Week", "Last Month", "Last 3 Months"],
                value="Last Week",
                label="TIR Window",
                # The callback will need the refs, so we might need a wrapper or late binding
                # For now setting to None, will bind after creation if possible or use a lambda that captures refs
                # BUT refs is not created yet.
                # Solution: The callback uses a lambda that calls update_summary_cards(refs)
                # We can't do that here easily because refs is defined after.
                # Use a specific handler defined outside or late bind.
            )
            .classes("tir-select w-64 text-sm")
            .props('dark outlined dense color="violet"')
        )

    # TIR and Distribution charts
    with ui.row().classes("w-full gap-4 flex-wrap"):
        with ui.card().classes(
            "flex-1 min-w-[300px] bg-slate-900 border border-slate-700 shadow-lg"
        ):
            ui.label("TIME IN RANGE").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold mb-0"
            )
            tir_chart = ui.plotly(
                create_placeholder_chart("Loading...", height=360, theme=STATE.theme)
            ).classes("w-full -mt-2")
        with ui.card().classes(
            "flex-1 min-w-[300px] bg-slate-900 border border-slate-700 shadow-lg"
        ):
            ui.label("DISTRIBUTION").classes(
                "text-xs uppercase tracking-widest text-slate-400 font-bold mb-0"
            )
            histogram_chart = ui.plotly(
                create_placeholder_chart("Loading...", height=360, theme=STATE.theme)
            ).classes("w-full -mt-2")

    # Patterns section
    with ui.row().classes("w-full"):
        with ui.card().classes(
            "pattern-card w-full bg-slate-900 border border-slate-700 shadow-xl"
        ):
            with ui.row().classes("items-center gap-3 mb-4"):
                ui.label("📊").classes("text-2xl")
                ui.label("DAILY PATTERNS").classes(
                    "text-sm uppercase tracking-widest text-slate-400 font-bold"
                )
                pattern_status = ui.label("Select a window to explore patterns.").classes(
                    "text-xs text-slate-500 font-mono ml-auto"
                )

            with ui.row().classes("gap-3 mb-4 items-center"):
                ui.label("⏱").classes("text-lg text-cyan-400")

                # FROM date with popup
                with (
                    ui.input(label="FROM", placeholder="Select date")
                    .classes("pattern-date-input w-40")
                    .props('dark outlined dense readonly color="cyan"') as pattern_start_input
                ):
                    with ui.menu().props("no-parent-event") as start_menu:
                        with (
                            ui.date()
                            .bind_value(pattern_start_input)
                            .props('dark color="cyan"') as start_date
                        ):
                            with ui.row().classes("justify-end gap-2 mt-2"):
                                ui.button("Close", on_click=start_menu.close).props(
                                    'flat dense color="cyan"'
                                )
                    with pattern_start_input.add_slot("append"):
                        ui.icon("edit_calendar").on("click", start_menu.open).classes(
                            "cursor-pointer text-cyan-400"
                        )

                # TO date with popup
                with (
                    ui.input(label="TO", placeholder="Select date")
                    .classes("pattern-date-input w-40")
                    .props('dark outlined dense readonly color="violet"') as pattern_end_input
                ):
                    with ui.menu().props("no-parent-event") as end_menu:
                        with (
                            ui.date()
                            .bind_value(pattern_end_input)
                            .props('dark color="violet"') as end_date
                        ):
                            with ui.row().classes("justify-end gap-2 mt-2"):
                                ui.button("Close", on_click=end_menu.close).props(
                                    'flat dense color="violet"'
                                )
                    with pattern_end_input.add_slot("append"):
                        ui.icon("edit_calendar").on("click", end_menu.open).classes(
                            "cursor-pointer text-violet-400"
                        )

            with ui.row().classes("w-full gap-4 flex-wrap"):
                pattern_chart = ui.plotly(
                    create_placeholder_chart(
                        "Select a date range above", height=400, theme=STATE.theme
                    )
                ).classes("flex-1 min-w-[320px]")
                pattern_heatmap = ui.plotly(
                    create_placeholder_chart(
                        "Heatmap will load automatically", height=400, theme=STATE.theme
                    )
                ).classes("flex-1 min-w-[320px]")

    refs = UIRefs(
        last_value_label=last_value_label,
        last_subtitle_label=last_subtitle_label,
        delta_label=delta_label,
        delta_rate_label=delta_rate_label,
        updated_label=updated_label,
        updated_device_label=updated_device_label,
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
        status_container=status_container,
        loading_spinner=loading_spinner,
        streak_label=streak_label,
    )

    # Wire up callbacks that need refs
    tir_select.on_value_change(lambda _: update_summary_cards(refs))
    start_date.on("update:model-value", lambda _: update_pattern_section(refs))
    end_date.on("update:model-value", lambda _: update_pattern_section(refs))

    return refs, connection_refs
