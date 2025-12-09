from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from nicegui import app, ui

from src.data_services import ensure_timezone_aware
from src.ui.components import get_client_from_storage
from src.ui.dashboard import (
    STATE,
    UIRefs,
    build_dashboard_ui,
    ensure_historical_data,
    periodic_refresh,
    update_dashboard,
    update_pattern_section,
    update_recent_chart,
    update_summary_cards,
)
from src.ui.theme import DEFAULT_THEME, THEME_STORAGE_KEY, set_active_theme
from src.utils import strip_timezone


@ui.page("/")
async def index_page() -> None:
    """Main dashboard page."""
    storage = app.storage.user
    initial_theme = storage.get(THEME_STORAGE_KEY) or DEFAULT_THEME
    set_active_theme(STATE, initial_theme, storage)
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
        set_active_theme(STATE, value or DEFAULT_THEME, storage)
        if refs:
            update_recent_chart(refs)
            update_summary_cards(refs)
            update_pattern_section(refs)

    prefill_verification_pending = False
    has_saved_auth = bool(
        storage.get("ns_base_url")
        and (storage.get("ns_token") or storage.get("ns_api_secret"))
    )
    if has_saved_auth:
        prefill_verification_pending = True

    async def verify_connection_settings(connection_refs) -> None:
        nonlocal prefill_verification_pending
        client = get_client_from_storage()
        if client is None:
            prefill_verification_pending = False
            return
        # Show loading state on button
        if connection_refs.save_btn:
            connection_refs.save_btn.props("loading")

        try:
            await asyncio.to_thread(lambda: client.get_sgv(count=1))
        except Exception as exc:
            prefill_verification_pending = False
            # User-friendly error mapping
            err_msg = str(exc)
            if "HTTPSConnectionPool" in err_msg or "Failed to resolve" in err_msg:
                user_msg = "Could not reach server. Check URL."
            elif "401" in err_msg or "Unauthorized" in err_msg:
                user_msg = "Unauthorized. Check token/secret."
            elif "timeout" in err_msg.lower():
                user_msg = "Connection timed out."
            else:
                user_msg = f"Connection failed: {exc}"

            connection_refs.status_label.text = user_msg
            connection_refs.status_dot.set_visibility(True)
            connection_refs.status_dot.classes(
                remove="hidden connection-dot-active connection-dot-pending",
                add="connection-dot-error",
            )
        else:
            handle_connection_verified(connection_refs)
        finally:
            # Reset loading state
            if connection_refs.save_btn:
                connection_refs.save_btn.props(remove="loading")

    def handle_connection_verified(connection_refs) -> None:
        nonlocal prefill_verification_pending
        base = storage.get("ns_base_url") or "Nightscout"
        if connection_refs.callout:
            connection_refs.callout.set_visibility(False)
            connection_refs.callout = None
        suffix = (
            " (verified from saved credentials)" if prefill_verification_pending else ""
        )
        connection_refs.status_label.text = f"Connected · {base}{suffix}"
        connection_refs.expansion.value = False
        connection_refs.status_dot.classes(
            remove="hidden connection-dot-error connection-dot-pending",
            add="connection-dot-active",
        )
        connection_refs.status_dot.set_visibility(True)
        prefill_verification_pending = False

    def show_connection_pending(connection_refs, message: str) -> None:
        connection_refs.status_label.text = message
        connection_refs.status_dot.set_visibility(True)
        connection_refs.status_dot.classes(
            remove="hidden connection-dot-active connection-dot-error",
            add="connection-dot-pending",
        )

    def on_settings_saved() -> None:
        nonlocal prefill_verification_pending
        prefill_verification_pending = False
        base = storage.get("ns_base_url") or "Nightscout"
        show_connection_pending(connection_refs, f"Testing connection to {base}...")
        schedule_initial_load()
        asyncio.create_task(verify_connection_settings(connection_refs))

    with ui.column().classes("w-full max-w-6xl mx-auto py-10 gap-6"):
        refs, connection_refs = build_dashboard_ui(
            storage=storage,
            on_theme_toggle_callback=on_theme_toggle,
            on_settings_saved_callback=on_settings_saved,
        )

    # Helper function to type text with typewriter effect
    async def type_status(text: str, speed: int = 50):
        if not refs:
            return
        refs.status_label.content = (
            '<span id="terminal-status" class="terminal-cursor"></span>'
        )
        await asyncio.sleep(0.1)

        for i in range(len(text) + 1):
            refs.status_label.content = (
                f'<span id="terminal-status" class="terminal-cursor">{text[:i]}</span>'
            )
            await asyncio.sleep(speed / 1000)

        refs.status_label.content = f'<span id="terminal-status">{text}</span>'
        await asyncio.sleep(0.1)

    async def load_initial_data() -> None:
        if not refs:
            return

        nonlocal prefill_verification_pending

        async def try_load_fixture_data() -> bool:
            data_dir_env = os.environ.get("SUGARBOARD_TEST_DATA_DIR")
            if not data_dir_env:
                return False
            data_dir = Path(data_dir_env)
            recent_path = data_dir / "recent.json"
            history_path = data_dir / "history.json"
            if not recent_path.exists() or not history_path.exists():
                logging.warning(
                    "Fixture data directory %s is missing required files", data_dir
                )
                return False

            try:
                STATE.df_recent = ensure_timezone_aware(pd.read_json(recent_path))
                STATE.df_3months = ensure_timezone_aware(pd.read_json(history_path))
            except Exception as exc:  # pragma: no cover
                logging.error("Failed to load fixture data: %s", exc)
                return False

            STATE.fetched_at = time.time()

            if not STATE.df_recent.empty:
                ordered = STATE.df_recent.sort_values("date")

                def _row_to_entry(row: pd.Series) -> dict[str, Any]:
                    timestamp = strip_timezone(pd.Timestamp(row["date"]))
                    return {
                        "sgv": int(row["sgv"]),
                        "device": row.get("device", "TestDevice"),
                        "dateString": timestamp.isoformat(),
                    }

                last_row = ordered.iloc[-1]
                prev_row = ordered.iloc[-2] if len(ordered) > 1 else last_row
                STATE.last_value = _row_to_entry(last_row)
                STATE.previous_value = _row_to_entry(prev_row)

            update_dashboard(refs)
            handle_connection_verified(connection_refs)
            await type_status("$ system ready [fixtures]")
            refs.loading_spinner.set_visibility(False)
            await asyncio.sleep(0.5)
            refs.status_label.content = '<span id="terminal-status" class="terminal-cursor">$ Monitoring live data · Listening for device updates...</span>'
            refs.status_card.classes(
                remove="border-green-500", add="border-green-500/30"
            )
            refs.status_container.classes(
                remove="status-terminal-text", add="status-terminal-muted"
            )
            return True

        refs.loading_spinner.set_visibility(True)
        try:
            await type_status("$ init system...")
            await asyncio.sleep(0.3)

            if await try_load_fixture_data():
                return

            # Note: load_recent_cache logic assumes modules are imported.
            # It was imported from nicegui_app but needs to be imported here or handled in dashboard logic.
            # It is not imported in this file yet! I need to import load_recent_cache.
            # Wait, I imported load_initial_data logic but missed importing load_recent_cache at top.
            # I will assume I need to import it.
            # Checking imports... no load_recent_cache imported.
            from src.cache import load_recent_cache

            load_recent_cache(STATE)

            client = get_client_from_storage()
            if client is None:
                prefill_verification_pending = False
                refs.pattern_status.text = (
                    "✗ Configure Nightscout settings above to load data"
                )
                await type_status("$ waiting --nightscout-config")
                return

            await type_status("$ fetch --historical --days=90")
            refs.pattern_status.text = "⏳ Loading from cache/API..."
            await ensure_historical_data(client, refs)

            if not STATE.df_3months.empty:
                cache_path = Path(".cache/nicegui_historical.pkl")
                if cache_path.exists():
                    cache_age = time.time() - cache_path.stat().st_mtime
                    refs.pattern_status.text = f"✓ Cached ({int(cache_age / 60)}m old) · {len(STATE.df_3months):,} records"
                else:
                    refs.pattern_status.text = (
                        f"✓ Fetched from API · {len(STATE.df_3months):,} records"
                    )

            await type_status("$ fetch --recent --hours=4")

            client = get_client_from_storage()
            if client is None:
                refs.pattern_status.text = (
                    "✗ Nightscout settings removed; re-enter to continue."
                )
                await type_status("$ waiting --nightscout-config")
                return

            from src.ui.dashboard import refresh_recent_data

            await refresh_recent_data(client, full_refresh=True)

            if not STATE.df_3months.empty:
                data_min = STATE.df_3months.date.min().date()
                data_max = STATE.df_3months.date.max().date()
                refs.pattern_start_input.value = str(data_min)
                refs.pattern_end_input.value = str(data_max)

            update_dashboard(refs)
            handle_connection_verified(connection_refs)
            await type_status("$ system ready [OK] · refresh_interval=60s")

            refs.loading_spinner.set_visibility(False)
            await asyncio.sleep(1.5)

            refs.status_label.content = '<span id="terminal-status" class="terminal-cursor">$ Monitoring live data · Listening for device updates...</span>'
            refs.status_card.classes(
                remove="border-green-500", add="border-green-500/30"
            )
            refs.status_container.classes(
                remove="status-terminal-text", add="status-terminal-muted"
            )
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

    if has_saved_auth:
        base = storage.get("ns_base_url") or "Nightscout"
        show_connection_pending(
            connection_refs, f"Testing saved credentials for {base}..."
        )
        asyncio.create_task(verify_connection_settings(connection_refs))
        schedule_initial_load()

    # Start periodic refresh timer
    ui.timer(60.0, lambda: asyncio.create_task(periodic_refresh(refs)))
