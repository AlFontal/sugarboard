from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from nicegui import app, ui

from src.config import DEFAULT_NIGHTSCOUT_URL, STORAGE_SECRET_FROM_ENV
from src.nightscout_client import NightscoutClient


@dataclass
class NightscoutRefs:
    callout: Optional[Any]
    expansion: Any
    status_label: Any
    status_dot: Any
    save_btn: Any


CACHED_CREDENTIAL_PLACEHOLDER = "[saved credential]"
RECENT_REQUEST_TIMEOUT = (
    10  # Imported from nicegui_app or config? It was in nicegui_app imports but also config.
)
# checking imports in nicegui_app.py: RECENT_REQUEST_TIMEOUT from src.config


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
    on_saved: Optional[Callable[[], None]] = None,
    on_verify: Optional[Callable[[], None]] = None,
) -> NightscoutRefs:
    storage = app.storage.user
    stored_base = storage.get("ns_base_url") or ""
    stored_token = storage.get("ns_token") or ""
    stored_secret = storage.get("ns_api_secret") or ""
    base_prefill = stored_base or DEFAULT_NIGHTSCOUT_URL or ""
    token_prefilled = bool(stored_token)
    secret_prefilled = bool(stored_secret)

    expansion = ui.expansion(value=True).classes(
        "w-full bg-transparent text-slate-100 ns-expansion"
    )
    with expansion.add_slot("header"):
        with ui.row().classes("items-center justify-between w-full gap-3 pr-2"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("link").classes("text-cyan-300")
                ui.label("Nightscout Connection").classes(
                    "text-xs uppercase tracking-[0.5em] text-cyan-200"
                )
                status_dot = ui.icon("fiber_manual_record").classes("connection-dot hidden ml-2")

    with expansion:
        with ui.column().classes(
            "ns-settings-card w-full bg-[#0d1629]/95 border border-cyan-900/40 shadow-2xl shadow-black/40 "
            "rounded-2xl px-6 py-5 text-slate-100 backdrop-blur"
        ):
            status_label = ui.label(f"Current site: {stored_base or 'Not configured'}").classes(
                "text-xs text-slate-400 mb-3"
            )

            base_input = (
                ui.input(
                    label="Base URL",
                    placeholder="https://mysite.herokuapp.com",
                    value=base_prefill,
                )
                .props(
                    'type=url dark outlined dense color="cyan" label-color="cyan" input-class="night-input-text"'
                )
                .classes("night-input night-input-cyan w-full mb-3")
            )

            token_input = (
                ui.input(
                    label="Read token (preferred)",
                    placeholder="Optional",
                    password=True,
                    password_toggle_button=True,
                    value=CACHED_CREDENTIAL_PLACEHOLDER if token_prefilled else "",
                )
                .props(
                    'dark outlined dense color="violet" label-color="violet" input-class="night-input-text"'
                )
                .classes("night-input night-input-violet w-full mb-2")
            )

            secret_input = (
                ui.input(
                    label="API secret (fallback)",
                    placeholder="Optional",
                    password=True,
                    password_toggle_button=True,
                    value=CACHED_CREDENTIAL_PLACEHOLDER if secret_prefilled else "",
                )
                .props(
                    'dark outlined dense color="violet" label-color="violet" input-class="night-input-text"'
                )
                .classes("night-input night-input-violet w-full mb-4")
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

        save_btn = ui.button("Save Nightscout settings", on_click=save_settings).classes(
            "bg-gradient-to-r from-cyan-500 to-blue-500 text-slate-50 font-mono uppercase tracking-[0.4em] "
            "py-2 px-4 rounded-xl shadow-lg shadow-cyan-900/40 hover:opacity-90 transition"
        )

    return NightscoutRefs(
        callout=None,
        expansion=expansion,
        status_label=status_label,
        status_dot=status_dot,
        save_btn=save_btn,
    )


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
