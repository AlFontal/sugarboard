from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

from nicegui import app, ui

from src.config import (
    ALLOW_HTTP,
    ALLOW_INSECURE_NS_URLS,
    DEFAULT_NIGHTSCOUT_URL,
    RECENT_REQUEST_TIMEOUT,
    STORAGE_SECRET_FROM_ENV,
)
from src.nightscout_client import NightscoutClient


@dataclass
class NightscoutRefs:
    callout: Optional[Any]
    expansion: Any
    status_label: Any
    status_dot: Any
    save_btn: Any


CACHED_CREDENTIAL_PLACEHOLDER = "[saved credential]"


def _sanitize_base_url(value: str) -> str:
    raw_value = value.strip().rstrip("/")
    if not raw_value:
        return ""
    if "://" not in raw_value:
        raw_value = f"https://{raw_value}"

    parsed = urlparse(raw_value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Nightscout URL must use http or https.")
    if parsed.scheme == "http" and not ALLOW_HTTP:
        raise ValueError("Nightscout URL must use https unless ALLOW_HTTP=1 is set.")
    if parsed.scheme == "http":
        logging.warning("ALLOW_HTTP enabled; Nightscout traffic is not encrypted.")
    if not parsed.hostname:
        raise ValueError("Nightscout URL is missing a host.")
    if parsed.username or parsed.password:
        raise ValueError("Nightscout URL must not include embedded credentials.")
    if _host_is_private(parsed.hostname) and not ALLOW_INSECURE_NS_URLS:
        raise ValueError(
            "Nightscout URL points to a private/local address. "
            "Set ALLOW_INSECURE_NS_URLS=1 only for trusted local deployments."
        )

    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def _host_is_private(hostname: str) -> bool:
    if hostname.lower() in {"localhost", "localhost.localdomain"}:
        return True
    addresses: set[str] = set()
    try:
        addresses.add(str(ipaddress.ip_address(hostname)))
    except ValueError:
        try:
            addresses.update(info[4][0] for info in socket.getaddrinfo(hostname, None))
        except socket.gaierror:
            return False

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def get_client_from_storage() -> Optional[NightscoutClient]:
    storage = app.storage.user
    base_url = storage.get("ns_base_url")
    if not base_url:
        return None
    try:
        base_url = _sanitize_base_url(base_url)
    except ValueError as exc:
        logging.warning("Ignoring invalid Nightscout URL from storage: %s", exc)
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
            try:
                base = _sanitize_base_url(base_input.value or "")
            except ValueError as exc:
                ui.notify(str(exc), type="warning")
                return
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
