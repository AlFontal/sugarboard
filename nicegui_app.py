from __future__ import annotations

import os
from pathlib import Path

from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import STORAGE_SECRET
from src.log_setup import setup_logging
from src.ui.pages import index_page  # noqa: F401 - Register index page

setup_logging()

CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "connect-src 'self' ws: wss:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'"
)


async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Content-Security-Policy", CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


app.add_middleware(BaseHTTPMiddleware, dispatch=_security_headers)


@ui.page("/health")
def healthcheck() -> None:
    """Health check endpoint."""
    ui.label("ok")


if __name__ in {"__main__", "__mp_main__"}:
    # Add static files route for assets
    app.add_static_files("/static", str(Path(__file__).parent / "static"))

    port = int(os.environ.get("PORT", "8080"))
    reload_enabled = os.environ.get("NICEGUI_RELOAD", "false").lower() in {
        "1",
        "true",
        "yes",
    }

    ui.run(
        title="SugarBoard NiceGUI",
        host="0.0.0.0",
        port=port,
        reload=reload_enabled,
        storage_secret=STORAGE_SECRET,
        favicon="static/favicon.png",
    )
