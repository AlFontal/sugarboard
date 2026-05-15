from __future__ import annotations

import os
from pathlib import Path
from secrets import token_hex
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True, mode=0o700)
CACHE_DIR.chmod(0o700)


def _int_from_env(name: str, default: int) -> int:
    """Best-effort conversion for optional integer environment settings."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Optional default Nightscout site (used only to prefill the UI)
DEFAULT_NIGHTSCOUT_URL = os.environ.get("NIGHTSCOUT_BASE_URL") or os.environ.get("CGM_SITE", "")
_RAW_STORAGE_SECRET = os.environ.get("STORAGE_SECRET")
if _RAW_STORAGE_SECRET:
    STORAGE_SECRET = _RAW_STORAGE_SECRET
    STORAGE_SECRET_FROM_ENV = True
else:
    if os.environ.get("SUGARBOARD_REQUIRE_STORAGE_SECRET", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        raise ValueError("STORAGE_SECRET must be set when SUGARBOARD_REQUIRE_STORAGE_SECRET=1.")
    STORAGE_SECRET = token_hex(32)
    STORAGE_SECRET_FROM_ENV = False
LINEPLOT_HOURS = _int_from_env("LINEPLOT_HOURS", 4)
RECENT_POINTS = LINEPLOT_HOURS * 75
RECENT_REQUEST_TIMEOUT = _int_from_env("RECENT_REQUEST_TIMEOUT", 60)

_RAW_DISPLAY_TIMEZONE = os.environ.get("DISPLAY_TIMEZONE") or os.environ.get("TZ") or "UTC"
try:
    ZoneInfo(_RAW_DISPLAY_TIMEZONE)
    DISPLAY_TIMEZONE = _RAW_DISPLAY_TIMEZONE
except ZoneInfoNotFoundError:
    DISPLAY_TIMEZONE = "UTC"

ALLOW_HTTP = os.environ.get("ALLOW_HTTP", "").lower() in {"1", "true", "yes"}
ALLOW_INSECURE_NS_URLS = os.environ.get("ALLOW_INSECURE_NS_URLS", "").lower() in {
    "1",
    "true",
    "yes",
}

# Glucose targets
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

# Palette
STRONG_RED = "#960200"
LIGHT_RED = "#CE6C47"
MILD_YELLOW = "#FFD046"
LIGHT_GREEN = "#49D49D"

__all__ = [
    "BASE_DIR",
    "CACHE_DIR",
    "DEFAULT_NIGHTSCOUT_URL",
    "STORAGE_SECRET",
    "STORAGE_SECRET_FROM_ENV",
    "LINEPLOT_HOURS",
    "RECENT_POINTS",
    "RECENT_REQUEST_TIMEOUT",
    "DISPLAY_TIMEZONE",
    "ALLOW_HTTP",
    "ALLOW_INSECURE_NS_URLS",
    "TARGET_SEVERE_LOW",
    "TARGET_LOW",
    "TARGET_MILD_HIGH",
    "TARGET_HIGH",
    "TARGET_SEVERE_HIGH",
    "BG_CATEGORIES",
    "DIRECTIONS",
    "STRONG_RED",
    "LIGHT_RED",
    "MILD_YELLOW",
    "LIGHT_GREEN",
]
