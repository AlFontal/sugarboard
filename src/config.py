from __future__ import annotations

import os
from pathlib import Path
from secrets import token_hex

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _int_from_env(name: str, default: int) -> int:
    """Best-effort conversion for optional integer environment settings."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Optional default Nightscout site (used only to prefill the UI)
DEFAULT_NIGHTSCOUT_URL = os.environ.get("NIGHTSCOUT_BASE_URL", "")
_RAW_STORAGE_SECRET = os.environ.get("STORAGE_SECRET")
if _RAW_STORAGE_SECRET:
    STORAGE_SECRET = _RAW_STORAGE_SECRET
    STORAGE_SECRET_FROM_ENV = True
else:
    STORAGE_SECRET = token_hex(32)
    STORAGE_SECRET_FROM_ENV = False
LINEPLOT_HOURS = _int_from_env("LINEPLOT_HOURS", 4)
RECENT_POINTS = LINEPLOT_HOURS * 75
RECENT_REQUEST_TIMEOUT = _int_from_env("RECENT_REQUEST_TIMEOUT", 20)

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
