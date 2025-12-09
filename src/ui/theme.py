from typing import Any, Optional

from nicegui import ui

from src.state import DataState

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


def set_active_theme(
    state: DataState, theme: str, storage: Optional[dict[str, Any]] = None
) -> None:
    """Persist and apply the active theme."""
    state.theme = theme if theme in {"dark", "light"} else DEFAULT_THEME
    if storage is not None:
        storage[THEME_STORAGE_KEY] = state.theme
    apply_theme_classes(state.theme)
