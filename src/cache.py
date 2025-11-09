from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import CACHE_DIR
from .state import DataState

RECENT_CACHE = CACHE_DIR / "nicegui_recent.pkl"
HISTORICAL_CACHE = CACHE_DIR / "nicegui_historical.pkl"


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save_pickle(path: Path, data: Any) -> None:
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def load_historical_cache() -> Optional[pd.DataFrame]:
    cached = _load_pickle(HISTORICAL_CACHE)
    if not cached:
        return None
    return cached.get("df_3months")


def save_historical_cache(df: pd.DataFrame) -> None:
    _save_pickle(HISTORICAL_CACHE, {"df_3months": df, "cached_at": time.time()})


def load_recent_cache(state: DataState) -> None:
    cached = _load_pickle(RECENT_CACHE)
    if not cached:
        return
    state.last_value = cached.get("last_value")
    state.previous_value = cached.get("previous_value")
    state.df_recent = cached.get("df_recent", pd.DataFrame())
    state.fetched_at = cached.get("fetched_at")


def save_recent_cache(state: DataState) -> None:
    if state.df_recent.empty or state.last_value is None:
        return
    _save_pickle(
        RECENT_CACHE,
        {
            "last_value": state.last_value,
            "previous_value": state.previous_value,
            "df_recent": state.df_recent,
            "fetched_at": state.fetched_at,
        },
    )


__all__ = [
    "load_historical_cache",
    "save_historical_cache",
    "load_recent_cache",
    "save_recent_cache",
]
