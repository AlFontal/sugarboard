from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import CACHE_DIR
from .state import DataState

CACHE_SCHEMA_VERSION = 1
RECENT_CACHE = CACHE_DIR / "nicegui_recent.json"
RECENT_CACHE_META = CACHE_DIR / "nicegui_recent.meta.json"
HISTORICAL_CACHE = CACHE_DIR / "nicegui_historical.json"
HISTORICAL_CACHE_META = CACHE_DIR / "nicegui_historical.meta.json"

RECENT_TTL_SECONDS = 24 * 60 * 60
HISTORICAL_TTL_SECONDS = 7 * 24 * 60 * 60


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None


def _save_meta(path: Path, *, source_url: Optional[str]) -> None:
    _write_json(
        path,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source_url": source_url.rstrip("/") if source_url else None,
            "fetched_at": time.time(),
        },
    )


def _meta_is_valid(
    path: Path,
    *,
    source_url: Optional[str],
    ttl_seconds: Optional[int],
) -> bool:
    metadata = _read_json(path)
    if not metadata:
        return False
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False
    expected_source = source_url.rstrip("/") if source_url else None
    cached_source = metadata.get("source_url")
    if expected_source and cached_source and cached_source != expected_source:
        return False
    fetched_at = metadata.get("fetched_at")
    if ttl_seconds is not None and isinstance(fetched_at, (int, float)):
        if time.time() - fetched_at > ttl_seconds:
            return False
    elif ttl_seconds is not None:
        return False
    return True


def _dataframe_from_json(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_json(path, orient="table")
    except (ValueError, OSError) as exc:
        logging.warning("Ignoring invalid dataframe cache %s: %s", path, exc)
        return None


def _save_dataframe(path: Path, df: pd.DataFrame) -> None:
    df.to_json(path, orient="table", date_format="iso", index=False)


def load_historical_cache(source_url: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not _meta_is_valid(
        HISTORICAL_CACHE_META,
        source_url=source_url,
        ttl_seconds=HISTORICAL_TTL_SECONDS,
    ):
        return None
    return _dataframe_from_json(HISTORICAL_CACHE)


def save_historical_cache(df: pd.DataFrame, source_url: Optional[str] = None) -> None:
    _save_dataframe(HISTORICAL_CACHE, df)
    _save_meta(HISTORICAL_CACHE_META, source_url=source_url)


def load_recent_cache(state: DataState, source_url: Optional[str] = None) -> None:
    if not _meta_is_valid(
        RECENT_CACHE_META,
        source_url=source_url,
        ttl_seconds=RECENT_TTL_SECONDS,
    ):
        return
    cached = _read_json(RECENT_CACHE)
    if not cached:
        return
    state.last_value = cached.get("last_value")
    state.previous_value = cached.get("previous_value")
    df_recent_path = CACHE_DIR / cached.get("df_recent_file", "")
    df_recent = _dataframe_from_json(df_recent_path)
    state.df_recent = df_recent if df_recent is not None else pd.DataFrame()
    state.fetched_at = cached.get("fetched_at")


def save_recent_cache(state: DataState, source_url: Optional[str] = None) -> None:
    if state.df_recent.empty or state.last_value is None:
        return
    recent_df_path = CACHE_DIR / "nicegui_recent_df.json"
    _save_dataframe(recent_df_path, state.df_recent)
    _write_json(
        RECENT_CACHE,
        {
            "last_value": state.last_value,
            "previous_value": state.previous_value,
            "df_recent_file": recent_df_path.name,
            "fetched_at": state.fetched_at,
        },
    )
    _save_meta(RECENT_CACHE_META, source_url=source_url)


__all__ = [
    "load_historical_cache",
    "save_historical_cache",
    "load_recent_cache",
    "save_recent_cache",
]
