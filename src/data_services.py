from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .config import (
    BG_CATEGORIES,
    LINEPLOT_HOURS,
    RECENT_POINTS,
    TARGET_HIGH,
    TARGET_LOW,
    TARGET_MILD_HIGH,
    TARGET_SEVERE_HIGH,
    TARGET_SEVERE_LOW,
)
from .nightscout_client import NightscoutClient


def fetch_recent_data(client: NightscoutClient) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Fetch the most recent CGM entries and return structured data."""
    entries = client.get_sgv(count=RECENT_POINTS)
    if not entries:
        raise ValueError("Nightscout returned no recent entries.")

    df_recent = pd.DataFrame(entries)
    if "dateString" not in df_recent:
        raise ValueError("Recent data payload is missing dateString.")

    df_recent["date"] = pd.to_datetime(df_recent["dateString"], utc=True)
    df_recent = (
        df_recent[["date", "sgv", "device"]]
        .drop_duplicates()
        .set_index("date")
        .sort_index()
    )

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=LINEPLOT_HOURS)
    df_recent = df_recent.loc[df_recent.index >= cutoff].reset_index()

    last_value = entries[0]
    previous_value = entries[1] if len(entries) > 1 else entries[0]
    return last_value, previous_value, df_recent


def fetch_latest_entry(client: NightscoutClient) -> Dict[str, Any]:
    """Fetch only the most recent CGM entry for minute-level refresh."""
    entries = client.get_sgv(count=10)
    if not entries:
        raise ValueError("Nightscout returned no latest entry.")

    return entries[0]


def fetch_historical_data(
    client: NightscoutClient,
    days: int = 90,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Fetch up to `days` of historical data in manageable chunks."""
    chunk_size = 10_000
    delay_between_requests = 3

    estimated_records = days * 1440
    num_chunks = (estimated_records + chunk_size - 1) // chunk_size

    all_data: list[Dict[str, Any]] = []
    oldest_date: Optional[pd.Timestamp] = None

    for index in range(num_chunks):
        if progress_callback:
            progress_callback(index + 1, num_chunks)

        params: Dict[str, Any] = {"count": chunk_size}
        if oldest_date is not None:
            params["find[dateString][$lt]"] = oldest_date.isoformat()

        if index > 0:
            time.sleep(delay_between_requests)

        raw_chunk = client._get("/api/v1/entries/sgv.json", params=params)
        chunk_data = raw_chunk or []

        if not chunk_data:
            break

        all_data.extend(chunk_data)

        if not chunk_data:
            if raw_chunk:
                newest_timestamp = pd.to_datetime(raw_chunk[-1]["dateString"])
                if isinstance(newest_timestamp, pd.Timestamp) and newest_timestamp.tzinfo is not None:
                    newest_timestamp = newest_timestamp.tz_localize(None)
                oldest_date = newest_timestamp if isinstance(newest_timestamp, pd.Timestamp) else None
            continue

        newest_timestamp = pd.to_datetime(chunk_data[-1]["dateString"])
        if isinstance(newest_timestamp, pd.Timestamp) and newest_timestamp.tzinfo is not None:
            newest_timestamp = newest_timestamp.tz_localize(None)
        oldest_date = newest_timestamp if isinstance(newest_timestamp, pd.Timestamp) else None

        target_start = pd.Timestamp.now() - pd.Timedelta(days=days)
        if oldest_date is not None and oldest_date < target_start:
            break

    if not all_data:
        raise ValueError("Nightscout returned no historical data.")

    df_raw = pd.DataFrame(all_data)
    df_raw["date"] = pd.to_datetime(df_raw["dateString"])
    df_raw = (
        df_raw[["date", "sgv", "device"]]
        .drop_duplicates()
        .set_index("date")
        .sort_index()
    )

    df_3months = (
        df_raw.resample("5 min")["sgv"]
        .mean()
        .reset_index()
        .assign(
            cat_glucose=lambda frame: pd.cut(
                frame["sgv"],
                bins=[
                    0,
                    TARGET_SEVERE_LOW,
                    TARGET_LOW,
                    TARGET_MILD_HIGH,
                    TARGET_HIGH,
                    TARGET_SEVERE_HIGH,
                    np.inf,
                ],
                labels=BG_CATEGORIES,
            )
        )
    )

    return df_3months


def parse_entry_timestamp(entry: Optional[Dict[str, Any]]) -> Optional[pd.Timestamp]:
    if not entry:
        return None
    ts_raw = entry.get("dateString") or entry.get("date")
    if ts_raw is None:
        return None
    return pd.to_datetime(ts_raw, utc=True)


def ensure_timezone_aware(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "date" not in df.columns:
        raise ValueError("Dataframe missing 'date' column.")
    if not is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


async def fetch_historical_async(
    client: NightscoutClient, days: int, callback: Optional[Callable[[int, int], None]]
) -> pd.DataFrame:
    """Allow awaiting the blocking fetch call."""
    return await asyncio.to_thread(fetch_historical_data, client, days, callback)


__all__ = [
    "ensure_timezone_aware",
    "fetch_historical_async",
    "fetch_historical_data",
    "fetch_latest_entry",
    "fetch_recent_data",
    "parse_entry_timestamp",
]
