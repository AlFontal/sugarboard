from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .config import DIRECTIONS, TARGET_LOW, TARGET_MILD_HIGH
from .data_services import ensure_timezone_aware, parse_entry_timestamp
from .utils import strip_timezone


@dataclass
class HeroMetrics:
    last_value_text: str
    last_subtitle_text: str
    delta_text: str
    delta_value: float
    delta_rate_text: str
    updated_text: str
    device_text: str
    streak_minutes: int


def _calculate_in_range_streak_minutes(df_recent: pd.DataFrame) -> int:
    if df_recent.empty or "date" not in df_recent or "sgv" not in df_recent:
        return 0
    try:
        df = ensure_timezone_aware(df_recent.copy()).sort_values("date")
    except Exception:
        return 0
    last_row = df.iloc[-1]
    if not (TARGET_LOW <= last_row["sgv"] <= TARGET_MILD_HIGH):
        return 0
    streak = 0
    prev_time = pd.Timestamp(last_row["date"]) + pd.Timedelta(minutes=5)
    for _, row in df.iloc[::-1].iterrows():
        if not (TARGET_LOW <= row["sgv"] <= TARGET_MILD_HIGH):
            break
        row_time = pd.Timestamp(row["date"])
        delta_minutes = int((prev_time - row_time).total_seconds() / 60)
        if delta_minutes <= 0:
            delta_minutes = 5
        streak += delta_minutes
        prev_time = row_time
    return streak


def calculate_hero_metrics(
    last_entry: Optional[Dict[str, Any]],
    previous_entry: Optional[Dict[str, Any]],
    df_recent: pd.DataFrame,
    local_timezone: str = "Europe/Madrid",
) -> Optional[HeroMetrics]:
    if last_entry is None:
        return None

    last_ts = parse_entry_timestamp(last_entry) or pd.Timestamp.now(tz="UTC")
    prev_ts = parse_entry_timestamp(previous_entry) or last_ts
    last_local = last_ts.tz_convert(local_timezone)

    last_value = last_entry.get("sgv", "--")
    prev_value = previous_entry.get("sgv", last_value)
    delta = (last_value or 0) - (prev_value or 0)
    delta_minutes = (last_ts - prev_ts).total_seconds() / 60 if prev_ts else 0
    delta_per_min = delta / delta_minutes if delta_minutes else 0.0

    trend_raw = last_entry.get("direction") or "Flat"
    curr_dir = DIRECTIONS.get(trend_raw, "→")

    minutes_since = max(0, int((pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60))
    subtitle = "just now" if minutes_since == 0 else f"{minutes_since} min ago"
    delta_rate_text = f"{delta_per_min:+.2f} mg/dL/min" if delta_minutes else ""

    streak_minutes = _calculate_in_range_streak_minutes(df_recent)

    device_name = last_entry.get("device") or "—"

    return HeroMetrics(
        last_value_text=f"{last_value} mg/dL {curr_dir}",
        last_subtitle_text=subtitle,
        delta_text=f"{delta:+.0f} mg/dL",
        delta_value=float(delta),
        delta_rate_text=delta_rate_text,
        updated_text=f"{last_local.strftime('%d %b %Y · %H:%M')}",
        device_text=f"from device: {device_name}",
        streak_minutes=streak_minutes,
    )


__all__ = ["calculate_hero_metrics", "HeroMetrics"]
