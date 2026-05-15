import pandas as pd

from src.data_services import ensure_timezone_aware
from src.hero import calculate_hero_metrics


def _make_entry(row):
    return {
        "sgv": int(row["sgv"]),
        "device": row.get("device", "xDrip"),
        "dateString": pd.Timestamp(row["date"]).isoformat(),
        "direction": "Flat",
    }


def load_recent_df():
    return ensure_timezone_aware(pd.read_json("tests/data/recent.json"))


def test_calculate_metrics_returns_values():
    df = load_recent_df()
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    metrics = calculate_hero_metrics(_make_entry(last_row), _make_entry(prev_row), df)
    assert metrics is not None
    assert "mg/dL" in metrics.last_value_text
    assert metrics.streak_minutes >= 0
    assert metrics.device_text.startswith("from device")


def test_streak_zero_when_out_of_range():
    df = load_recent_df().copy()
    df.loc[df.index[-1], "sgv"] = 300
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    metrics = calculate_hero_metrics(_make_entry(last_row), _make_entry(prev_row), df)
    assert metrics.streak_minutes == 0


def test_streak_does_not_cross_large_data_gap():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T02:00:00Z",
                ],
                utc=True,
            ),
            "sgv": [100, 105, 110],
            "device": ["xDrip", "xDrip", "xDrip"],
        }
    )
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    metrics = calculate_hero_metrics(_make_entry(last_row), _make_entry(prev_row), df)

    assert metrics.streak_minutes == 0
