import pandas as pd

from src.data_services import ensure_timezone_aware, fetch_recent_data


class FakeClient:
    def get_sgv(self, count):
        now = pd.Timestamp.now(tz="UTC")
        return [
            {
                "dateString": now.isoformat(),
                "sgv": 110,
                "device": "test",
            },
            {
                "dateString": (now - pd.Timedelta(minutes=5)).isoformat(),
                "sgv": 100,
                "device": "test",
            },
        ]


def test_ensure_timezone_aware_localizes_naive_dates():
    df = pd.DataFrame({"date": [pd.Timestamp("2026-01-01 00:00:00")], "sgv": [100]})

    result = ensure_timezone_aware(df)

    assert result["date"].dt.tz is not None
    assert str(result["date"].dt.tz) == "UTC"


def test_fetch_recent_data_normalizes_to_utc():
    last_value, previous_value, df_recent = fetch_recent_data(FakeClient())

    assert last_value["sgv"] == 110
    assert previous_value["sgv"] == 100
    assert df_recent["date"].dt.tz is not None
