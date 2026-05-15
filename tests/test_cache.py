import time

import pandas as pd
from pandas.testing import assert_frame_equal

from src import cache
from src.state import DataState


def _patch_cache_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "RECENT_CACHE", tmp_path / "recent.json")
    monkeypatch.setattr(cache, "RECENT_CACHE_META", tmp_path / "recent.meta.json")
    monkeypatch.setattr(cache, "HISTORICAL_CACHE", tmp_path / "historical.json")
    monkeypatch.setattr(cache, "HISTORICAL_CACHE_META", tmp_path / "historical.meta.json")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)


def test_historical_cache_round_trips_with_matching_source(monkeypatch, tmp_path):
    _patch_cache_paths(monkeypatch, tmp_path)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
            "sgv": [100, 105],
            "cat_glucose": ["70-150", "70-150"],
        }
    )

    cache.save_historical_cache(df, source_url="https://nightscout.example")
    loaded = cache.load_historical_cache(source_url="https://nightscout.example")

    assert loaded is not None
    assert_frame_equal(loaded, df, check_dtype=False)


def test_historical_cache_rejects_mismatched_source(monkeypatch, tmp_path):
    _patch_cache_paths(monkeypatch, tmp_path)
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=1, tz="UTC"), "sgv": [100]})

    cache.save_historical_cache(df, source_url="https://one.example")

    assert cache.load_historical_cache(source_url="https://two.example") is None


def test_recent_cache_rejects_expired_metadata(monkeypatch, tmp_path):
    _patch_cache_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(cache, "RECENT_TTL_SECONDS", 1)
    state = DataState(
        last_value={"sgv": 100, "dateString": "2026-01-01T00:00:00Z"},
        previous_value={"sgv": 99, "dateString": "2025-12-31T23:55:00Z"},
        df_recent=pd.DataFrame(
            {"date": pd.date_range("2026-01-01", periods=1, tz="UTC"), "sgv": [100]}
        ),
        fetched_at=time.time(),
    )

    cache.save_recent_cache(state, source_url="https://nightscout.example")
    metadata = cache._read_json(cache.RECENT_CACHE_META)
    metadata["fetched_at"] = time.time() - 10
    cache._write_json(cache.RECENT_CACHE_META, metadata)

    restored = DataState()
    cache.load_recent_cache(restored, source_url="https://nightscout.example")

    assert restored.last_value is None
    assert restored.df_recent.empty
