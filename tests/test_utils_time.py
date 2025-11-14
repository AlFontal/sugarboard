import pandas as pd

from src.utils import strip_timezone


def test_strip_timezone_removes_offset():
    aware = pd.Timestamp("2025-01-01T00:00:00Z")
    naive = strip_timezone(aware)
    assert naive.tzinfo is None
    assert str(naive) == "2025-01-01 00:00:00"


def test_strip_timezone_noop_for_naive():
    naive = pd.Timestamp("2025-01-01 12:00:00")
    assert strip_timezone(naive) == naive
