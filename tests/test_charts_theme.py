import pandas as pd
import pytest

from src.charts import CHART_THEMES, build_recent_chart, build_tir_chart, create_placeholder_chart
from src.config import TARGET_LOW, TARGET_MILD_HIGH


@pytest.fixture
def recent_df() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    values = [110, 120, 130, 140, 150, 160]
    return pd.DataFrame({"date": times, "sgv": values})


@pytest.fixture
def tir_df() -> pd.DataFrame:
    times = pd.date_range("2025-01-02", periods=8, freq="h", tz="UTC")
    category = f"{TARGET_LOW}-{TARGET_MILD_HIGH}"
    return pd.DataFrame(
        {
            "date": times,
            "sgv": [115 + idx for idx in range(len(times))],
            "cat_glucose": [category for _ in range(len(times))],
        }
    )


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_recent_chart_respects_palette(theme: str, recent_df: pd.DataFrame) -> None:
    fig = build_recent_chart(recent_df, theme=theme)
    palette = CHART_THEMES[theme]

    assert fig.layout.paper_bgcolor == palette["paper_bg"]
    assert fig.layout.plot_bgcolor == palette["plot_bg"]
    assert fig.layout.font.color == palette["font_color"]
    assert fig.layout.xaxis.gridcolor == palette["grid_color"]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_tir_chart_respects_palette(theme: str, tir_df: pd.DataFrame) -> None:
    fig = build_tir_chart(tir_df, theme=theme)
    palette = CHART_THEMES[theme]

    assert fig.layout.paper_bgcolor == palette["paper_bg"]
    assert fig.layout.yaxis.gridcolor == palette["grid_color"]
    assert fig.layout.font.color == palette["font_color"]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_placeholder_chart_uses_theme_colors(theme: str) -> None:
    fig = create_placeholder_chart("Theme Test", height=120, theme=theme)
    palette = CHART_THEMES[theme]

    assert fig.layout.title.font.color == palette["muted_text"]
    assert fig.layout.paper_bgcolor == palette["paper_bg"]
