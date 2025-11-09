from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class DataState:
    """Global application state shared across UI callbacks."""

    last_value: Optional[Dict[str, Any]] = None
    previous_value: Optional[Dict[str, Any]] = None
    df_recent: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_3months: pd.DataFrame = field(default_factory=pd.DataFrame)
    fetched_at: Optional[float] = None


__all__ = ["DataState"]
