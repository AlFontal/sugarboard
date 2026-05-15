import pandas as pd


def mean_glucose_to_gmi(g: float) -> float:
    """
    Compute glucose management indicator (GMI) as defined in Bergenstal et al.

    Parameters
    ----------
    g : float
    Mean glucose from the CGM readings

    Returns
    -------
    float
    GMI (in %).
    """
    return 3.31 + 0.02395 * g


def mean_glucose_to_hba1c(g: float) -> float:
    """Backward-compatible alias for callers not yet migrated to GMI naming."""
    return mean_glucose_to_gmi(g)


def strip_timezone(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-naive timestamp normalized to UTC when tz-aware."""
    if isinstance(ts, pd.Timestamp) and ts.tz is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


__all__ = ["mean_glucose_to_gmi", "mean_glucose_to_hba1c", "strip_timezone"]
