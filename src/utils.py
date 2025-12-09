import pandas as pd


def mean_glucose_to_hba1c(g: float) -> float:
    """
    Computes the estimated Hba1C (or GMI) as defined in Bergenstal et al (doi: 10.2337/dc18-1581)

    Parameters
    ----------
    g : float
    Mean glucose from the CGM readings

    Returns
    -------
    float
    Estimated Hba1c (in %).
    """
    return 3.31 + 0.02395 * g


def strip_timezone(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-naive timestamp normalized to UTC when tz-aware."""
    if isinstance(ts, pd.Timestamp) and ts.tz is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


__all__ = ["mean_glucose_to_hba1c", "strip_timezone"]
