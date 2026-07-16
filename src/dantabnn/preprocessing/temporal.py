"""Temporal feature utilities: cyclical time encoding (v0.3.4)."""

import numpy as np
import pandas as pd


def add_cyclical_time(
    df: pd.DataFrame,
    datetime_col: str = None,
    period_col: str = None,
) -> pd.DataFrame:
    """Add sin/cos encoding for hourly, daily, and monthly periodicity.

    Converts raw hours (0-23) and day-of-week (0-6) into continuous
    cyclical features so that hour 23 and hour 0 are close in feature
    space. Neural networks benefit from this; tree-based models do not
    need it (splits handle ranges natively).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. Not mutated — a copy is returned.
    datetime_col : str, optional
        Name of a datetime column. If provided, hour and day-of-week
        are extracted via ``pd.to_datetime``.
    period_col : str, optional
        Integer period column (e.g. 0-47 for half-hourly). If provided,
        hour = period % 24, dow = (period // 24) % 7.

    Returns
    -------
    pd.DataFrame
        DataFrame with up to 6 new columns: ``hour_sin``, ``hour_cos``,
        ``dow_sin``, ``dow_cos``, ``month_sin``, ``month_cos``.
        Original columns are preserved.
    """
    df = df.copy()

    if datetime_col is not None and datetime_col in df.columns:
        dt = pd.to_datetime(df[datetime_col])
        hour = dt.dt.hour.astype(float)
        dow = dt.dt.dayofweek.astype(float)
        month = dt.dt.month.astype(float)
    elif period_col is not None and period_col in df.columns:
        period = df[period_col].astype(float)
        hour = period % 24
        dow = (period // 24) % 7
        month = None
    else:
        return df

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    if month is not None:
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df