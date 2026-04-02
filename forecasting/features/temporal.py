"""Temporal feature engineering."""

import numpy as np
import pandas as pd


def create_temporal_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """Create calendar-based features from a date index.

    Args:
        dates: DatetimeIndex to extract features from.

    Returns:
        Array of shape (n_features, n_dates) with values in [0, 1].
    """
    features = {
        "month": dates.month / 12,
        "day": dates.day / 31,
        "day_of_week": dates.dayofweek / 6,
        "week_of_year": dates.isocalendar().week.values / 52,
        "quarter": dates.quarter / 4,
        "is_weekend": (dates.dayofweek >= 5).astype(float),
        "is_month_start": dates.is_month_start.astype(float),
        "is_month_end": dates.is_month_end.astype(float),
    }

    return np.array(list(features.values()))


def add_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add temporal feature columns to a flat DataFrame.

    Args:
        df: DataFrame with a datetime column.
        date_col: Name of the date column.

    Returns:
        DataFrame with new temporal columns added.
    """
    dates = pd.to_datetime(df[date_col])
    df = df.copy()
    df["month"] = dates.dt.month
    df["day"] = dates.dt.day
    df["day_of_week"] = dates.dt.dayofweek
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int).values
    df["day_of_year"] = dates.dt.dayofyear

    return df
