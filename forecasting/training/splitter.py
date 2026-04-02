"""Train / validation / test splitting logic."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_by_date(
    pivot: pd.DataFrame,
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split pivoted data into train, validation, and test sets.

    The split is based on the prediction_length:
    - Train: everything except last 2 * prediction_length days.
    - Validation: everything except last prediction_length days.
    - Test: everything.

    This mirrors the GluonTS convention where each split contains
    the full history up to its cutoff point.

    Args:
        pivot: Pivoted DataFrame (index=dates, columns=stores).
        prediction_length: Number of days to forecast.

    Returns:
        Tuple of (train, validation, test) DataFrames.
    """
    n = len(pivot)
    train_end = n - 2 * prediction_length
    val_end = n - prediction_length

    train = pivot.iloc[:train_end]
    val = pivot.iloc[:val_end]
    test = pivot  # full series

    logger.info(
        "Split: train=%d days, val=%d days, test=%d days",
        len(train), len(val), len(test),
    )

    return train, val, test


def split_tabular_by_date(
    df: pd.DataFrame,
    date_col: str,
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split flat tabular data into train, validation, and test sets.

    Args:
        df: Flat DataFrame with a date column.
        date_col: Name of the date column.
        prediction_length: Number of days for the test/val window.

    Returns:
        Tuple of (train, validation, test) DataFrames.
    """
    dates = sorted(df[date_col].unique())
    test_start = dates[-prediction_length]
    val_start = dates[-2 * prediction_length]

    train = df[df[date_col] < val_start]
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)]
    test = df[df[date_col] >= test_start]

    logger.info(
        "Tabular split: train=%d rows, val=%d rows, test=%d rows",
        len(train), len(val), len(test),
    )

    return train, val, test
