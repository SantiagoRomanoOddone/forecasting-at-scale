"""Data preprocessing: pivot, fill dates, prepare for models."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from forecasting.config.schema import DataConfig

logger = logging.getLogger(__name__)



def pivot_by_store(
    df: pd.DataFrame,
    date_col: str,
    store_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Pivot so each store becomes a column. One row per date.

    Args:
        df: Raw dataframe.
        date_col: Date column name.
        store_col: Store column name.
        target_col: Target variable column name.

    Returns:
        Pivoted DataFrame: index=date, columns=store_ids, values=target.
    """
    pivot = df.pivot_table(
        index=date_col,
        columns=store_col,
        values=target_col,
        aggfunc="sum",
    ).fillna(0)

    pivot = pivot.reindex(
        pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
    ).fillna(0)
    pivot.index.name = date_col

    return pivot


def prepare_series(
    df: pd.DataFrame,
    config: DataConfig,
    product_filter: Optional[object] = None,
    category_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Filter data for a specific product/category and pivot by store.

    Args:
        df: Full dataframe.
        config: Data configuration.
        product_filter: Specific product code to filter.
        category_filter: Category value to filter (uses config.category_col).

    Returns:
        Pivoted DataFrame ready for model consumption.
    """
    subset = df.copy()

    if product_filter is not None:
        subset = subset[subset[config.product_col] == product_filter]
        logger.info("Filtered product=%s -> %d rows", product_filter, len(subset))

    if category_filter is not None and config.category_col is not None:
        subset = subset[subset[config.category_col] == category_filter]
        logger.info("Filtered category=%s -> %d rows", category_filter, len(subset))

    if subset.empty:
        raise ValueError(
            f"No data after filtering (product={product_filter}, "
            f"category={category_filter})"
        )

    pivot = pivot_by_store(
        subset,
        date_col=config.date_col,
        store_col=config.store_col,
        target_col=config.target_col,
    )

    return pivot
