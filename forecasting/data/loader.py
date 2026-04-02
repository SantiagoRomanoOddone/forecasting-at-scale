"""Data loading and basic cleaning."""

import logging

import pandas as pd

from forecasting.config.schema import DataConfig

logger = logging.getLogger(__name__)


def load_data(config: DataConfig) -> pd.DataFrame:
    """Load raw data from file and standardize column types.

    Args:
        config: Data configuration with path, column names, separator.

    Returns:
        DataFrame with parsed dates and lowercase columns.
    """
    logger.info("Loading data from %s", config.path)

    df = pd.read_csv(config.path, sep=config.sep)
    df.columns = df.columns.str.lower()

    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.sort_values(config.date_col).reset_index(drop=True)

    logger.info(
        "Loaded %d rows | %d products | %d stores | %s to %s",
        len(df),
        df[config.product_col].nunique(),
        df[config.store_col].nunique(),
        df[config.date_col].min().date(),
        df[config.date_col].max().date(),
    )

    return df
