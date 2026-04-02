"""Evaluation metrics for forecast quality."""

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Ignores zeros in actual."""
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def evaluate(results: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics per store from prediction results.

    Args:
        results: DataFrame with columns: store_idx, date, prediction, actual.

    Returns:
        DataFrame with one row per store and columns: store_idx, rmse, mae, mape.
    """
    metrics = []
    for store_idx, group in results.groupby("store_idx"):
        actual = group["actual"].values
        predicted = group["prediction"].values
        metrics.append({
            "store_idx": store_idx,
            "rmse": rmse(actual, predicted),
            "mae": mae(actual, predicted),
            "mape": mape(actual, predicted),
        })

    return pd.DataFrame(metrics)
