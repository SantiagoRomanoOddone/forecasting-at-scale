"""Series profiling: analyze each store-product combination before training."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def profile_series(
    df: pd.DataFrame,
    date_col: str,
    store_col: str,
    product_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Profile every store-product combination.

    Missing dates within each series' own range are treated as 0 sales.
    Fill rates are calculated within the series' own date range.
    days_since_last_sale uses the global max date.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    global_max = df[date_col].max()

    records = []

    for (store, product), group in df.groupby([store_col, product_col]):
        first_date = group[date_col].min()
        last_date = group[date_col].max()

        all_days = pd.date_range(first_date, last_date, freq="D")
        total_days = len(all_days)
        series = (
            group.groupby(date_col)[target_col].sum()
            .reindex(all_days, fill_value=0)
        )

        days_with_data = (series > 0).sum()
        fill_rate = days_with_data / total_days

        # Fill rates for last 30, 90, 120 days of the series
        def _fill_rate_last_n(s, n):
            tail = s.iloc[-n:] if len(s) >= n else s
            return (tail > 0).sum() / len(tail)

        records.append({
            store_col: store,
            product_col: product,
            "first_date": first_date.date(),
            "last_date": last_date.date(),
            "history_days": total_days,
            "fill_rate": round(fill_rate, 3),
            "fill_rate_last_30": round(_fill_rate_last_n(series, 30), 3),
            "fill_rate_last_90": round(_fill_rate_last_n(series, 90), 3),
            "fill_rate_last_120": round(_fill_rate_last_n(series, 120), 3),
            "total_sales": series.sum(),
            "mean_sales": round(series.mean(), 2),
            "days_since_last_sale": (global_max - last_date).days,
            "global_max_date": global_max.date(),
        })

    profile = pd.DataFrame(records)
    profile["series_class"] = _classify_series(profile)

    logger.info(
        "Profiled %d combinations. Class distribution:\n%s",
        len(profile),
        profile["series_class"].value_counts().to_string(),
    )

    return profile


def _classify_series(profile: pd.DataFrame) -> pd.Series:
    """Classify each series into quality tiers.

    A - Strong: 6+ months, fill rate > 0.8, active recently
    B - Moderate: 3+ months, fill rate > 0.5, active recently
    C - Weak: short history or sparse
    D - Dead: no sales in last 30 days
    """
    classes = pd.Series("C", index=profile.index)

    strong = (
        (profile["history_days"] >= 180)
        & (profile["fill_rate"] >= 0.8)
        & (profile["fill_rate_last_30"] >= 0.7)
    )
    classes[strong] = "A"

    moderate = (
        ~strong
        & (profile["history_days"] >= 90)
        & (profile["fill_rate"] >= 0.5)
        & (profile["fill_rate_last_30"] >= 0.5)
    )
    classes[moderate] = "B"

    dead = profile["days_since_last_sale"] >= 30
    classes[dead] = "D"

    return classes
