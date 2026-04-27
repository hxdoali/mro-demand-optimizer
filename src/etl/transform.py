"""
Feature engineering and data transformation.

Transforms raw demand data into model-ready features including
lag features, rolling statistics, and calendar encodings.
"""

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the date column."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_col: str = "sku_id",
    target_col: str = "demand",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged demand values per SKU."""
    if lags is None:
        lags = [1, 7, 14, 28]

    df = df.sort_values(["sku_id", "date"]).copy()
    for lag in lags:
        df[f"demand_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str = "sku_id",
    target_col: str = "demand",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std per SKU."""
    if windows is None:
        windows = [7, 14, 30]

    df = df.sort_values(["sku_id", "date"]).copy()
    for w in windows:
        grouped = df.groupby(group_col)[target_col]
        df[f"demand_roll_mean_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"demand_roll_std_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    return df


def build_feature_table(
    demand_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full transformation pipeline: calendar features, lags, rolling stats,
    joined with product catalog metadata.
    """
    df = add_calendar_features(demand_df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    df = df.merge(
        catalog_df[["sku_id", "category", "unit_cost", "lead_time_days", "holding_cost_pct"]],
        on="sku_id",
        how="left",
    )

    df = df.dropna(subset=["demand_lag_28"])

    return df.reset_index(drop=True)
