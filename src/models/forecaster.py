"""
Demand forecasting module.

Supports Prophet for time series forecasting per SKU with automatic
trend and seasonality detection, tuned for MRO demand patterns.
"""

import pandas as pd
from prophet import Prophet
import logging

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def prepare_prophet_data(demand_df: pd.DataFrame, sku_id: str) -> pd.DataFrame:
    """Extract and format a single SKU's history for Prophet."""
    sku_data = demand_df[demand_df["sku_id"] == sku_id][["date", "demand"]].copy()
    sku_data.columns = ["ds", "y"]
    sku_data["ds"] = pd.to_datetime(sku_data["ds"])
    return sku_data.sort_values("ds").reset_index(drop=True)


def train_prophet(
    train_data: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """Train a Prophet model on a single SKU's demand history."""
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.fit(train_data)
    return model


def forecast_sku(
    demand_df: pd.DataFrame,
    sku_id: str,
    horizon_days: int = 30,
    train_cutoff: str | None = None,
) -> dict:
    """
    Train and forecast demand for a single SKU.

    Returns dict with model, forecast DataFrame, and train/test split info.
    """
    data = prepare_prophet_data(demand_df, sku_id)

    if train_cutoff is None:
        train_cutoff = str(data["ds"].max() - pd.Timedelta(days=horizon_days))

    train = data[data["ds"] <= train_cutoff]
    test = data[data["ds"] > train_cutoff]

    model = train_prophet(train)

    future = model.make_future_dataframe(periods=horizon_days + len(test))
    forecast = model.predict(future)

    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

    return {
        "sku_id": sku_id,
        "model": model,
        "forecast": forecast,
        "train": train,
        "test": test,
        "horizon_days": horizon_days,
    }


def forecast_all_skus(
    demand_df: pd.DataFrame,
    sku_ids: list[str] | None = None,
    horizon_days: int = 30,
) -> dict[str, dict]:
    """Run forecasting for multiple SKUs."""
    if sku_ids is None:
        sku_ids = demand_df["sku_id"].unique().tolist()

    results = {}
    for sku_id in sku_ids:
        try:
            results[sku_id] = forecast_sku(demand_df, sku_id, horizon_days)
        except Exception as e:
            print(f"Warning: forecast failed for {sku_id}: {e}")
            continue

    return results


def get_forecast_summary(forecast_result: dict) -> pd.DataFrame:
    """Extract a clean forecast summary table from a forecast result."""
    fc = forecast_result["forecast"]
    train_end = forecast_result["train"]["ds"].max()

    future_fc = fc[fc["ds"] > train_end][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_fc.columns = ["date", "predicted_demand", "lower_bound", "upper_bound"]
    future_fc["predicted_demand"] = future_fc["predicted_demand"].round(0).astype(int)
    future_fc["lower_bound"] = future_fc["lower_bound"].round(0).astype(int)
    future_fc["upper_bound"] = future_fc["upper_bound"].round(0).astype(int)
    future_fc["sku_id"] = forecast_result["sku_id"]

    return future_fc.reset_index(drop=True)
