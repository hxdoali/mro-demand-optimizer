"""Evaluation metrics for demand forecasting and inventory optimization."""

import numpy as np
import pandas as pd


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error, excluding zeros in actual."""
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def fill_rate(demand: np.ndarray, available_stock: np.ndarray) -> float:
    """Fraction of demand fulfilled from available stock."""
    total_demand = demand.sum()
    if total_demand == 0:
        return 1.0
    fulfilled = np.minimum(demand, available_stock).sum()
    return float(fulfilled / total_demand)


def inventory_cost(
    avg_inventory: float,
    unit_cost: float,
    holding_cost_pct: float,
    stockout_units: float = 0,
    stockout_penalty_per_unit: float = 50.0,
) -> dict:
    """Calculate total inventory cost breakdown."""
    holding = avg_inventory * unit_cost * holding_cost_pct
    stockout = stockout_units * stockout_penalty_per_unit
    return {
        "holding_cost": round(holding, 2),
        "stockout_cost": round(stockout, 2),
        "total_cost": round(holding + stockout, 2),
    }


def forecast_summary(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute a summary of forecast accuracy metrics."""
    a = actual.values
    p = predicted.values
    return {
        "mape": round(mape(a, p), 2),
        "rmse": round(rmse(a, p), 2),
        "mean_actual": round(float(a.mean()), 2),
        "mean_predicted": round(float(p.mean()), 2),
        "n_periods": len(a),
    }
