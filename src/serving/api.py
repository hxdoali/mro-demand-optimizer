"""
FastAPI serving layer for demand forecasting and inventory optimization.

Provides REST endpoints for programmatic access to forecasts
and optimization recommendations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.etl.ingest import load_pandas
from src.etl.cluster import cluster_products
from src.models.forecaster import forecast_sku, get_forecast_summary
from src.models.optimizer import compute_demand_stats, optimize_single_sku

app = FastAPI(
    title="MRO Inventory Optimizer API",
    description="Demand forecasting and inventory optimization for MRO products",
    version="1.0.0",
)

DATA_DIR = "data/synthetic"
_data_cache: dict = {}


def get_data():
    if not _data_cache:
        data = load_pandas(DATA_DIR)
        _data_cache["catalog"] = cluster_products(data["catalog"])
        _data_cache["demand"] = data["demand"]
    return _data_cache


class ForecastRequest(BaseModel):
    sku_id: str
    horizon_days: int = Field(default=30, ge=1, le=90)


class OptimizeRequest(BaseModel):
    sku_id: str
    service_level: float = Field(default=0.95, ge=0.80, le=0.99)


class HealthResponse(BaseModel):
    status: str
    n_skus: int
    n_records: int


@app.get("/health", response_model=HealthResponse)
def health_check():
    data = get_data()
    return HealthResponse(
        status="healthy",
        n_skus=len(data["catalog"]),
        n_records=len(data["demand"]),
    )


@app.get("/skus")
def list_skus():
    data = get_data()
    catalog = data["catalog"]
    return catalog[["sku_id", "category", "description", "unit_cost", "lead_time_days"]].to_dict(orient="records")


@app.post("/forecast")
def run_forecast(request: ForecastRequest):
    data = get_data()
    if request.sku_id not in data["demand"]["sku_id"].values:
        raise HTTPException(status_code=404, detail=f"SKU {request.sku_id} not found")

    result = forecast_sku(data["demand"], request.sku_id, horizon_days=request.horizon_days)
    summary = get_forecast_summary(result)

    return {
        "sku_id": request.sku_id,
        "horizon_days": request.horizon_days,
        "forecast": summary.to_dict(orient="records"),
    }


@app.post("/optimize")
def run_optimization(request: OptimizeRequest):
    data = get_data()
    catalog = data["catalog"]
    demand = data["demand"]

    if request.sku_id not in catalog["sku_id"].values:
        raise HTTPException(status_code=404, detail=f"SKU {request.sku_id} not found")

    stats = compute_demand_stats(demand, request.sku_id)
    cat_row = catalog[catalog["sku_id"] == request.sku_id].iloc[0]

    result = optimize_single_sku(
        mean_daily=stats["mean_daily"],
        std_daily=stats["std_daily"],
        lead_time_days=int(cat_row["lead_time_days"]),
        unit_cost=float(cat_row["unit_cost"]),
        holding_cost_pct=float(cat_row["holding_cost_pct"]),
        service_level=request.service_level,
    )

    return {
        "sku_id": request.sku_id,
        "service_level": request.service_level,
        "demand_stats": stats,
        "recommendation": result,
    }
