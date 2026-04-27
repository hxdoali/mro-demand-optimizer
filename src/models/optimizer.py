"""
Inventory optimization via linear programming.

Uses PuLP to solve for optimal reorder points and order quantities
that minimize total inventory cost (holding + stockout) subject to
service level and capacity constraints.
"""

import numpy as np
import pandas as pd
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpStatus,
    value,
)


def compute_demand_stats(
    demand_df: pd.DataFrame,
    sku_id: str,
    recent_days: int = 90,
) -> dict:
    """Compute demand statistics for a SKU over a recent window."""
    sku_data = demand_df[demand_df["sku_id"] == sku_id].copy()
    sku_data["date"] = pd.to_datetime(sku_data["date"])
    cutoff = sku_data["date"].max() - pd.Timedelta(days=recent_days)
    recent = sku_data[sku_data["date"] >= cutoff]

    daily_demand = recent["demand"]
    return {
        "mean_daily": float(daily_demand.mean()),
        "std_daily": float(daily_demand.std()),
        "max_daily": float(daily_demand.max()),
        "total_recent": float(daily_demand.sum()),
    }


def safety_stock(
    std_daily_demand: float,
    lead_time_days: int,
    service_level_z: float = 1.65,
) -> float:
    """
    Calculate safety stock using the standard formula.
    z * sigma_demand * sqrt(lead_time)
    """
    return service_level_z * std_daily_demand * np.sqrt(lead_time_days)


def reorder_point(
    mean_daily_demand: float,
    lead_time_days: int,
    safety_stock_units: float,
) -> float:
    """Reorder point = lead time demand + safety stock."""
    return mean_daily_demand * lead_time_days + safety_stock_units


def economic_order_quantity(
    annual_demand: float,
    ordering_cost: float = 25.0,
    unit_cost: float = 10.0,
    holding_cost_pct: float = 0.25,
) -> float:
    """Classic EOQ formula."""
    holding = unit_cost * holding_cost_pct
    if holding <= 0:
        return annual_demand / 12
    return float(np.sqrt(2 * annual_demand * ordering_cost / holding))


def optimize_inventory_lp(
    skus: list[dict],
    warehouse_capacity: float = 100000,
    min_service_level: float = 0.95,
) -> dict:
    """
    Solve a multi-SKU inventory optimization via linear programming.

    Minimizes total cost = sum(holding_cost + stockout_penalty) for each SKU,
    subject to warehouse capacity and service level constraints.

    Args:
        skus: List of dicts with keys: sku_id, mean_daily, std_daily,
              lead_time_days, unit_cost, holding_cost_pct
        warehouse_capacity: Total units the warehouse can hold.
        min_service_level: Minimum fill rate (0-1).

    Returns:
        Dict with optimization status, total cost, and per-SKU recommendations.
    """
    prob = LpProblem("MRO_Inventory_Optimization", LpMinimize)

    order_qty = {}
    safety_stk = {}
    stockout_units = {}

    STOCKOUT_PENALTY = 50.0
    SERVICE_Z = 1.65 if min_service_level >= 0.95 else 1.28

    for sku in skus:
        sid = sku["sku_id"]
        max_order = max(int(sku["mean_daily"] * 365), 1)

        order_qty[sid] = LpVariable(f"order_{sid}", lowBound=1, upBound=max_order, cat="Continuous")
        safety_stk[sid] = LpVariable(f"safety_{sid}", lowBound=0, cat="Continuous")
        stockout_units[sid] = LpVariable(f"stockout_{sid}", lowBound=0, cat="Continuous")

    objective_terms = []
    for sku in skus:
        sid = sku["sku_id"]
        holding_rate = sku["unit_cost"] * sku["holding_cost_pct"]
        avg_inv = order_qty[sid] * 0.5 + safety_stk[sid]
        objective_terms.append(avg_inv * holding_rate)
        objective_terms.append(stockout_units[sid] * STOCKOUT_PENALTY)

    prob += lpSum(objective_terms), "Total_Inventory_Cost"

    total_stock = []
    for sku in skus:
        sid = sku["sku_id"]
        total_stock.append(order_qty[sid] + safety_stk[sid])

    prob += lpSum(total_stock) <= warehouse_capacity, "Warehouse_Capacity"

    for sku in skus:
        sid = sku["sku_id"]
        lt = sku["lead_time_days"]
        ss_min = SERVICE_Z * sku["std_daily"] * np.sqrt(lt)
        prob += safety_stk[sid] >= ss_min, f"Min_Safety_Stock_{sid}"

        lt_demand = sku["mean_daily"] * lt
        prob += stockout_units[sid] >= lt_demand - (order_qty[sid] + safety_stk[sid]), f"Stockout_Def_{sid}"

    prob.solve()

    recommendations = []
    total_cost = 0.0

    for sku in skus:
        sid = sku["sku_id"]
        oq = value(order_qty[sid])
        ss = value(safety_stk[sid])
        so = value(stockout_units[sid])

        rop = sku["mean_daily"] * sku["lead_time_days"] + ss
        holding_cost = (oq * 0.5 + ss) * sku["unit_cost"] * sku["holding_cost_pct"]
        stockout_cost = so * STOCKOUT_PENALTY

        rec = {
            "sku_id": sid,
            "order_quantity": round(oq),
            "safety_stock": round(ss),
            "reorder_point": round(rop),
            "holding_cost": round(holding_cost, 2),
            "stockout_cost": round(stockout_cost, 2),
            "total_sku_cost": round(holding_cost + stockout_cost, 2),
            "mean_daily_demand": round(sku["mean_daily"], 1),
            "lead_time_days": sku["lead_time_days"],
        }
        recommendations.append(rec)
        total_cost += holding_cost + stockout_cost

    return {
        "status": LpStatus[prob.status],
        "total_cost": round(total_cost, 2),
        "warehouse_utilization": round(
            sum(value(order_qty[s["sku_id"]]) + value(safety_stk[s["sku_id"]]) for s in skus)
            / warehouse_capacity * 100, 1
        ),
        "recommendations": pd.DataFrame(recommendations),
    }


def optimize_single_sku(
    mean_daily: float,
    std_daily: float,
    lead_time_days: int,
    unit_cost: float,
    holding_cost_pct: float = 0.25,
    service_level: float = 0.95,
) -> dict:
    """Quick single-SKU optimization without LP (closed-form)."""
    z = 1.65 if service_level >= 0.95 else 1.28
    ss = safety_stock(std_daily, lead_time_days, z)
    rop = reorder_point(mean_daily, lead_time_days, ss)
    eoq = economic_order_quantity(mean_daily * 365, unit_cost=unit_cost, holding_cost_pct=holding_cost_pct)

    avg_inv = eoq / 2 + ss
    holding = avg_inv * unit_cost * holding_cost_pct

    return {
        "safety_stock": round(ss),
        "reorder_point": round(rop),
        "order_quantity": round(eoq),
        "avg_inventory": round(avg_inv),
        "annual_holding_cost": round(holding, 2),
    }
