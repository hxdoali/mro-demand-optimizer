"""Tests for inventory optimization module."""

import pytest
import numpy as np
from src.models.optimizer import (
    safety_stock,
    reorder_point,
    economic_order_quantity,
    optimize_single_sku,
    optimize_inventory_lp,
)


class TestClosedFormOptimization:
    def test_safety_stock_positive(self):
        ss = safety_stock(std_daily_demand=10, lead_time_days=7, service_level_z=1.65)
        assert ss > 0
        assert ss == pytest.approx(1.65 * 10 * np.sqrt(7), rel=1e-6)

    def test_safety_stock_zero_std(self):
        ss = safety_stock(std_daily_demand=0, lead_time_days=7)
        assert ss == 0

    def test_reorder_point(self):
        rop = reorder_point(mean_daily_demand=50, lead_time_days=7, safety_stock_units=30)
        assert rop == 50 * 7 + 30

    def test_eoq(self):
        eoq = economic_order_quantity(
            annual_demand=10000, ordering_cost=25, unit_cost=10, holding_cost_pct=0.25
        )
        assert eoq > 0
        expected = np.sqrt(2 * 10000 * 25 / (10 * 0.25))
        assert eoq == pytest.approx(expected, rel=1e-6)

    def test_optimize_single_sku(self):
        result = optimize_single_sku(
            mean_daily=50, std_daily=15, lead_time_days=7,
            unit_cost=20, holding_cost_pct=0.25, service_level=0.95,
        )
        assert "safety_stock" in result
        assert "reorder_point" in result
        assert "order_quantity" in result
        assert result["safety_stock"] > 0
        assert result["reorder_point"] > result["safety_stock"]


class TestLPOptimization:
    def test_optimize_lp_feasible(self):
        skus = [
            {"sku_id": "A", "mean_daily": 50, "std_daily": 10,
             "lead_time_days": 5, "unit_cost": 10, "holding_cost_pct": 0.25},
            {"sku_id": "B", "mean_daily": 30, "std_daily": 8,
             "lead_time_days": 7, "unit_cost": 25, "holding_cost_pct": 0.20},
        ]
        result = optimize_inventory_lp(skus, warehouse_capacity=100000)
        assert result["status"] == "Optimal"
        assert len(result["recommendations"]) == 2
        assert result["total_cost"] > 0

    def test_optimize_lp_capacity_constraint(self):
        skus = [
            {"sku_id": "A", "mean_daily": 100, "std_daily": 20,
             "lead_time_days": 10, "unit_cost": 5, "holding_cost_pct": 0.25},
        ]
        result = optimize_inventory_lp(skus, warehouse_capacity=500)
        assert result["status"] == "Optimal"
        recs = result["recommendations"]
        total_stock = recs["order_quantity"].iloc[0] + recs["safety_stock"].iloc[0]
        assert total_stock <= 500
