"""Tests for demand forecasting module."""

import pytest
from src.utils.data_generator import generate_product_catalog, generate_demand_history
from src.models.forecaster import prepare_prophet_data


@pytest.fixture
def sample_data():
    catalog = generate_product_catalog(n_skus=3, seed=42)
    demand = generate_demand_history(
        catalog, start_date="2024-01-01", end_date="2024-06-30", seed=42
    )
    return catalog, demand


class TestForecaster:
    def test_prepare_prophet_data(self, sample_data):
        _, demand = sample_data
        sku_id = demand["sku_id"].iloc[0]
        result = prepare_prophet_data(demand, sku_id)
        assert list(result.columns) == ["ds", "y"]
        assert result["ds"].is_monotonic_increasing
        assert len(result) > 0

    def test_prepare_prophet_data_missing_sku(self, sample_data):
        _, demand = sample_data
        result = prepare_prophet_data(demand, "NONEXISTENT-SKU")
        assert len(result) == 0
