"""Tests for ETL pipeline components."""

import pytest
import pandas as pd
import numpy as np
from src.utils.data_generator import generate_product_catalog, generate_demand_history
from src.etl.transform import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    build_feature_table,
)
from src.etl.cluster import cluster_products


@pytest.fixture
def sample_catalog():
    return generate_product_catalog(n_skus=5, seed=42)


@pytest.fixture
def sample_demand(sample_catalog):
    return generate_demand_history(
        sample_catalog, start_date="2024-01-01", end_date="2024-03-31", seed=42
    )


class TestTransform:
    def test_calendar_features(self, sample_demand):
        result = add_calendar_features(sample_demand)
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "month" in result.columns
        assert result["is_weekend"].isin([0, 1]).all()

    def test_lag_features(self, sample_demand):
        result = add_lag_features(sample_demand)
        assert "demand_lag_1" in result.columns
        assert "demand_lag_7" in result.columns
        assert "demand_lag_28" in result.columns

    def test_rolling_features(self, sample_demand):
        result = add_rolling_features(sample_demand)
        assert "demand_roll_mean_7" in result.columns
        assert "demand_roll_std_30" in result.columns

    def test_build_feature_table(self, sample_demand, sample_catalog):
        result = build_feature_table(sample_demand, sample_catalog)
        assert "category" in result.columns
        assert "unit_cost" in result.columns
        assert "demand_lag_28" in result.columns
        assert result["demand_lag_28"].notna().all()


class TestCluster:
    def test_cluster_products(self, sample_catalog):
        result = cluster_products(sample_catalog, n_clusters=3)
        assert "cluster_id" in result.columns
        assert result["cluster_id"].nunique() <= 3
        assert len(result) == len(sample_catalog)
