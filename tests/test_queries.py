"""Tests for DuckDB SQL query layer."""

import pytest
from src.utils.data_generator import generate_and_save
from src.etl.queries import DemandQueryEngine


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("data"))
    generate_and_save(output_dir=data_dir, n_skus=5, seed=42)
    eng = DemandQueryEngine(data_dir=data_dir)
    yield eng
    eng.close()


class TestDemandQueryEngine:
    def test_raw_query(self, engine):
        result = engine.query("SELECT COUNT(*) AS cnt FROM demand")
        assert result["cnt"].iloc[0] > 0

    def test_top_skus(self, engine):
        result = engine.top_skus_by_volume(n=3)
        assert len(result) == 3
        assert "total_demand" in result.columns
        assert result["total_demand"].is_monotonic_decreasing

    def test_monthly_summary(self, engine):
        result = engine.monthly_demand_summary()
        assert len(result) > 0
        assert "total_revenue" in result.columns

    def test_category_performance(self, engine):
        result = engine.category_performance()
        assert len(result) > 0
        assert "sku_count" in result.columns

    def test_weekend_vs_weekday(self, engine):
        result = engine.weekend_vs_weekday()
        assert len(result) == 2
        day_types = set(result["day_type"].tolist())
        assert day_types == {"Weekend", "Weekday"}
