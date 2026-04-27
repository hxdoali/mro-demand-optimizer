"""
SQL query layer for demand analytics using DuckDB.

Provides SQL-based access to demand data for ad-hoc analysis,
aggregation, and reporting without requiring a separate database server.
"""

import duckdb
import pandas as pd
from pathlib import Path


class DemandQueryEngine:
    """SQL interface over parquet-backed demand and catalog data."""

    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect(":memory:")
        self._register_tables()

    def _register_tables(self):
        catalog_path = str(self.data_dir / "product_catalog.parquet")
        demand_path = str(self.data_dir / "demand_history.parquet")

        self.conn.execute(f"""
            CREATE OR REPLACE VIEW catalog AS
            SELECT * FROM read_parquet('{catalog_path}')
        """)
        self.conn.execute(f"""
            CREATE OR REPLACE VIEW demand AS
            SELECT * FROM read_parquet('{demand_path}')
        """)

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def daily_demand_by_sku(self, sku_id: str, limit: int = 90) -> pd.DataFrame:
        return self.query(f"""
            SELECT date, demand, unit_price
            FROM demand
            WHERE sku_id = '{sku_id}'
            ORDER BY date DESC
            LIMIT {limit}
        """)

    def top_skus_by_volume(self, n: int = 10) -> pd.DataFrame:
        return self.query(f"""
            SELECT
                d.sku_id,
                c.category,
                c.description,
                SUM(d.demand) AS total_demand,
                ROUND(AVG(d.demand), 1) AS avg_daily_demand,
                ROUND(STDDEV(d.demand), 1) AS std_daily_demand
            FROM demand d
            JOIN catalog c ON d.sku_id = c.sku_id
            GROUP BY d.sku_id, c.category, c.description
            ORDER BY total_demand DESC
            LIMIT {n}
        """)

    def monthly_demand_summary(self) -> pd.DataFrame:
        return self.query("""
            SELECT
                DATE_TRUNC('month', date) AS month,
                COUNT(DISTINCT sku_id) AS active_skus,
                SUM(demand) AS total_demand,
                ROUND(AVG(demand), 1) AS avg_daily_demand,
                ROUND(SUM(demand * unit_price), 2) AS total_revenue
            FROM demand
            GROUP BY DATE_TRUNC('month', date)
            ORDER BY month
        """)

    def category_performance(self) -> pd.DataFrame:
        return self.query("""
            SELECT
                c.category,
                COUNT(DISTINCT d.sku_id) AS sku_count,
                SUM(d.demand) AS total_demand,
                ROUND(AVG(d.demand), 1) AS avg_daily_demand,
                ROUND(SUM(d.demand * d.unit_price), 2) AS total_revenue,
                ROUND(AVG(c.lead_time_days), 0) AS avg_lead_time
            FROM demand d
            JOIN catalog c ON d.sku_id = c.sku_id
            GROUP BY c.category
            ORDER BY total_revenue DESC
        """)

    def demand_volatility(self, min_cv: float = 0.3) -> pd.DataFrame:
        """Find high-volatility SKUs (coefficient of variation > threshold)."""
        return self.query(f"""
            SELECT
                d.sku_id,
                c.category,
                ROUND(AVG(d.demand), 1) AS mean_demand,
                ROUND(STDDEV(d.demand), 1) AS std_demand,
                ROUND(STDDEV(d.demand) / NULLIF(AVG(d.demand), 0), 3) AS coeff_variation
            FROM demand d
            JOIN catalog c ON d.sku_id = c.sku_id
            GROUP BY d.sku_id, c.category
            HAVING STDDEV(d.demand) / NULLIF(AVG(d.demand), 0) > {min_cv}
            ORDER BY coeff_variation DESC
        """)

    def weekend_vs_weekday(self) -> pd.DataFrame:
        return self.query("""
            SELECT
                CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
                ROUND(AVG(demand), 1) AS avg_demand,
                SUM(demand) AS total_demand,
                COUNT(*) AS record_count
            FROM demand
            GROUP BY day_type
        """)

    def stockout_risk_skus(self, threshold_days: int = 7) -> pd.DataFrame:
        """Identify SKUs where recent demand exceeds historical avg by a margin."""
        return self.query(f"""
            WITH recent AS (
                SELECT sku_id, AVG(demand) AS recent_avg
                FROM demand
                WHERE date >= (SELECT MAX(date) - INTERVAL '{threshold_days} days' FROM demand)
                GROUP BY sku_id
            ),
            historical AS (
                SELECT sku_id, AVG(demand) AS hist_avg, STDDEV(demand) AS hist_std
                FROM demand
                GROUP BY sku_id
            )
            SELECT
                r.sku_id,
                c.category,
                ROUND(r.recent_avg, 1) AS recent_avg_demand,
                ROUND(h.hist_avg, 1) AS historical_avg_demand,
                ROUND((r.recent_avg - h.hist_avg) / NULLIF(h.hist_std, 0), 2) AS z_score,
                c.lead_time_days
            FROM recent r
            JOIN historical h ON r.sku_id = h.sku_id
            JOIN catalog c ON r.sku_id = c.sku_id
            WHERE (r.recent_avg - h.hist_avg) / NULLIF(h.hist_std, 0) > 1.5
            ORDER BY z_score DESC
        """)

    def close(self):
        self.conn.close()
