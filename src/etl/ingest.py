"""
Data ingestion layer.

Loads raw product catalog and demand history from parquet files,
with support for both PySpark and pandas backends.
"""

from pathlib import Path
import pandas as pd

try:
    from pyspark.sql import SparkSession, DataFrame as SparkDF
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def load_pandas(data_dir: str = "data/synthetic") -> dict[str, pd.DataFrame]:
    """Load data as pandas DataFrames."""
    p = Path(data_dir)
    return {
        "catalog": pd.read_parquet(p / "product_catalog.parquet"),
        "demand": pd.read_parquet(p / "demand_history.parquet"),
    }


def load_spark(data_dir: str = "data/synthetic", app_name: str = "MRO-ETL") -> dict:
    """Load data as Spark DataFrames for scalable processing."""
    if not SPARK_AVAILABLE:
        raise ImportError("PySpark is not installed. Install with: pip install pyspark")

    spark = SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    p = Path(data_dir)
    return {
        "catalog": spark.read.parquet(str(p / "product_catalog.parquet")),
        "demand": spark.read.parquet(str(p / "demand_history.parquet")),
        "spark": spark,
    }


def load(data_dir: str = "data/synthetic", backend: str = "pandas") -> dict:
    """
    Load data using the specified backend.

    Args:
        data_dir: Path to directory containing parquet files.
        backend: "pandas" or "spark".
    """
    if backend == "spark":
        return load_spark(data_dir)
    return load_pandas(data_dir)
