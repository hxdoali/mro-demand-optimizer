"""
Airflow DAG for the MRO demand forecasting and inventory optimization pipeline.

Schedule: Daily at 6:00 AM UTC
Pipeline: Ingest -> Transform -> Cluster -> Forecast -> Optimize -> Serve
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mle-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "mro_demand_optimization_pipeline",
    default_args=default_args,
    description="End-to-end MRO demand forecasting and inventory optimization",
    schedule_interval="0 6 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mro", "forecasting", "optimization", "ml"],
)


def ingest_data(**kwargs):
    from src.etl.ingest import load_pandas
    data = load_pandas("data/synthetic")
    print(f"Ingested {len(data['catalog'])} SKUs, {len(data['demand'])} demand records")
    return {"catalog_count": len(data["catalog"]), "demand_count": len(data["demand"])}


def transform_features(**kwargs):
    from src.etl.ingest import load_pandas
    from src.etl.transform import build_feature_table
    data = load_pandas("data/synthetic")
    features = build_feature_table(data["demand"], data["catalog"])
    features.to_parquet("data/synthetic/feature_table.parquet", index=False)
    print(f"Built feature table with {len(features)} rows, {len(features.columns)} columns")


def cluster_products_task(**kwargs):
    from src.etl.ingest import load_pandas
    from src.etl.cluster import cluster_products
    data = load_pandas("data/synthetic")
    clustered = cluster_products(data["catalog"])
    clustered.to_parquet("data/synthetic/clustered_catalog.parquet", index=False)
    print(f"Clustered {len(clustered)} products into {clustered['cluster_id'].nunique()} clusters")


def run_forecasts(**kwargs):
    from src.etl.ingest import load_pandas
    from src.models.forecaster import forecast_all_skus
    data = load_pandas("data/synthetic")
    sku_ids = data["demand"]["sku_id"].unique()[:10]
    results = forecast_all_skus(data["demand"], sku_ids=list(sku_ids), horizon_days=30)
    print(f"Generated forecasts for {len(results)} SKUs")


def run_optimization(**kwargs):
    from src.etl.ingest import load_pandas
    from src.etl.cluster import cluster_products
    from src.models.optimizer import compute_demand_stats, optimize_inventory_lp
    data = load_pandas("data/synthetic")
    catalog = cluster_products(data["catalog"])

    sku_inputs = []
    for _, row in catalog.head(20).iterrows():
        stats = compute_demand_stats(data["demand"], row["sku_id"])
        sku_inputs.append({
            "sku_id": row["sku_id"],
            "mean_daily": stats["mean_daily"],
            "std_daily": stats["std_daily"],
            "lead_time_days": int(row["lead_time_days"]),
            "unit_cost": float(row["unit_cost"]),
            "holding_cost_pct": float(row["holding_cost_pct"]),
        })

    result = optimize_inventory_lp(sku_inputs)
    print(f"Optimization: {result['status']}, total cost: ${result['total_cost']:,.0f}")


ingest = PythonOperator(task_id="ingest_data", python_callable=ingest_data, dag=dag)
transform = PythonOperator(task_id="transform_features", python_callable=transform_features, dag=dag)
cluster = PythonOperator(task_id="cluster_products", python_callable=cluster_products_task, dag=dag)
forecast = PythonOperator(task_id="run_forecasts", python_callable=run_forecasts, dag=dag)
optimize = PythonOperator(task_id="run_optimization", python_callable=run_optimization, dag=dag)

health_check = BashOperator(
    task_id="api_health_check",
    bash_command="curl -f http://localhost:8000/health || echo 'API not running, skipping health check'",
    dag=dag,
)

ingest >> transform >> cluster >> [forecast, optimize] >> health_check
