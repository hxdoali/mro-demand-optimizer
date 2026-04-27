"""
Streamlit dashboard for MRO demand forecasting and inventory optimization.

Provides interactive views for stakeholders to explore forecasts,
optimization recommendations, and product clustering.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.etl.ingest import load_pandas
from src.etl.cluster import cluster_products, get_cluster_embeddings
from src.models.forecaster import forecast_sku, get_forecast_summary
from src.models.optimizer import compute_demand_stats, optimize_inventory_lp
from src.utils.metrics import forecast_summary

DATA_DIR = "data/synthetic"


@st.cache_data
def load_data():
    data = load_pandas(DATA_DIR)
    catalog = cluster_products(data["catalog"])
    return catalog, data["demand"]


def render_header():
    st.set_page_config(page_title="MRO Inventory Optimizer", layout="wide")
    st.title("MRO Demand Forecasting & Inventory Optimization")
    st.caption("Scalable ML + Operations Research pipeline for MRO supply chain decisions")


def render_overview(catalog: pd.DataFrame, demand: pd.DataFrame):
    st.header("Pipeline Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs", len(catalog))
    col2.metric("Product Categories", catalog["category"].nunique())
    col3.metric("Demand Records", f"{len(demand):,}")
    date_range = f"{demand['date'].min().date()} to {demand['date'].max().date()}"
    col4.metric("Date Range", date_range)

    st.subheader("Demand by Category")
    cat_demand = (
        demand.merge(catalog[["sku_id", "category"]], on="sku_id")
        .groupby("category")["demand"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    fig = px.bar(cat_demand, x="demand", y="category", orientation="h",
                 labels={"demand": "Total Units", "category": "Category"})
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_tab(catalog: pd.DataFrame, demand: pd.DataFrame):
    st.header("Demand Forecasting")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_sku = st.selectbox("Select SKU", catalog["sku_id"].tolist())
        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)

    with st.spinner(f"Training forecast model for {selected_sku}..."):
        result = forecast_sku(demand, selected_sku, horizon_days=horizon)

    fc = result["forecast"]
    train = result["train"]
    train_end = train["ds"].max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], mode="lines",
                             name="Historical", line=dict(color="#636EFA")))

    future = fc[fc["ds"] > train_end]
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines",
                             name="Forecast", line=dict(color="#EF553B")))
    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"][::-1]]),
        y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(239,85,59,0.15)",
        line=dict(width=0), name="Confidence Interval",
    ))

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Daily Demand (units)",
        height=400, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(result["test"]) > 0:
        test = result["test"]
        test_fc = fc[fc["ds"].isin(test["ds"])]
        if len(test_fc) > 0:
            merged = test.merge(test_fc[["ds", "yhat"]], on="ds")
            metrics = forecast_summary(merged["y"], merged["yhat"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAPE", f"{metrics['mape']}%")
            mc2.metric("RMSE", f"{metrics['rmse']}")
            mc3.metric("Test Periods", metrics["n_periods"])

    summary = get_forecast_summary(result)
    st.subheader("Forecast Table")
    st.dataframe(summary[["date", "predicted_demand", "lower_bound", "upper_bound"]], use_container_width=True)


def render_optimization_tab(catalog: pd.DataFrame, demand: pd.DataFrame):
    st.header("Inventory Optimization")

    col1, col2 = st.columns(2)
    with col1:
        service_level = st.slider("Target Service Level", 0.90, 0.99, 0.95, 0.01)
    with col2:
        warehouse_cap = st.number_input("Warehouse Capacity (units)", 10000, 500000, 100000, 10000)

    sku_subset = catalog["sku_id"].tolist()[:20]  # optimize first 20 for speed

    with st.spinner("Running LP optimization..."):
        sku_inputs = []
        for sku_id in sku_subset:
            stats = compute_demand_stats(demand, sku_id)
            cat_row = catalog[catalog["sku_id"] == sku_id].iloc[0]
            sku_inputs.append({
                "sku_id": sku_id,
                "mean_daily": stats["mean_daily"],
                "std_daily": stats["std_daily"],
                "lead_time_days": int(cat_row["lead_time_days"]),
                "unit_cost": float(cat_row["unit_cost"]),
                "holding_cost_pct": float(cat_row["holding_cost_pct"]),
            })

        result = optimize_inventory_lp(sku_inputs, warehouse_cap, service_level)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Optimization Status", result["status"])
    mc2.metric("Total Annual Cost", f"${result['total_cost']:,.0f}")
    mc3.metric("Warehouse Utilization", f"{result['warehouse_utilization']}%")

    st.subheader("SKU Recommendations")
    recs = result["recommendations"].sort_values("total_sku_cost", ascending=False)
    st.dataframe(recs, use_container_width=True)

    st.subheader("Cost Distribution")
    fig = px.bar(
        recs.head(15),
        x="sku_id", y=["holding_cost", "stockout_cost"],
        barmode="stack",
        labels={"value": "Annual Cost ($)", "sku_id": "SKU"},
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_cluster_tab(catalog: pd.DataFrame):
    st.header("Product Clustering")
    st.caption("Products grouped by TF-IDF on descriptions + numeric attributes via KMeans")

    embeddings = get_cluster_embeddings(catalog)

    fig = px.scatter(
        embeddings, x="pca_x", y="pca_y",
        color=embeddings["cluster_id"].astype(str),
        hover_data=["sku_id", "category"],
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2", "color": "Cluster"},
    )
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Composition")
    cluster_cats = catalog.groupby(["cluster_id", "category"]).size().reset_index(name="count")
    fig2 = px.sunburst(cluster_cats, path=["cluster_id", "category"], values="count")
    fig2.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


def main():
    render_header()
    catalog, demand = load_data()

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Demand Forecast", "Inventory Optimization", "Product Clusters"])

    with tab1:
        render_overview(catalog, demand)
    with tab2:
        render_forecast_tab(catalog, demand)
    with tab3:
        render_optimization_tab(catalog, demand)
    with tab4:
        render_cluster_tab(catalog)


if __name__ == "__main__":
    main()
