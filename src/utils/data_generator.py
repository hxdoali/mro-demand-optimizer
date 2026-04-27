"""
Synthetic MRO demand data generator.

Produces realistic maintenance, repair, and operating supply transaction data
with trend, seasonality, and noise patterns that mirror industrial distribution.
"""

import numpy as np
import pandas as pd
from pathlib import Path

MRO_CATEGORIES = [
    "Fasteners", "Pipe Fittings", "Electrical", "Safety Equipment",
    "Lubricants", "Bearings", "Motors", "Pumps", "HVAC Filters",
    "Abrasives", "Hand Tools", "Power Tools", "Welding Supplies",
    "Plumbing", "Janitorial",
]

PRODUCT_TEMPLATES = {
    "Fasteners": [
        ("HEX-BOLT-{sz}", "Hex bolt, Grade 5, zinc plated, {sz} diameter, ASTM A325"),
        ("SS-SCREW-{sz}", "Stainless steel machine screw, Phillips head, {sz}, 18-8 SS"),
    ],
    "Pipe Fittings": [
        ("PIPE-ELB-{sz}", "Galvanized pipe elbow 90-deg, {sz}, Schedule 40, ASTM A105"),
        ("PIPE-TEE-{sz}", "Malleable iron pipe tee, {sz}, Class 150, threaded"),
    ],
    "Electrical": [
        ("WIRE-THHN-{sz}", "THHN copper building wire, {sz} AWG, 600V, stranded"),
        ("BREAKER-{sz}", "Miniature circuit breaker, {sz}A, 120/240V, bolt-on"),
    ],
    "Safety Equipment": [
        ("GLOVE-NIT-{sz}", "Nitrile disposable gloves, {sz}, 4 mil, powder-free, box/100"),
        ("EARPLUGS-{sz}", "Disposable foam earplugs, NRR {sz}dB, uncorded, box/200"),
    ],
    "Lubricants": [
        ("GREASE-MP-{sz}", "Multi-purpose lithium grease, NLGI Grade 2, {sz} oz cartridge"),
        ("OIL-HYD-{sz}", "Hydraulic oil ISO {sz}, AW anti-wear, 5 gal pail"),
    ],
    "Bearings": [
        ("BEAR-BALL-{sz}", "Deep groove ball bearing, {sz}mm bore, sealed, C3 clearance"),
        ("BEAR-ROLL-{sz}", "Tapered roller bearing, {sz}mm bore, single row"),
    ],
    "Motors": [
        ("MOTOR-AC-{sz}", "AC induction motor, {sz} HP, 1750 RPM, TEFC, 230/460V"),
    ],
    "Pumps": [
        ("PUMP-CENT-{sz}", "Centrifugal pump, {sz} GPM, cast iron, 1-1/2 in discharge"),
    ],
    "HVAC Filters": [
        ("FILTER-HVAC-{sz}", "Pleated HVAC air filter, {sz}, MERV 8, 12-pack"),
        ("FILTER-HEPA-{sz}", "HEPA filter panel, {sz}, 99.97% efficiency"),
    ],
    "Abrasives": [
        ("DISC-GRIND-{sz}", "Grinding disc, {sz} in, Type 27, aluminum oxide, 24 grit"),
    ],
    "Hand Tools": [
        ("WRENCH-COMB-{sz}", "Combination wrench, {sz} mm, chrome vanadium steel"),
    ],
    "Power Tools": [
        ("DRILL-CORD-{sz}", "Cordless drill/driver, {sz}V, 1/2 in chuck, brushless"),
    ],
    "Welding Supplies": [
        ("WELD-ROD-{sz}", "Welding electrode, E6013, {sz} in diameter, 10 lb box"),
    ],
    "Plumbing": [
        ("VALVE-BALL-{sz}", "Brass ball valve, {sz} in, full port, 600 WOG"),
    ],
    "Janitorial": [
        ("TRASH-BAG-{sz}", "Heavy-duty trash bags, {sz} gal, 1.5 mil, black, box/100"),
    ],
}

SIZES = ["S", "M", "L", "XL", "1/4", "1/2", "3/4", "1", "2", "10", "12",
         "14", "20", "25", "30", "32", "40", "46", "50", "55"]


def generate_product_catalog(n_skus: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate a catalog of MRO products with descriptions."""
    rng = np.random.default_rng(seed)
    products = []

    for i in range(n_skus):
        category = MRO_CATEGORIES[i % len(MRO_CATEGORIES)]
        templates = PRODUCT_TEMPLATES[category]
        sku_tmpl, desc_tmpl = templates[i % len(templates)]
        size = rng.choice(SIZES)

        sku_id = sku_tmpl.format(sz=size)
        description = desc_tmpl.format(sz=size)
        unit_cost = round(rng.uniform(1.50, 500.0), 2)
        lead_time_days = int(rng.integers(3, 30))
        holding_cost_pct = round(rng.uniform(0.15, 0.35), 2)

        products.append({
            "sku_id": f"SKU-{i:04d}-{sku_id}",
            "category": category,
            "description": description,
            "unit_cost": unit_cost,
            "lead_time_days": lead_time_days,
            "holding_cost_pct": holding_cost_pct,
        })

    return pd.DataFrame(products)


def generate_demand_history(
    catalog: pd.DataFrame,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate daily demand for each SKU with trend, weekly seasonality,
    annual seasonality, and random noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []

    for _, product in catalog.iterrows():
        base_demand = rng.uniform(5, 100)
        trend_slope = rng.uniform(-0.02, 0.05)
        annual_amplitude = rng.uniform(0.1, 0.4) * base_demand
        weekly_amplitude = rng.uniform(0.05, 0.15) * base_demand
        noise_std = rng.uniform(0.1, 0.25) * base_demand

        for i, date in enumerate(dates):
            trend = trend_slope * i
            annual_season = annual_amplitude * np.sin(2 * np.pi * i / 365.25)
            weekly_season = weekly_amplitude * np.sin(2 * np.pi * date.dayofweek / 7)
            # Reduced weekend demand for industrial MRO
            weekend_factor = 0.3 if date.dayofweek >= 5 else 1.0
            noise = rng.normal(0, noise_std)

            demand = max(0, (base_demand + trend + annual_season + weekly_season + noise) * weekend_factor)

            records.append({
                "date": date,
                "sku_id": product["sku_id"],
                "demand": round(demand),
                "unit_price": round(product["unit_cost"] * rng.uniform(1.2, 1.8), 2),
            })

    return pd.DataFrame(records)


def generate_and_save(
    output_dir: str = "data/synthetic",
    n_skus: int = 50,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data and save to parquet files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = generate_product_catalog(n_skus=n_skus, seed=seed)
    demand = generate_demand_history(catalog, seed=seed)

    catalog.to_parquet(out / "product_catalog.parquet", index=False)
    demand.to_parquet(out / "demand_history.parquet", index=False)

    print(f"Generated {len(catalog)} SKUs, {len(demand)} demand records")
    print(f"Saved to {out.resolve()}")

    return catalog, demand


if __name__ == "__main__":
    generate_and_save()
