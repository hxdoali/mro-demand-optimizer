"""
Product clustering via TF-IDF on product descriptions.

Groups similar MRO products to enable cluster-level forecasting
and inventory strategy segmentation.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cluster_products(
    catalog: pd.DataFrame,
    n_clusters: int = 8,
    text_col: str = "description",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cluster products by TF-IDF vectors of their descriptions combined
    with numeric attributes (unit_cost, lead_time_days).
    """
    tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    text_features = tfidf.fit_transform(catalog[text_col]).toarray()

    numeric_cols = ["unit_cost", "lead_time_days"]
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(catalog[numeric_cols].values)

    combined = np.hstack([text_features, numeric_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    catalog = catalog.copy()
    catalog["cluster_id"] = kmeans.fit_predict(combined)

    return catalog


def get_cluster_embeddings(
    catalog: pd.DataFrame,
    text_col: str = "description",
    n_components: int = 2,
) -> pd.DataFrame:
    """Compute 2D PCA embeddings for visualization."""
    tfidf = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    text_features = tfidf.fit_transform(catalog[text_col]).toarray()

    numeric_cols = ["unit_cost", "lead_time_days"]
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(catalog[numeric_cols].values)

    combined = np.hstack([text_features, numeric_features])

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(combined)

    result = catalog[["sku_id", "category", "cluster_id"]].copy()
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]
    return result
