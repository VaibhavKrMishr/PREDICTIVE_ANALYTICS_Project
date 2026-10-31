# kmeans.py - k-means clustering model and elbow curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

VIEWPOINT = "Round equal-size groups: partitions commodities into compact spherical clusters around geometric centroids."


def compute_wcss(x_mat, labels):
    total = 0.0
    for c in np.unique(labels):
        pts = x_mat[labels == c]
        if len(pts) > 0:
            total += np.sum((pts - pts.mean(axis=0)) ** 2)
    return float(total)


def train_kmeans(x_scaled, k=5, seed=42):
    # compute elbow curve for k=2..8
    elbow_rows = []
    for k_val in range(2, 9):
        km_tmp = KMeans(n_clusters=k_val, n_init=10, random_state=seed).fit(x_scaled)
        elbow_rows.append({"k": k_val, "KMeans WCSS": float(km_tmp.inertia_)})
    elbow_df = pd.DataFrame(elbow_rows)

    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x_scaled)
    wcss = float(model.inertia_)
    sil = float(silhouette_score(x_scaled, model.labels_))

    return model, model.labels_, wcss, sil, elbow_df


def plot_elbow_curve(elbow_df):
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.plot(elbow_df["k"], elbow_df["KMeans WCSS"], "o-", color="#059669", markersize=4, lw=1.5)
    ax.set_title("Elbow Method (WCSS vs k)", fontsize=8, fontweight="bold")
    ax.set_xlabel("k (Number of clusters)", fontsize=8)
    ax.set_ylabel("WCSS", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
