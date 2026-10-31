# hierarchical.py - hierarchical agglomerative clustering and dendrogram

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

VIEWPOINT = "Merge tree: builds a bottom-up dendrogram by iteratively joining the most similar clusters using Ward's minimum variance."


def compute_wcss(x_mat, labels):
    total = 0.0
    for c in np.unique(labels):
        pts = x_mat[labels == c]
        if len(pts) > 0:
            total += np.sum((pts - pts.mean(axis=0)) ** 2)
    return float(total)


def train_hierarchical(x_scaled, k=5):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(x_scaled)
    wcss = compute_wcss(x_scaled, model.labels_)
    sil = float(silhouette_score(x_scaled, model.labels_))
    z_linkage = linkage(x_scaled, method="ward")
    return model, model.labels_, wcss, sil, z_linkage


def plot_dendrogram_short(z_linkage):
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    dendrogram(z_linkage, p=20, truncate_mode="lastp", ax=ax, leaf_rotation=90, leaf_font_size=7)
    ax.set_title("Shortened Dendrogram (Last 20 Merges)", fontsize=8, fontweight="bold")
    ax.set_xlabel("Cluster size", fontsize=8)
    ax.set_ylabel("Ward Distance", fontsize=8)
    fig.tight_layout()
    return fig


def plot_dendrogram_full(z_linkage, labels):
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    clean_labels = [re.sub(r"[^\x00-\x7F]+", "", str(c)) for c in labels]
    dendrogram(z_linkage, labels=clean_labels, leaf_rotation=90, leaf_font_size=4, ax=ax)
    ax.set_title("Full Dendrogram (All Commodities)", fontsize=8, fontweight="bold")
    fig.tight_layout()
    return fig
