"""
models.py — Clustering Algorithms & Evaluation Metrics Module

This module provides functions for running clustering algorithms (K-Means,
DBSCAN) on preprocessed RFM features and evaluating the quality of the
resulting clusters.

Key Exports:
    - plot_elbow(data, max_k):          Plots the Elbow curve (inertia vs. k)
                                         to help determine optimal K for K-Means.
    - run_kmeans(data, k):              Runs K-Means clustering and returns labels.
    - run_dbscan(data, eps, min_samples): Runs DBSCAN clustering and returns labels.
    - calculate_silhouette(data, labels): Computes the Silhouette Score for
                                          cluster quality evaluation.
"""

# =============================================================================
# Imports
# =============================================================================

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_elbow(data, max_k=10):
    """
    Plot the Elbow curve to find the optimal number of clusters for K-Means.

    The Elbow Method plots the total within-cluster sum of squares (inertia)
    against different values of k.  The "elbow" point — where inertia starts
    to decrease more slowly — suggests a good balance between model complexity
    and fit.

    Args:
        data (array-like): Preprocessed feature matrix (e.g., scaled RFM values).
            Should be a 2D array or DataFrame with shape (n_samples, n_features).
        max_k (int): Maximum number of clusters to evaluate. Defaults to 10.
            The function tests k = 1, 2, ..., max_k.

    Returns:
        None: Displays the Elbow plot inline.
    """
    inertia = []
    k_range = range(1, max_k + 1)

    # Fit K-Means for each value of k and record the inertia
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Plot the Elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='#2e86c1')
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()


# =============================================================================
# Clustering Functions
# =============================================================================


def run_kmeans(data, k):
    """
    Run K-Means clustering and return cluster labels.

    K-Means is a centroid-based algorithm that partitions data into k clusters
    by minimizing within-cluster variance.  It works well with spherical,
    similarly-sized clusters and is the primary baseline model for this project.

    Args:
        data (array-like): Preprocessed feature matrix (scaled RFM values).
        k (int): Number of clusters to create.

    Returns:
        np.ndarray: Array of cluster labels (0 to k-1) for each sample.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels


def run_dbscan(data, eps=0.5, min_samples=5):
    """
    Run DBSCAN clustering and return cluster labels.

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is
    a density-based algorithm that can discover clusters of arbitrary shape
    and automatically identifies noise/outlier points (labeled as -1).

    Unlike K-Means, DBSCAN does not require specifying the number of clusters
    in advance.  Instead, it uses two parameters:
      - eps: maximum distance between two points to be considered neighbors.
      - min_samples: minimum number of points to form a dense region (core point).

    Args:
        data (array-like): Preprocessed feature matrix (scaled RFM values).
        eps (float): Neighborhood radius. Defaults to 0.5.
        min_samples (int): Minimum points to form a cluster. Defaults to 5.

    Returns:
        np.ndarray: Array of cluster labels. Noise points are labeled -1.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels


# =============================================================================
# Evaluation Functions
# =============================================================================


def calculate_silhouette(data, labels):
    """
    Calculate the Silhouette Score for a set of cluster labels.

    The Silhouette Score measures how similar each point is to its own cluster
    (cohesion) compared to other clusters (separation).  Values range from
    -1 to +1:
      - +1: Points are well-matched to their cluster and poorly-matched to neighbors.
      -  0: Points are on the boundary between clusters.
      - -1: Points may be assigned to the wrong cluster.

    Edge cases handled:
      - If there are fewer than 2 unique labels, silhouette_score cannot be
        computed (it requires at least 2 clusters).  Returns -1 in this case.
      - If ALL labels are -1 (all noise, no clusters), returns -1.
      - If there is exactly 1 unique non-noise cluster, returns -1.

    Args:
        data (array-like): Preprocessed feature matrix used for clustering.
        labels (array-like): Cluster labels assigned to each sample.

    Returns:
        float: Silhouette Score in range [-1, 1], or -1 if the score
            cannot be computed (too few clusters).
    """
    # Count unique labels; silhouette_score requires at least 2 distinct clusters
    unique_labels = set(labels)

    # Edge case: all noise (-1 only) or only one cluster type present
    # sklearn's silhouette_score will raise an error if n_labels < 2
    # or if n_labels >= n_samples - 1, so we check first.
    if len(unique_labels) < 2 or (len(unique_labels) == 1 and -1 in unique_labels):
        return -1

    return silhouette_score(data, labels)
