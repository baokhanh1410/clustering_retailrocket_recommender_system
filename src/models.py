import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def plot_elbow(data, max_k=10):
    """
    Plots the Elbow curve to find optimal k for KMeans.
    """
    inertia = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='#2e86c1')
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

def run_kmeans(data, k):
    """
    Runs KMeans clustering.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels

def run_dbscan(data, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels

def calculate_silhouette(data, labels):
    """
    Calculates the Silhouette Score.
    """
    # Silhouette score requires at least 2 clusters and not more than n_samples - 1
    unique_labels = set(labels)
    if len(unique_labels) < 2 or (len(unique_labels) == 1 and -1 in unique_labels):
        return -1
    
    return silhouette_score(data, labels)
