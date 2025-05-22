from sklearn.cluster import KMeans

def cluster_jobs(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model
