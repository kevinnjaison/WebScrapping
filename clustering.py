# clustering.py
from sklearn.cluster import KMeans
import joblib

def cluster_jobs(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model

def save_model(model, vectorizer, path="models/"):
    joblib.dump(model, path + "kmeans_model.pkl")
    joblib.dump(vectorizer, path + "vectorizer.pkl")
    
# clustering.py
def load_model_and_vectorizer(path="models/"):
    model = joblib.load(path + "kmeans_model.pkl")
    vectorizer = joblib.load(path + "vectorizer.pkl")
    return model, vectorizer

