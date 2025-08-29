import numpy as np
from sklearn.cluster import DBSCAN

class Clusterer:
    def __init__(self, eps=0.45, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, embeddings):
        if len(embeddings) == 0:
            return []
        X = np.vstack(embeddings).astype(np.float32)
        # Cosine metric on L2-normalized embeddings
        # sklearn DBSCAN does not support cosine directly prior to 1.3 for precomputed,
        # so we compute distances via (1 - cosine_similarity).
        # But since embeddings are already L2-normalized, cosine distance ~ 1 - dot(a,b).
        S = X @ X.T  # cosine similarity
        D = 1.0 - S  # cosine distance
        # Set diagonal to 0 just to be safe
        np.fill_diagonal(D, 0.0)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        labels = db.fit_predict(D)
        return labels