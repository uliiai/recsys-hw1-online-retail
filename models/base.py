"""Shared deterministic top-K; training and scoring live in individual models."""
import numpy as np


class Recommender:
    def recommend(self, users, seen, k=10):
        scores = np.asarray(self.score(users), dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite model scores')
        for row, user in enumerate(users):
            scores[row, seen.indices[seen.indptr[user]:seen.indptr[user + 1]]] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError('Fewer than K unseen candidates')
        # Stable sorting breaks ties by increasing catalog ID for every model.
        return np.argsort(-scores, axis=1, kind='stable')[:, :k]
