"""Item cosine neighbors with co-occurrence shrinkage, from scratch."""
import numpy as np
from scipy import sparse
from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=80, shrink=10.0):
        self.neighbors, self.shrink = neighbors, shrink

    def fit(self, history, **kwargs):
        self.history = history.astype(np.float64).copy()
        self.history.data[:] = 1.0
        co = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / np.maximum(np.outer(norms, norms), 1e-12)
        sim *= co / (co + self.shrink)
        np.fill_diagonal(sim, 0.0)
        rows, cols, vals = [], [], []
        # Keep K source neighbors per target column.
        for target in range(sim.shape[1]):
            idx = np.argsort(-sim[:, target], kind='stable')[:self.neighbors]
            idx = idx[sim[idx, target] > 0]
            rows.extend(idx)
            cols.extend([target] * len(idx))
            vals.extend(sim[idx, target])
        self.weights = sparse.csc_matrix((vals, (rows, cols)), shape=sim.shape)
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
