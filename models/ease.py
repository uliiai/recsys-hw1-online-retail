"""EASE closed form from a constrained ridge objective (no trained library)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender


class EASE(Recommender):
    def __init__(self, reg=200.0):
        self.reg = reg

    def fit(self, history, **kwargs):
        self.history = history.astype(np.float64).copy()
        self.history.data[:] = 1.0
        g = (self.history.T @ self.history).toarray()
        g.flat[::g.shape[0] + 1] += self.reg
        inverse = cho_solve(cho_factor(g, lower=True), np.eye(g.shape[0]))
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score(self, users):
        return np.asarray(self.history[users] @ self.weights)
