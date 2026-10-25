"""Weighted implicit matrix factorization; exact alternating normal equations."""
import numpy as np
from scipy import sparse
from .base import Recommender


def least_squares(confidence_extra, targets, fixed, reg):
    """Solve sum_i c_ui(p_ui - x_u@y_i)^2 + reg*||x_u||^2.

    Both sparse matrices have identical support. Unobserved c=1, p=0.
    Crucially the RHS uses c*p, not (c-1)*p or a binary replacement.
    """
    out = np.zeros((confidence_extra.shape[0], fixed.shape[1]), dtype=np.float64)
    gram = fixed.T @ fixed + reg * np.eye(fixed.shape[1])
    for row in range(confidence_extra.shape[0]):
        lo, hi = confidence_extra.indptr[row:row + 2]
        ids = confidence_extra.indices[lo:hi]
        if len(ids) == 0:
            continue
        y = fixed[ids]
        extra = confidence_extra.data[lo:hi]
        p = targets.data[lo:hi]
        a = gram + (y.T * extra) @ y
        b = y.T @ ((1.0 + extra) * p)
        out[row] = np.linalg.solve(a, b)
    return out


class ImplicitALS(Recommender):
    def __init__(self, factors=32, alpha=20.0, reg=1.0, iterations=12, seed=42):
        self.factors, self.alpha, self.reg = factors, alpha, reg
        self.iterations, self.seed = iterations, seed

    def fit(self, history, targets=None, **kwargs):
        counts = sparse.csr_matrix(history, dtype=np.float64)
        counts.sort_indices()
        extra = counts.copy()
        extra.data = self.alpha * np.log1p(extra.data)
        if targets is None:
            targets = counts.copy()
            targets.data[:] = 1.0
        targets = sparse.csr_matrix(targets, dtype=np.float64)
        targets.sort_indices()
        if not (np.array_equal(extra.indptr, targets.indptr) and
                np.array_equal(extra.indices, targets.indices)):
            raise ValueError('Confidence and target support must agree')
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, .05, (counts.shape[1], self.factors))
        extra_t, targets_t = extra.T.tocsr(), targets.T.tocsr()
        self.loss_history = []
        for _ in range(self.iterations):
            self.users = least_squares(extra, targets, self.items, self.reg)
            self.items = least_squares(extra_t, targets_t, self.users, self.reg)
            # Exact objective without materializing the full user-item matrix.
            u, i = counts.nonzero()
            pred = np.einsum('ij,ij->i', self.users[u], self.items[i])
            all_sq = np.sum((self.users.T @ self.users) * (self.items.T @ self.items))
            c, p = 1 + extra.data, targets.data
            loss = all_sq + np.sum(c * (p - pred)**2 - pred**2)
            loss += self.reg * (np.sum(self.users**2) + np.sum(self.items**2))
            self.loss_history.append(float(loss))
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
