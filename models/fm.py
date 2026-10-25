"""Second-order FM, custom formula and negative sampling; PyTorch autograd."""
import numpy as np
import torch
from torch import nn
from .base import Recommender


class FMNetwork(nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(n_features, 1, padding_idx=0)
        self.factors = nn.Embedding(n_features, factors, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=.03)
        with torch.no_grad():
            self.factors.weight[0].zero_()

    def forward(self, ids, values):
        v = self.factors(ids) * values.unsqueeze(-1)
        interaction = .5 * ((v.sum(1)**2 - (v**2).sum(1)).sum(1))
        return self.bias + (self.linear(ids).squeeze(-1) * values).sum(1) + interaction


def sample_negatives(users, history, rng):
    """Uniform rejection sampling excludes *all* observed training positives."""
    items = rng.integers(history.shape[1], size=len(users))
    bad = np.asarray(history[users, items]).ravel() > 0
    while bad.any():
        items[bad] = rng.integers(history.shape[1], size=int(bad.sum()))
        bad = np.asarray(history[users, items]).ravel() > 0
    return items


class FactorizationMachine(Recommender):
    def __init__(self, factors=16, lr=.01, weight_decay=1e-5, epochs=6,
                 negatives=3, batch_size=4096, metadata=True, seed=42):
        self.factors, self.lr, self.weight_decay = factors, lr, weight_decay
        self.epochs, self.negatives, self.batch_size = epochs, negatives, batch_size
        self.metadata, self.seed = metadata, seed

    def fit(self, history, features=None, **kwargs):
        if features is None:
            raise ValueError('FM requires a training-only feature encoder')
        self.features = features
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = FMNetwork(features.n_features, self.factors)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_u, pos_i = history.nonzero()
        self.loss_history = []
        for _ in range(self.epochs):
            neg_u = np.repeat(pos_u, self.negatives)
            neg_i = sample_negatives(neg_u, history, rng)
            users, items = np.r_[pos_u, neg_u], np.r_[pos_i, neg_i]
            labels = np.r_[np.ones(len(pos_u)), np.zeros(len(neg_u))].astype(np.float32)
            order = rng.permutation(len(users))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                sel = order[start:start + self.batch_size]
                ids, vals = features.encode(users[sel], items[sel], metadata=self.metadata)
                logits = self.net(torch.from_numpy(ids), torch.from_numpy(vals))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels[sel]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(sel)
            self.loss_history.append(total / len(order))
        return self

    def score(self, users):
        # Exact FM decomposition: separate user/item terms, then one matrix product.
        # This equals forward() on the concatenated feature vector; verified in tests.
        with torch.no_grad():
            ui, uv = self.features.user_parts(np.asarray(users), self.metadata)
            ii, iv = self.features.item_parts(np.arange(self.features.n_items), self.metadata)
            def parts(ids, vals):
                ids, vals = torch.from_numpy(ids), torch.from_numpy(vals)
                v = self.net.factors(ids) * vals.unsqueeze(-1)
                summed = v.sum(1)
                own = (self.net.linear(ids).squeeze(-1) * vals).sum(1)
                own += .5 * ((summed**2 - (v**2).sum(1)).sum(1))
                return summed, own
            u, ub = parts(ui, uv)
            i, ib = parts(ii, iv)
            return (self.net.bias + ub[:, None] + ib[None, :] + u @ i.T).numpy()
