"""User-macro ranking metrics with deterministic full-catalog recommendations."""
import numpy as np
import pandas as pd


def metrics_at_k(recommendations, users, truth, k=10):
    discounts = 1 / np.log2(np.arange(2, k + 2))
    result = []
    for user, ranked in zip(users, recommendations):
        relevant = truth[int(user)]
        hits = np.array([item in relevant for item in ranked[:k]], dtype=float)
        ideal = discounts[:min(k, len(relevant))].sum()
        result.append({'user': int(user), 'NDCG@10': float(hits @ discounts / ideal),
                       'Recall@10': float(hits.sum() / len(relevant)),
                       'Precision@10': float(hits.sum() / k)})
    return pd.DataFrame(result)


def evaluate(model, training, users, truth, k=10, batch_size=128):
    recs = []
    for start in range(0, len(users), batch_size):
        recs.append(model.recommend(users[start:start + batch_size], training.history, k))
    recommendations = np.vstack(recs)
    per_user = metrics_at_k(recommendations, users, truth, k)
    metrics = per_user.drop(columns='user').mean().to_dict()
    metrics['Coverage@10'] = len(np.unique(recommendations)) / training.history.shape[1]
    return metrics, per_user, recommendations


def paired_bootstrap(first, second, repetitions=2000, seed=42):
    """CI for paired per-user NDCG difference, conditional on fitted models."""
    delta = np.asarray(first) - np.asarray(second)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(repetitions)])
    return {'difference': float(delta.mean()), 'ci_low': float(np.quantile(means, .025)),
            'ci_high': float(np.quantile(means, .975))}
