"""Small mathematical checks, independent of the UCI download."""
import unittest
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from data_utils import build_training, make_evaluation, FeatureEncoder
from evaluation import metrics_at_k
from models import ImplicitALS, EASE, ItemKNN, RecencyFrequencyALS, FactorizationMachine
from models.ials import least_squares
from models.fm import FMNetwork, sample_negatives


def tiny_training():
    rows = pd.DataFrame({
        'CustomerID': ['a', 'a', 'b', 'b', 'c', 'c'],
        'StockCode': ['1', '2', '2', '3', '1', '3'],
        'InvoiceNo': ['1', '1', '2', '2', '3', '3'],
        'InvoiceDate': pd.to_datetime(['2011-01-01'] * 6),
        'Description': ['RED CUP', 'BLUE PLATE', 'BLUE PLATE', 'RED PLATE', 'RED CUP', 'RED PLATE'],
        'UnitPrice': [1., 2., 2., 3., 1., 3.],
        'Country': ['UK', 'UK', 'FR', 'FR', 'UK', 'UK']})
    return build_training(rows, pd.Timestamp('2011-02-01'))


class MathematicalTests(unittest.TestCase):
    def test_normal_equation_matches_dense_weighted_objective(self):
        y = np.array([[1., .2], [.3, 2.], [-.2, 1.]])
        extra = sparse.csr_matrix([[2., 0., 4.]])
        target = sparse.csr_matrix([[.25, 0., .8]])
        got = least_squares(extra, target, y, .7)[0]
        c, p = 1 + extra.toarray()[0], target.toarray()[0]
        expected = np.linalg.solve(y.T @ (c[:, None] * y) + .7 * np.eye(2), y.T @ (c * p))
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_als_objective_decreases(self):
        x = sparse.csr_matrix([[2., 0, 1], [0, 1, 3], [1, 2, 0]])
        model = ImplicitALS(factors=2, iterations=6).fit(x)
        self.assertTrue(np.all(np.diff(model.loss_history) <= 1e-8))

    def test_custom_changes_target_with_fixed_confidence(self):
        t = tiny_training()
        m = RecencyFrequencyALS(half_life=31, factors=2, iterations=2)
        p = m.make_targets(t.history, t.last_purchase, t.cutoff)
        np.testing.assert_allclose(p.data, (1 - np.exp(-.5)) * .5)
        self.assertTrue(np.array_equal(p.indices, t.history.indices))
        self.assertTrue(np.all((p.data > 0) & (p.data < 1)))

    def test_ease_constrained_stationarity(self):
        t = tiny_training()
        m = EASE(reg=2).fit(t.history)
        x = t.history.toarray()
        gradient = x.T @ (x @ m.weights - x) + 2 * m.weights
        np.fill_diagonal(gradient, 0)
        np.testing.assert_allclose(gradient, 0, atol=1e-10)
        np.testing.assert_allclose(np.diag(m.weights), 0)

    def test_neighbors_and_seen_mask(self):
        t = tiny_training()
        m = ItemKNN(neighbors=2, shrink=0).fit(t.history)
        self.assertAlmostEqual(m.weights[0, 1], .5)
        self.assertEqual(m.weights[0, 0], 0)
        self.assertEqual(m.recommend([0], t.history, k=1)[0, 0], 2)

    def test_metrics_have_known_values(self):
        recommendations = np.arange(10)[None, :]
        got = metrics_at_k(recommendations, [0], {0: {0, 10}}).iloc[0]
        self.assertAlmostEqual(got['Recall@10'], .5)
        self.assertAlmostEqual(got['Precision@10'], .1)
        self.assertAlmostEqual(got['NDCG@10'], 1 / (1 + 1 / np.log2(3)))

    def test_fm_formula_matches_explicit_pairs_and_fast_scoring(self):
        t = tiny_training()
        f = FeatureEncoder(t)
        m = FactorizationMachine(factors=3)
        m.features, m.net = f, FMNetwork(f.n_features, 3)
        users, items = np.array([0, 1]), np.array([2, 0])
        ids, vals = f.encode(users, items)
        direct = m.net(torch.from_numpy(ids), torch.from_numpy(vals)).detach().numpy()
        explicit = []
        for features, values in zip(ids, vals):
            v = m.net.factors.weight.detach().numpy()[features]
            w = m.net.linear.weight.detach().numpy()[features, 0]
            score = float(m.net.bias.detach()) + w @ values
            for a in range(len(features)):
                for b in range(a + 1, len(features)):
                    score += v[a] @ v[b] * values[a] * values[b]
            explicit.append(score)
        np.testing.assert_allclose(direct, explicit, atol=1e-7)
        np.testing.assert_allclose(m.score(users)[np.arange(2), items], direct, atol=1e-7)

    def test_negative_sampling_excludes_training_and_protocol_excludes_repeats(self):
        t = tiny_training()
        users = np.tile(np.arange(3), 10)
        items = sample_negatives(users, t.history, np.random.default_rng(42))
        self.assertTrue(np.all(np.asarray(t.history[users, items]).ravel() == 0))
        future = pd.DataFrame({'CustomerID': ['a', 'a', 'new'], 'StockCode': ['1', '3', '1']})
        u, truth, audit = make_evaluation(t, future, minimum_history=1)
        self.assertEqual(truth, {0: {2}})
        self.assertEqual(audit['cold_user_pairs'], 1)
        self.assertEqual(audit['known_pairs_excluded'], 1)


if __name__ == '__main__':
    unittest.main()
