"""Custom preferences change p, while keeping base iALS confidence identical."""
import numpy as np
import pandas as pd
from scipy import sparse
from .ials import ImplicitALS


class RecencyFrequencyALS(ImplicitALS):
    def __init__(self, half_life=90.0, frequency_scale=2.0, **kwargs):
        super().__init__(**kwargs)
        self.half_life, self.frequency_scale = half_life, frequency_scale

    def make_targets(self, history, last_purchase, cutoff):
        # frequency = number of separate invoices, never the wholesale quantity.
        last = sparse.csr_matrix(last_purchase)
        if not (np.array_equal(history.indptr, last.indptr) and
                np.array_equal(history.indices, last.indices)):
            raise ValueError('Last-purchase matrix must match history')
        age_days = (pd.Timestamp(cutoff).value / 1e9 - last.data) / 86400.0
        if np.any(age_days < 0):
            raise ValueError('Future transaction leaked into training')
        target = history.astype(np.float64).copy()
        target.data = (1 - np.exp(-history.data / self.frequency_scale)) * np.exp2(-age_days / self.half_life)
        return target

    def fit(self, history, last_purchase=None, cutoff=None, **kwargs):
        if last_purchase is None or cutoff is None:
            raise ValueError('Custom target needs historical dates and cutoff')
        target = self.make_targets(history, last_purchase, cutoff)
        self.target_summary = pd.Series(target.data).describe().to_dict()
        return super().fit(history, targets=target)
