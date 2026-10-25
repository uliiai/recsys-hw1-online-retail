"""Source download, auditable cleanup, global temporal protocol and FM metadata."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import zipfile

import numpy as np
import pandas as pd
import requests
from scipy import sparse

SOURCE = 'https://archive.ics.uci.edu/dataset/352/online+retail'
DOWNLOAD = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
ARCHIVE_SHA256 = 'f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b'
TRAIN_END = pd.Timestamp('2011-09-01')
VALIDATION_END = pd.Timestamp('2011-10-15')
SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data(directory='data'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'online_retail.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with requests.get(DOWNLOAD, timeout=(30, 180), stream=True) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        temporary.replace(archive)
    actual_hash = sha256(archive)
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError('Source archive differs from the reproducible reference SHA256')
    with zipfile.ZipFile(archive) as z:
        # Read only this member; never extract arbitrary archive paths.
        with z.open('Online Retail.xlsx') as stream:
            raw = pd.read_excel(stream, engine='openpyxl')
    return raw, {'source': SOURCE, 'download': DOWNLOAD, 'archive_sha256': sha256(archive)}


def clean_data(raw):
    frame = raw.copy()
    audit = [{'step': 'Исходные строки', 'rows': len(frame), 'removed': 0}]
    def keep(mask, label):
        nonlocal frame
        before = len(frame)
        frame = frame.loc[mask].copy()
        audit.append({'step': label, 'rows': len(frame), 'removed': before - len(frame)})
    keep(frame.CustomerID.notna() & frame.StockCode.notna() & frame.InvoiceDate.notna(), 'Есть user/item/time')
    keep(~frame.InvoiceNo.astype(str).str.upper().str.startswith('C'), 'Без отменённых счетов')
    keep((frame.Quantity > 0) & (frame.UnitPrice > 0), 'Положительные количество и цена')
    keep(frame.StockCode.astype(str).str.fullmatch(r'\d{5}[A-Za-z]{0,2}'), 'Товарные SKU (без доставки/комиссий)')
    keep(~frame.duplicated(), 'Без точных дубликатов строк')
    frame['CustomerID'] = frame.CustomerID.astype('int64').astype(str)
    frame['StockCode'] = frame.StockCode.astype(str)
    frame['InvoiceNo'] = frame.InvoiceNo.astype(str)
    frame['Description'] = frame.Description.fillna('').astype(str)
    frame['Country'] = frame.Country.fillna('UNKNOWN').astype(str)
    frame['InvoiceDate'] = pd.to_datetime(frame.InvoiceDate)
    frame = frame.sort_values(['InvoiceDate', 'InvoiceNo', 'CustomerID', 'StockCode'], kind='stable').reset_index(drop=True)
    assert 100_000 <= len(frame) <= 2_000_000
    assert min(frame.CustomerID.nunique(), frame.StockCode.nunique()) >= 1_000
    return frame, pd.DataFrame(audit)


@dataclass
class TrainingData:
    rows: pd.DataFrame
    user_ids: list
    item_ids: list
    history: sparse.csr_matrix
    last_purchase: sparse.csr_matrix
    cutoff: pd.Timestamp


def build_training(rows, cutoff):
    if not (rows.InvoiceDate < cutoff).all():
        raise ValueError('Training includes transactions at/after cutoff')
    user_ids, item_ids = sorted(rows.CustomerID.unique()), sorted(rows.StockCode.unique())
    user_map = {x: j for j, x in enumerate(user_ids)}
    item_map = {x: j for j, x in enumerate(item_ids)}
    pairs = rows.groupby(['CustomerID', 'StockCode'], sort=True).agg(
        frequency=('InvoiceNo', 'nunique'), last=('InvoiceDate', 'max')).reset_index()
    u, i = pairs.CustomerID.map(user_map).to_numpy(), pairs.StockCode.map(item_map).to_numpy()
    shape = len(user_ids), len(item_ids)
    history = sparse.csr_matrix((pairs.frequency.to_numpy(dtype=float), (u, i)), shape=shape)
    dates = pairs['last'].astype('int64').to_numpy() / 1e9
    last = sparse.csr_matrix((dates, (u, i)), shape=shape)
    history.sort_indices()
    last.sort_indices()
    return TrainingData(rows, user_ids, item_ids, history, last, pd.Timestamp(cutoff))


def make_evaluation(training, future_rows, minimum_history=5):
    """A common warm-start/new-item task. No target weighting or sampled catalog."""
    user_map = {x: j for j, x in enumerate(training.user_ids)}
    item_map = {x: j for j, x in enumerate(training.item_ids)}
    future = future_rows[['CustomerID', 'StockCode']].drop_duplicates().copy()
    audit = {'future_unique_pairs': len(future)}
    audit['cold_user_pairs'] = int((~future.CustomerID.isin(user_map)).sum())
    audit['cold_item_pairs'] = int((~future.StockCode.isin(item_map)).sum())
    # Cold-user/item counts can overlap; these are diagnostic, not additive drops.
    future = future[future.CustomerID.isin(user_map) & future.StockCode.isin(item_map)]
    u = future.CustomerID.map(user_map).to_numpy(dtype=int)
    i = future.StockCode.map(item_map).to_numpy(dtype=int)
    already_seen = np.asarray(training.history[u, i]).ravel() > 0
    audit['known_pairs_excluded'] = int(already_seen.sum())
    active = np.diff(training.history.indptr) >= minimum_history
    keep = ~already_seen & active[u]
    truth = {}
    for user, item in zip(u[keep], i[keep]):
        truth.setdefault(int(user), set()).add(int(item))
    users = np.array(sorted(truth), dtype=int)
    audit.update(eligible_users=len(users), relevant_pairs=int(keep.sum()),
                 catalog_items=len(item_map), minimum_history=minimum_history)
    if len(users) == 0:
        raise ValueError('No evaluable users')
    return users, truth, audit


class FeatureEncoder:
    """Country, last historical product description, historical median-price decile.

    Vocabulary, prices, countries and titles are all fit on training rows only.
    Feature 0 is zero padding; all real feature namespaces are disjoint.
    """
    def __init__(self, training, max_vocabulary=512, max_tokens=8):
        self.n_users, self.n_items = training.history.shape
        self.max_tokens = max_tokens
        rows = training.rows
        latest_users = rows.drop_duplicates('CustomerID', keep='last').set_index('CustomerID')
        latest_items = rows.drop_duplicates('StockCode', keep='last').set_index('StockCode')
        countries = sorted(latest_users.Country.unique())
        offset = 1 + self.n_users + self.n_items
        country_ids = {name: offset + j for j, name in enumerate(countries)}
        self.country = np.array([country_ids[latest_users.loc[u, 'Country']] for u in training.user_ids])
        offset += len(countries)
        median = rows.groupby('StockCode').UnitPrice.median().reindex(training.item_ids).to_numpy()
        self.price_edges = np.unique(np.quantile(median, np.linspace(0, 1, 11)[1:-1]))
        self.price = offset + np.searchsorted(self.price_edges, median, side='right')
        offset += len(self.price_edges) + 1
        def tokens(title):
            return sorted(set(re.findall(r'[a-z]{3,}', title.lower())))
        titles = [tokens(latest_items.loc[item, 'Description']) for item in training.item_ids]
        frequency = Counter(token for title in titles for token in title)
        top = sorted((t for t in frequency if frequency[t] >= 3), key=lambda t: (-frequency[t], t))[:max_vocabulary]
        self.vocabulary = {token: offset + j for j, token in enumerate(top)}
        self.n_features = offset + len(top)
        self.tokens = np.zeros((self.n_items, max_tokens), dtype=np.int64)
        self.token_values = np.zeros((self.n_items, max_tokens), dtype=np.float32)
        for item, title in enumerate(titles):
            encoded = [self.vocabulary[t] for t in title if t in self.vocabulary][:max_tokens]
            self.tokens[item, :len(encoded)] = encoded
            if encoded:
                self.token_values[item, :len(encoded)] = 1 / np.sqrt(len(encoded))

    def user_parts(self, users, metadata=True):
        ids = (1 + users)[:, None].astype(np.int64)
        if metadata:
            ids = np.column_stack([ids, self.country[users]])
        return ids, np.ones(ids.shape, dtype=np.float32)

    def item_parts(self, items, metadata=True):
        ids = (1 + self.n_users + items)[:, None].astype(np.int64)
        values = np.ones(ids.shape, dtype=np.float32)
        if metadata:
            ids = np.column_stack([ids, self.price[items], self.tokens[items]])
            values = np.column_stack([values, np.ones(len(items), dtype=np.float32), self.token_values[items]])
        return ids, values

    def encode(self, users, items, metadata=True):
        ui, uv = self.user_parts(users, metadata)
        ii, iv = self.item_parts(items, metadata)
        return np.column_stack([ui, ii]), np.column_stack([uv, iv])
