"""Datasets and a minimal DataLoader."""

import numpy as np

from .random import get_rng
from .tensor import Tensor


class Dataset:
    """Subclass and implement __len__ and __getitem__(i) -> tuple of arrays."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Wrap arrays that share their first dimension: TensorDataset(X, y)."""

    def __init__(self, *arrays):
        self.arrays = [a.data if isinstance(a, Tensor) else np.asarray(a) for a in arrays]
        assert all(len(a) == len(self.arrays[0]) for a in self.arrays), "arrays differ in length"

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, i):
        return tuple(a[i] for a in self.arrays)


def _to_tensor(batch):
    batch = np.asarray(batch)
    if batch.dtype.kind == "f":
        batch = batch.astype(np.float32)
    return Tensor(batch)


class DataLoader:
    """Yield mini-batches of Tensors: for xb, yb in DataLoader(ds, batch_size=64, shuffle=True)."""

    def __init__(self, dataset, batch_size=32, shuffle=False, drop_last=False):
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        order = get_rng().permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                return
            if isinstance(self.dataset, TensorDataset):  # fast path: fancy-index whole arrays
                items = [a[idx] for a in self.dataset.arrays]
            else:
                samples = [self.dataset[int(i)] for i in idx]
                items = [np.stack(col) for col in zip(*samples)]
            yield tuple(_to_tensor(x) for x in items)
