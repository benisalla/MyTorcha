"""One global random generator for weight init and dropout, so runs are reproducible."""

import numpy as np

_rng = np.random.default_rng()


def manual_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng():
    return _rng
