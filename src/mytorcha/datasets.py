"""Small datasets for the tutorials. Nothing is downloaded."""

import numpy as np

from .random import get_rng


def make_spirals(n_per_class=100, n_classes=3, noise=0.2, rng=None):
    """The classic CS231n spiral: not linearly separable, so a linear model fails."""
    rng = rng or get_rng()
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.0, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.standard_normal(n_per_class) * noise
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


def make_moons(n=200, noise=0.1, rng=None):
    rng = rng or get_rng()
    t = rng.uniform(0, np.pi, n)
    half = rng.random(n) < 0.5
    X = np.where(half[:, None], np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
    X = X + rng.standard_normal((n, 2)) * noise
    return X.astype(np.float32), (~half).astype(np.int64)


def sine_sequences(n=256, T=30, rng=None):
    """Windows of noisy sine waves; target = the next value. For RNN/LSTM/GRU demos."""
    rng = rng or get_rng()
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    freq = rng.uniform(0.5, 1.5, (n, 1))
    t = np.arange(T + 1)[None, :] * 0.25
    wave = np.sin(freq * t + phase) + rng.standard_normal((n, T + 1)) * 0.02
    return wave[:, :T, None].astype(np.float32), wave[:, 1:, None].astype(np.float32)


def load_digits_csv(path="data/train.csv", limit=None):
    """Kaggle 'Digit Recognizer' CSV (label, pixel0..pixel783) -> X (N, 1, 28, 28) in [0, 1], y (N,)."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64, max_rows=limit)
    y = raw[:, 0]
    X = (raw[:, 1:] / 255.0).astype(np.float32).reshape(-1, 1, 28, 28)
    return X, y
