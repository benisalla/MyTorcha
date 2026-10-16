"""Weight initialisers. They fill `tensor.data` in place and return the tensor.

Why it matters: with uniform(-1, 1) weights (MyTorcha v1) a 784-input neuron
outputs values of size ~sqrt(784/3) ≈ 16, which saturates tanh/sigmoid and
blows up softmax. These schemes scale by fan-in/fan-out to keep activations ~1.
"""

import math

import numpy as np

from .random import get_rng


def _fans(shape):
    if len(shape) < 2:
        return shape[0], shape[0]
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive  # (fan_in, fan_out)


def _fill(t, values):
    t.data[...] = values.astype(t.dtype)
    return t


def uniform_(t, a=-1.0, b=1.0):
    return _fill(t, get_rng().uniform(a, b, t.shape))


def normal_(t, mean=0.0, std=1.0):
    return _fill(t, get_rng().normal(mean, std, t.shape))


def zeros_(t):
    return _fill(t, np.zeros(t.shape))


def ones_(t):
    return _fill(t, np.ones(t.shape))


def xavier_uniform_(t, gain=1.0):
    """Glorot: Var(w) = 2 / (fan_in + fan_out). Good for tanh / sigmoid."""
    fan_in, fan_out = _fans(t.shape)
    bound = gain * math.sqrt(6 / (fan_in + fan_out))
    return uniform_(t, -bound, bound)


def xavier_normal_(t, gain=1.0):
    fan_in, fan_out = _fans(t.shape)
    return normal_(t, 0, gain * math.sqrt(2 / (fan_in + fan_out)))


def kaiming_uniform_(t, a=0.0):
    """He: Var(w) = 2 / fan_in, because ReLU zeroes half of its inputs."""
    fan_in, _ = _fans(t.shape)
    bound = math.sqrt(6 / ((1 + a * a) * fan_in))
    return uniform_(t, -bound, bound)


def kaiming_normal_(t, a=0.0):
    fan_in, _ = _fans(t.shape)
    return normal_(t, 0, math.sqrt(2 / ((1 + a * a) * fan_in)))


def orthogonal_(t, gain=1.0):
    rows, cols = t.shape[0], int(np.prod(t.shape[1:]))
    q, r = np.linalg.qr(get_rng().standard_normal((max(rows, cols), min(rows, cols))))
    q = q * np.sign(np.diag(r))
    q = q if rows >= cols else q.T
    return _fill(t, gain * q[:rows, :cols].reshape(t.shape))


def default_uniform_(t, fan_in):
    """PyTorch's default for Linear/Conv/RNN: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return uniform_(t, -bound, bound)
