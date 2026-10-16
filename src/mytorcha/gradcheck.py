"""Check analytic gradients against finite differences.

If backward() is right, nudging an input by ±h must change the loss by about
grad · 2h. This one function is what makes every op in MyTorcha trustworthy.
"""

import numpy as np

from .tensor import Tensor


def numerical_grad(f, x, h=1e-6):
    """d f() / d x.data by central differences (x is changed in place, then restored)."""
    grad = np.zeros_like(x.data, dtype=np.float64)
    it = np.nditer(x.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x.data[i]
        x.data[i] = old + h
        plus = float(f().data.sum())
        x.data[i] = old - h
        minus = float(f().data.sum())
        x.data[i] = old
        grad[i] = (plus - minus) / (2 * h)
    return grad


def gradcheck(f, inputs, atol=1e-5, rtol=1e-4):
    """Compare backward() of `f().sum()` with numerical gradients for each input.

    Use float64 inputs: float32 is too coarse for finite differences.
    Returns True or raises AssertionError with the worst mismatch.
    """
    for x in inputs:
        x.grad = None
    f().sum().backward()
    for k, x in enumerate(inputs):
        analytic = np.zeros_like(x.data) if x.grad is None else x.grad
        numeric = numerical_grad(f, x)
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            err = np.abs(analytic - numeric).max()
            raise AssertionError(f"input {k}: max |analytic - numeric| = {err:.3g}")
    return True


def rand(*shape, seed=0, positive=False, requires_grad=True):
    """float64 test tensor."""
    a = np.random.default_rng(seed).standard_normal(shape)
    if positive:
        a = np.abs(a) + 0.5
    return Tensor(a, requires_grad=requires_grad)
