"""Tensor: an n-dimensional array that remembers how it was computed.

Same idea as `mytorcha.scalar.Node`, but every node holds a whole NumPy array,
so one matrix multiply replaces thousands of scalar nodes.

Every operation does two things:
  1. compute the output array (forward), and
  2. store `_grad_fn(g) -> one gradient per input`: given dL/d(output) = g,
     return dL/d(input) for each input (the local chain-rule step).

`backward()` walks the graph in reverse topological order and adds those
gradients into each input's `.grad`.
"""

from contextlib import contextmanager

import numpy as np

DEFAULT_DTYPE = np.float32
_grad_enabled = True


@contextmanager
def no_grad():
    """Run code without recording a graph (evaluation, optimizer updates)."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


def is_grad_enabled():
    return _grad_enabled


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, undoing NumPy broadcasting.

    If b with shape (3,) was added to x with shape (N, 3), b was used N times,
    so its gradient is the sum of the N row gradients.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _to_array(data, dtype=None):
    if isinstance(data, Tensor):
        data = data.data
    if dtype is not None:
        return np.asarray(data, dtype=dtype)
    if isinstance(data, (np.ndarray, np.generic)):  # np.generic: e.g. arr.sum() -> np.float64
        return np.asarray(data)  # keep float64 / int arrays as they are (gradcheck, labels)
    arr = np.asarray(data)
    if arr.dtype.kind == "f":
        arr = arr.astype(DEFAULT_DTYPE)
    return arr


class Tensor:
    __array_priority__ = 1000  # make `ndarray + Tensor` call Tensor.__radd__

    def __init__(self, data, requires_grad=False, dtype=None, _prev=(), _op=""):
        self.data = _to_array(data, dtype)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._prev = _prev
        self._op = _op
        self._grad_fn = None

    # ---- graph construction --------------------------------------------------
    @staticmethod
    def _make(data, parents, grad_fn, op):
        """Wrap an op's output; record the graph only if a gradient is needed."""
        track = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=track, _prev=parents if track else (), _op=op)
        if track:
            out._grad_fn = grad_fn
        return out

    def _wrap(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype if self.is_float else None))

    # ---- backpropagation -----------------------------------------------------
    def backward(self, grad=None):
        if grad is None:
            assert self.data.size == 1, "backward() without a gradient needs a scalar output"
            grad = np.ones_like(self.data)
        self.grad = _to_array(grad, self.data.dtype) + (0 if self.grad is None else self.grad)

        order, seen, stack = [], set(), [(self, False)]
        while stack:  # iterative topological sort (no recursion limit)
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._prev if id(p) not in seen)

        for node in reversed(order):
            if node._grad_fn is None or node.grad is None:
                continue
            for parent, g in zip(node._prev, node._grad_fn(node.grad)):
                if g is None or not parent.requires_grad:
                    continue
                g = _unbroadcast(np.asarray(g), parent.shape).astype(parent.data.dtype, copy=False)
                parent.grad = g if parent.grad is None else parent.grad + g

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data)

    # ---- array-like helpers --------------------------------------------------
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    size = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)
    is_float = property(lambda self: self.data.dtype.kind == "f")
    T = property(lambda self: self.transpose())

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        g = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4, threshold=20)}{g})"

    def astype(self, dtype):
        return Tensor._make(self.data.astype(dtype), (self,), lambda g: (g,), "astype")

    def argmax(self, axis=None):
        return Tensor(self.data.argmax(axis=axis))

    # comparisons give plain (non-differentiable) boolean tensors
    def __eq__(self, o):
        return Tensor(self.data == _to_array(o))

    def __ne__(self, o):
        return Tensor(self.data != _to_array(o))

    def __lt__(self, o):
        return Tensor(self.data < _to_array(o))

    def __le__(self, o):
        return Tensor(self.data <= _to_array(o))

    def __gt__(self, o):
        return Tensor(self.data > _to_array(o))

    def __ge__(self, o):
        return Tensor(self.data >= _to_array(o))

    __hash__ = object.__hash__  # identity, like scalar.Node

    # ---- elementwise arithmetic ----------------------------------------------
    def __add__(self, other):
        other = self._wrap(other)
        return Tensor._make(self.data + other.data, (self, other), lambda g: (g, g), "+")

    def __mul__(self, other):
        other = self._wrap(other)
        a, b = self.data, other.data
        return Tensor._make(a * b, (self, other), lambda g: (g * b, g * a), "*")

    def __neg__(self):
        return Tensor._make(-self.data, (self,), lambda g: (-g,), "neg")

    def __sub__(self, other):
        other = self._wrap(other)
        return Tensor._make(self.data - other.data, (self, other), lambda g: (g, -g), "-")

    def __truediv__(self, other):
        other = self._wrap(other)
        a, b = self.data, other.data
        return Tensor._make(a / b, (self, other), lambda g: (g / b, -g * a / (b * b)), "/")

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        a = self.data
        return Tensor._make(a**k, (self,), lambda g: (g * k * a ** (k - 1),), f"**{k}")

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    def __matmul__(self, other):
        other = self._wrap(other)
        a, b = self.data, other.data
        a2 = a[None, :] if a.ndim == 1 else a  # treat vectors as 1-row / 1-col matrices
        b2 = b[:, None] if b.ndim == 1 else b

        def grad_fn(g):
            g2 = g
            if a.ndim == 1:
                g2 = np.expand_dims(g2, -2)
            if b.ndim == 1:
                g2 = np.expand_dims(g2, -1)
            ga = g2 @ np.swapaxes(b2, -1, -2)  # dL/dA = dL/dC · Bᵀ
            gb = np.swapaxes(a2, -1, -2) @ g2  # dL/dB = Aᵀ · dL/dC
            if a.ndim == 1:
                ga = ga.squeeze(-2)
            if b.ndim == 1:
                gb = gb.squeeze(-1)
            return ga, gb

        return Tensor._make(a @ b, (self, other), grad_fn, "@")

    def __rmatmul__(self, other):
        return self._wrap(other) @ self

    # ---- elementwise functions -----------------------------------------------
    def exp(self):
        e = np.exp(self.data)
        return Tensor._make(e, (self,), lambda g: (g * e,), "exp")

    def log(self):
        a = self.data
        return Tensor._make(np.log(a), (self,), lambda g: (g / a,), "log")

    def sqrt(self):
        s = np.sqrt(self.data)
        return Tensor._make(s, (self,), lambda g: (g * 0.5 / s,), "sqrt")

    def abs(self):
        a = self.data
        return Tensor._make(np.abs(a), (self,), lambda g: (g * np.sign(a),), "abs")

    __abs__ = abs

    def clip(self, lo=None, hi=None):
        a = self.data
        inside = np.ones_like(a, dtype=bool)
        if lo is not None:
            inside &= a >= lo
        if hi is not None:
            inside &= a <= hi
        return Tensor._make(np.clip(a, lo, hi), (self,), lambda g: (g * inside,), "clip")

    def masked_fill(self, mask, value):
        """Replace entries where `mask` is True by `value` (e.g. -inf in attention)."""
        mask = _to_array(mask).astype(bool)
        out = np.where(mask, np.asarray(value, dtype=self.dtype), self.data)
        return Tensor._make(out, (self,), lambda g: (np.where(mask, 0, g),), "masked_fill")

    # ---- reductions ------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        shape = self.shape

        def grad_fn(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, shape),)

        return Tensor._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), grad_fn, "sum")

    def mean(self, axis=None, keepdims=False):
        n = self.data.size / np.asarray(self.data.sum(axis=axis, keepdims=keepdims)).size
        return self.sum(axis=axis, keepdims=keepdims) / n

    def var(self, axis=None, keepdims=False):
        """Biased variance (divides by N), as used by BatchNorm / LayerNorm."""
        mu = self.mean(axis=axis, keepdims=True)
        return ((self - mu) ** 2).mean(axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        out = self.data.max(axis=axis, keepdims=True)
        mask = self.data == out
        mask = mask / mask.sum(axis=axis, keepdims=True)  # split ties evenly

        def grad_fn(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (mask * g,)

        res = out if keepdims else (out.squeeze(axis) if axis is not None else out.reshape(()))
        return Tensor._make(res, (self,), grad_fn, "max")

    def min(self, axis=None, keepdims=False):
        return -((-self).max(axis=axis, keepdims=keepdims))

    # ---- shape ops -------------------------------------------------------------
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        old = self.shape
        return Tensor._make(self.data.reshape(shape), (self,), lambda g: (g.reshape(old),), "reshape")

    view = reshape

    def flatten(self, start=1):
        return self.reshape(*self.shape[:start], -1)

    def transpose(self, *axes):
        axes = axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        axes = axes or tuple(reversed(range(self.ndim)))
        inv = np.argsort(axes)
        return Tensor._make(self.data.transpose(axes), (self,), lambda g: (g.transpose(inv),), "T")

    permute = transpose

    def swapaxes(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(axes)

    def unsqueeze(self, axis):
        return self.reshape(np.expand_dims(self.data, axis).shape)

    def squeeze(self, axis=None):
        return self.reshape(np.squeeze(self.data, axis).shape)

    def __getitem__(self, idx):
        idx = idx.data if isinstance(idx, Tensor) else idx
        shape, dtype = self.shape, self.dtype

        def grad_fn(g):
            full = np.zeros(shape, dtype=dtype)
            np.add.at(full, idx, g)  # add.at: repeated indices (embeddings) accumulate
            return (full,)

        return Tensor._make(self.data[idx], (self,), grad_fn, "getitem")

    def pad(self, pad_width, value=0.0):
        """np.pad with a constant; pad_width like ((0,0),(0,0),(1,1),(1,1))."""
        slices = tuple(slice(lo, lo + n) for (lo, _), n in zip(pad_width, self.shape))
        out = np.pad(self.data, pad_width, constant_values=value)
        return Tensor._make(out, (self,), lambda g: (g[slices],), "pad")


def tensor(data, requires_grad=False, dtype=None):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad=False):
    return Tensor(np.zeros(shape, DEFAULT_DTYPE), requires_grad)


def ones(*shape, requires_grad=False):
    return Tensor(np.ones(shape, DEFAULT_DTYPE), requires_grad)


def randn(*shape, requires_grad=False, rng=None):
    rng = rng or np.random
    return Tensor(rng.standard_normal(shape).astype(DEFAULT_DTYPE), requires_grad)


def arange(*args):
    return Tensor(np.arange(*args))


def concat(tensors, axis=0):
    sizes = [t.shape[axis] for t in tensors]
    cuts = np.cumsum(sizes)[:-1]
    out = np.concatenate([t.data for t in tensors], axis=axis)
    return Tensor._make(out, tuple(tensors), lambda g: tuple(np.split(g, cuts, axis=axis)), "concat")


cat = concat


def stack(tensors, axis=0):
    out = np.stack([t.data for t in tensors], axis=axis)
    n = len(tensors)
    grad_fn = lambda g: tuple(np.squeeze(s, axis) for s in np.split(g, n, axis=axis))
    return Tensor._make(out, tuple(tensors), grad_fn, "stack")


def where(cond, a, b):
    cond = _to_array(cond).astype(bool)
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    out = np.where(cond, a.data, b.data)
    return Tensor._make(out, (a, b), lambda g: (np.where(cond, g, 0), np.where(cond, 0, g)), "where")
