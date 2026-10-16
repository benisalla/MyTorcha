"""Optimizers. Each keeps its OWN state per parameter (a momentum buffer the
same shape as that weight). MyTorcha v1 shared one scalar across all weights,
so every weight's update was mixed with every other weight's gradient.

All formulas follow the PyTorch documentation, so results match torch.optim.
"""

import numpy as np


class Optimizer:
    def __init__(self, params, lr, **defaults):
        self.params = [p for p in params]
        assert self.params, "optimizer got an empty parameter list"
        self.lr = lr
        self.defaults = defaults
        self.state = [{} for _ in self.params]
        self.t = 0  # number of steps taken

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        for p, state in zip(self.params, self.state):
            if p.grad is not None:
                p.data -= self.update(p.data, p.grad, state).astype(p.dtype, copy=False)

    def update(self, w, g, state):
        """Return the amount to SUBTRACT from the weights w given gradient g."""
        raise NotImplementedError

    def _buffer(self, state, name, like):
        if name not in state:
            state[name] = np.zeros_like(like)
        return state[name]

    def state_dict(self):
        return {"t": self.t, "lr": self.lr, "state": self.state}

    def load_state_dict(self, d):
        self.t, self.lr, self.state = d["t"], d["lr"], d["state"]


class SGD(Optimizer):
    """w -= lr·g, optionally with momentum (a running sum of past gradients) and
    Nesterov look-ahead. weight_decay adds wd·w to the gradient (L2 penalty)."""

    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0, dampening=0.0):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.wd, self.dampening = momentum, nesterov, weight_decay, dampening

    def update(self, w, g, state):
        if self.wd:
            g = g + self.wd * w
        if self.momentum:
            if "buf" not in state:
                state["buf"] = g.copy()
            else:
                state["buf"] = self.momentum * state["buf"] + (1 - self.dampening) * g
            g = g + self.momentum * state["buf"] if self.nesterov else state["buf"]
        return self.lr * g


class Adagrad(Optimizer):
    """Divide by sqrt(sum of all past g²): rarely-updated weights get bigger steps."""

    def __init__(self, params, lr=0.01, eps=1e-10, weight_decay=0.0):
        super().__init__(params, lr)
        self.eps, self.wd = eps, weight_decay

    def update(self, w, g, state):
        if self.wd:
            g = g + self.wd * w
        s = self._buffer(state, "sum", g)
        s += g * g
        return self.lr * g / (np.sqrt(s) + self.eps)


class RMSprop(Optimizer):
    """Adagrad with a *moving average* of g², so the step size does not decay to 0."""

    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, momentum=0.0, centered=False, weight_decay=0.0):
        super().__init__(params, lr)
        self.alpha, self.eps, self.momentum, self.centered, self.wd = alpha, eps, momentum, centered, weight_decay

    def update(self, w, g, state):
        if self.wd:
            g = g + self.wd * w
        sq = self._buffer(state, "square_avg", g)
        sq[...] = self.alpha * sq + (1 - self.alpha) * g * g
        avg = sq
        if self.centered:
            ga = self._buffer(state, "grad_avg", g)
            ga[...] = self.alpha * ga + (1 - self.alpha) * g
            avg = sq - ga * ga
        step = g / (np.sqrt(avg) + self.eps)
        if self.momentum:
            buf = self._buffer(state, "buf", g)
            buf[...] = self.momentum * buf + step
            step = buf
        return self.lr * step


class Adadelta(Optimizer):
    """RMSprop whose step size is itself a running RMS of past updates (no lr tuning)."""

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        super().__init__(params, lr)
        self.rho, self.eps, self.wd = rho, eps, weight_decay

    def update(self, w, g, state):
        if self.wd:
            g = g + self.wd * w
        sq, acc = self._buffer(state, "square_avg", g), self._buffer(state, "acc_delta", g)
        sq[...] = self.rho * sq + (1 - self.rho) * g * g
        delta = np.sqrt(acc + self.eps) / np.sqrt(sq + self.eps) * g
        acc[...] = self.rho * acc + (1 - self.rho) * delta * delta
        return self.lr * delta


class Adam(Optimizer):
    """Momentum (m, 1st moment) + RMSprop (v, 2nd moment) + bias correction.

    m and v start at 0, so early on they are biased towards 0; dividing by
    (1 - β^t) fixes that. MyTorcha v1 had no bias correction and β=(0.99, 0.994).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.wd, self.amsgrad = eps, weight_decay, amsgrad

    def _moments(self, g, state):
        m, v = self._buffer(state, "m", g), self._buffer(state, "v", g)
        m[...] = self.b1 * m + (1 - self.b1) * g
        v[...] = self.b2 * v + (1 - self.b2) * g * g
        if self.amsgrad:
            vmax = self._buffer(state, "v_max", g)
            np.maximum(vmax, v, out=vmax)
            v = vmax
        m_hat = m / (1 - self.b1**self.t)
        v_hat = v / (1 - self.b2**self.t)
        return m_hat / (np.sqrt(v_hat) + self.eps)

    def update(self, w, g, state):
        if self.wd:
            g = g + self.wd * w  # L2 penalty inside the gradient (classic Adam)
        return self.lr * self._moments(g, state)


class AdamW(Adam):
    """Adam with *decoupled* weight decay: shrink weights directly instead of
    adding wd·w to the gradient (which Adam's normalisation would rescale)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        super().__init__(params, lr, betas, eps, 0.0, amsgrad)
        self.decoupled_wd = weight_decay

    def update(self, w, g, state):
        return self.lr * self.decoupled_wd * w + self.lr * self._moments(g, state)


class Adamax(Optimizer):
    """Adam with the infinity norm: v = max(β2·v, |g|)."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay

    def update(self, w, g, state):
        if self.wd:
            g = g + self.wd * w
        m, u = self._buffer(state, "m", g), self._buffer(state, "u", g)
        m[...] = self.b1 * m + (1 - self.b1) * g
        u[...] = np.maximum(self.b2 * u, np.abs(g) + self.eps)
        return self.lr / (1 - self.b1**self.t) * m / u


class Lion(Optimizer):
    """Lion (Chen et al., 2023): step by the SIGN of an interpolated momentum.
    Every weight moves by exactly ±lr, so use a ~10x smaller lr than Adam."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.wd = weight_decay

    def update(self, w, g, state):
        m = self._buffer(state, "m", g)
        direction = np.sign(self.b1 * m + (1 - self.b1) * g)
        m[...] = self.b2 * m + (1 - self.b2) * g
        return self.lr * (direction + self.wd * w)


def clip_grad_norm_(params, max_norm):
    """Rescale all gradients together so their global L2 norm is <= max_norm.
    Essential for RNNs, whose gradients can explode. Returns the norm before clipping."""
    grads = [p.grad for p in params if p.grad is not None]
    total = float(np.sqrt(sum((g.astype(np.float64) ** 2).sum() for g in grads)))
    if total > max_norm:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * (max_norm / (total + 1e-6))
    return total
