"""Stateless functions: activations, softmax, losses, convolution, pooling, dropout.

Each activation is one *fused* op: we write f(x) and f'(x) together instead of
composing primitives, which is faster and keeps every formula on one line.
"""

import math

import numpy as np

from .random import get_rng
from .tensor import Tensor, _to_array

# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------


def _act(x, out, dfdx, op):
    """Elementwise op with output `out` and local derivative `dfdx()` (lazy)."""
    return Tensor._make(out, (x,), lambda g: (g * dfdx(),), op)


def _sigmoid(a):
    return np.exp(-np.logaddexp(0, -a))  # 1 / (1 + e^-a) without overflow


def identity(x):
    return x


linear_act = identity


def relu(x):
    a = x.data
    return _act(x, np.maximum(a, 0), lambda: (a > 0).astype(a.dtype), "relu")


def relu6(x):
    a = x.data
    return _act(x, np.clip(a, 0, 6), lambda: ((a > 0) & (a < 6)).astype(a.dtype), "relu6")


def leaky_relu(x, negative_slope=0.01):
    a = x.data
    return _act(x, np.where(a > 0, a, negative_slope * a), lambda: np.where(a > 0, 1.0, negative_slope), "leaky_relu")


def elu(x, alpha=1.0):
    a = x.data
    e = np.exp(np.minimum(a, 0))
    return _act(x, np.where(a > 0, a, alpha * (e - 1)), lambda: np.where(a > 0, 1.0, alpha * e), "elu")


def celu(x, alpha=1.0):
    a = x.data
    e = np.exp(np.minimum(a, 0) / alpha)
    return _act(x, np.where(a > 0, a, alpha * (e - 1)), lambda: np.where(a > 0, 1.0, e), "celu")


SELU_ALPHA, SELU_SCALE = 1.6732632423543772, 1.0507009873554805


def selu(x):
    a = x.data
    e = np.exp(np.minimum(a, 0))
    out = SELU_SCALE * np.where(a > 0, a, SELU_ALPHA * (e - 1))
    return _act(x, out, lambda: SELU_SCALE * np.where(a > 0, 1.0, SELU_ALPHA * e), "selu")


def sigmoid(x):
    s = _sigmoid(x.data)
    return _act(x, s, lambda: s * (1 - s), "sigmoid")


def logsigmoid(x):
    a = x.data
    return _act(x, -np.logaddexp(0, -a), lambda: _sigmoid(-a), "logsigmoid")


def hard_sigmoid(x):
    a = x.data
    return _act(x, np.clip(a / 6 + 0.5, 0, 1), lambda: ((a > -3) & (a < 3)) / 6.0, "hard_sigmoid")


def tanh(x):
    t = np.tanh(x.data)
    return _act(x, t, lambda: 1 - t * t, "tanh")


def hardtanh(x, lo=-1.0, hi=1.0):
    a = x.data
    return _act(x, np.clip(a, lo, hi), lambda: ((a > lo) & (a < hi)).astype(a.dtype), "hardtanh")


def softplus(x, beta=1.0):
    a = x.data
    return _act(x, np.logaddexp(0, beta * a) / beta, lambda: _sigmoid(beta * a), "softplus")


def softsign(x):
    a = x.data
    return _act(x, a / (1 + np.abs(a)), lambda: 1 / (1 + np.abs(a)) ** 2, "softsign")


def silu(x):
    """SiLU / swish: x · sigmoid(x)."""
    a = x.data
    s = _sigmoid(a)
    return _act(x, a * s, lambda: s * (1 + a * (1 - s)), "silu")


swish = silu


def hardswish(x):
    a = x.data
    out = a * np.clip(a + 3, 0, 6) / 6
    return _act(x, out, lambda: np.where(a < -3, 0.0, np.where(a > 3, 1.0, (2 * a + 3) / 6)), "hardswish")


def mish(x):
    """x · tanh(softplus(x))."""
    a = x.data
    t = np.tanh(np.logaddexp(0, a))
    return _act(x, a * t, lambda: t + a * (1 - t * t) * _sigmoid(a), "mish")


_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu(x, approximate="none"):
    """GELU: x · Φ(x). approximate="tanh" uses the GPT-2 formula."""
    a = x.data
    if approximate == "tanh":
        c = math.sqrt(2 / math.pi)
        t = np.tanh(c * (a + 0.044715 * a**3))
        dfdx = lambda: 0.5 * (1 + t) + 0.5 * a * (1 - t * t) * c * (1 + 3 * 0.044715 * a * a)
        return _act(x, 0.5 * a * (1 + t), dfdx, "gelu")
    cdf = (0.5 * (1 + _erf(a / math.sqrt(2)))).astype(a.dtype)
    pdf = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    return _act(x, a * cdf, lambda: cdf + a * pdf, "gelu")


def arctan(x):
    a = x.data
    return _act(x, np.arctan(a), lambda: 1 / (1 + a * a), "arctan")


def prelu(x, weight):
    """Leaky ReLU whose negative slope `weight` is learned (composed from primitives)."""
    pos = (x.data > 0).astype(x.dtype)
    return x * pos + x * (1 - pos) * weight


def softmax(x, axis=-1):
    a = x.data
    e = np.exp(a - a.max(axis=axis, keepdims=True))  # subtracting the max avoids overflow
    s = e / e.sum(axis=axis, keepdims=True)
    # Jacobian-vector product of softmax: s * (g - <g, s>)
    return Tensor._make(s, (x,), lambda g: (s * (g - (g * s).sum(axis=axis, keepdims=True)),), "softmax")


def log_softmax(x, axis=-1):
    a = x.data
    shifted = a - a.max(axis=axis, keepdims=True)
    out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    s = np.exp(out)
    return Tensor._make(out, (x,), lambda g: (g - s * g.sum(axis=axis, keepdims=True),), "log_softmax")


def softmin(x, axis=-1):
    return softmax(-x, axis)


ACTIVATIONS = {
    f.__name__: f
    for f in [identity, relu, relu6, leaky_relu, elu, celu, selu, sigmoid, logsigmoid, hard_sigmoid,
              tanh, hardtanh, softplus, softsign, silu, hardswish, mish, gelu, arctan]
}
ACTIVATIONS.update(linear=identity, swish=silu)


def get_activation(name):
    try:
        return ACTIVATIONS[name]
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}") from None


# ---------------------------------------------------------------------------
# losses   (reduction: "mean" | "sum" | "none")
# ---------------------------------------------------------------------------


def _reduce(loss, reduction):
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction must be mean, sum or none, got {reduction!r}")


def _as_tensor(t, like):
    return t if isinstance(t, Tensor) else Tensor(np.asarray(t, dtype=like.dtype))


def mse_loss(pred, target, reduction="mean"):
    return _reduce((pred - _as_tensor(target, pred)) ** 2, reduction)


def l1_loss(pred, target, reduction="mean"):
    return _reduce((pred - _as_tensor(target, pred)).abs(), reduction)


def huber_loss(pred, target, delta=1.0, reduction="mean"):
    """Quadratic near 0, linear far away: robust to outliers."""
    d = pred - _as_tensor(target, pred)
    small = np.abs(d.data) <= delta
    quad, lin = 0.5 * d * d, delta * (d.abs() - 0.5 * delta)
    return _reduce(quad * small + lin * (~small), reduction)


def smooth_l1_loss(pred, target, beta=1.0, reduction="mean"):
    return _reduce(huber_loss(pred, target, delta=beta, reduction="none") / beta, reduction)


def binary_cross_entropy(prob, target, reduction="mean", eps=1e-7):
    """BCE on probabilities in (0, 1). Prefer the *_with_logits version."""
    y = _as_tensor(target, prob)
    p = prob.clip(eps, 1 - eps)
    return _reduce(-(y * p.log() + (1 - y) * (1 - p).log()), reduction)


def binary_cross_entropy_with_logits(logits, target, reduction="mean"):
    """Stable BCE: max(x,0) - x·y + log(1 + e^-|x|); gradient is sigmoid(x) - y."""
    a, y = logits.data, _to_array(target).astype(logits.dtype)
    loss = np.maximum(a, 0) - a * y + np.logaddexp(0, -np.abs(a))
    out = Tensor._make(loss, (logits,), lambda g: (g * (_sigmoid(a) - y),), "bce_logits")
    return _reduce(out, reduction)


def nll_loss(log_probs, target, reduction="mean", ignore_index=None):
    """Negative log-likelihood. Classes on the LAST axis: (..., C) and target (...)."""
    C = log_probs.shape[-1]
    lp = log_probs.reshape(-1, C)
    t = _to_array(target).reshape(-1).astype(np.int64)
    keep = np.ones_like(t, dtype=bool) if ignore_index is None else t != ignore_index
    picked = -lp[np.arange(len(t))[keep], t[keep]]
    return _reduce(picked, reduction)


def cross_entropy(logits, target, reduction="mean", label_smoothing=0.0, ignore_index=None):
    """Softmax + NLL in one stable step. logits (..., C); target: class indices (...).

    Note: MyTorcha puts classes on the last axis (PyTorch uses axis 1), so a
    sequence model's (N, T, C) logits work without transposing.
    """
    lp = log_softmax(logits, axis=-1)
    nll = nll_loss(lp, target, reduction=reduction, ignore_index=ignore_index)
    if label_smoothing == 0:
        return nll
    smooth = _reduce(-lp.mean(axis=-1).reshape(-1), reduction)
    return (1 - label_smoothing) * nll + label_smoothing * smooth


def kl_div(log_q, p, reduction="batchmean"):
    """KL(p || q) given log q (e.g. log_softmax output) and target probabilities p."""
    p = _as_tensor(p, log_q)
    pt = p.data
    logp = Tensor(np.log(np.where(pt > 0, pt, 1.0)))  # 0 · log 0 = 0
    loss = p * (logp - log_q)
    if reduction == "batchmean":
        return loss.sum() / log_q.shape[0]
    return _reduce(loss, reduction)


def hinge_loss(scores, target, margin=1.0, reduction="mean"):
    """Multi-class SVM loss: sum over wrong classes of max(0, s_j - s_y + margin)."""
    t = _to_array(target).astype(np.int64)
    n = np.arange(len(t))
    correct = scores[n, t].reshape(-1, 1)
    margins = relu(scores - correct + margin)
    mask = np.ones(scores.shape, dtype=scores.dtype)
    mask[n, t] = 0
    return _reduce((margins * mask).sum(axis=1), reduction)


def cosine_similarity(a, b, axis=-1, eps=1e-8):
    num = (a * b).sum(axis=axis)
    return num / ((a * a).sum(axis=axis).sqrt() * (b * b).sum(axis=axis).sqrt() + eps)


# ---------------------------------------------------------------------------
# linear, dropout
# ---------------------------------------------------------------------------


def linear(x, weight, bias=None):
    """x @ Wᵀ + b, with W of shape (out_features, in_features) like PyTorch."""
    out = x @ weight.T
    return out if bias is None else out + bias


def dropout(x, p=0.5, training=True, rng=None):
    """Inverted dropout: zero with prob p and scale survivors by 1/(1-p), so the
    expected activation is unchanged and evaluation needs no rescaling."""
    if not training or p == 0:
        return x
    rng = rng or get_rng()
    mask = (rng.random(x.shape) >= p).astype(x.dtype) / (1 - p)
    return x * mask


# ---------------------------------------------------------------------------
# convolution and pooling (NCHW layout, like PyTorch)
# ---------------------------------------------------------------------------


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _windows(xp, kh, kw, s, d, Ho, Wo):
    """Zero-copy view (N, C, Ho, Wo, kh, kw) of every receptive field (im2col)."""
    N, C = xp.shape[:2]
    sN, sC, sH, sW = xp.strides
    return np.lib.stride_tricks.as_strided(
        xp, (N, C, Ho, Wo, kh, kw), (sN, sC, sH * s[0], sW * s[1], sH * d[0], sW * d[1]), writeable=False
    )


def _scatter(cols, shape, s, d):
    """Inverse of _windows: add every window back to its place (col2im).
    Overlapping windows add up, which is exactly the chain rule for reused pixels."""
    out = np.zeros(shape, dtype=cols.dtype)
    Ho, Wo, kh, kw = cols.shape[2:]
    for i in range(kh):
        for j in range(kw):
            hi, wj = i * d[0], j * d[1]
            out[:, :, hi : hi + s[0] * (Ho - 1) + 1 : s[0], wj : wj + s[1] * (Wo - 1) + 1 : s[1]] += cols[..., i, j]
    return out


def _out_size(n, k, s, p, d):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """x (N, C, H, W) * weight (O, C, kh, kw) -> (N, O, Ho, Wo). Cross-correlation, as in PyTorch.

    im2col: copy every receptive field into one row of a matrix (N·Ho·Wo, C·kh·kw),
    then the convolution is a single matrix multiply with W reshaped to (O, C·kh·kw).
    """
    s, p, d = _pair(stride), _pair(padding), _pair(dilation)
    O, C, kh, kw = weight.shape
    N = x.shape[0]
    pw = ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1]))
    xp = np.pad(x.data, pw)
    Ho = _out_size(x.shape[2], kh, s[0], p[0], d[0])
    Wo = _out_size(x.shape[3], kw, s[1], p[1], d[1])
    cols = _windows(xp, kh, kw, s, d, Ho, Wo).transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * kh * kw)
    w2 = weight.data.reshape(O, -1)
    out = (cols @ w2.T).reshape(N, Ho, Wo, O).transpose(0, 3, 1, 2)

    def grad_fn(g):  # g: (N, O, Ho, Wo)
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, O)
        gw = (g2.T @ cols).reshape(weight.shape)  # dL/dW = gᵀ · cols
        gcols = (g2 @ w2).reshape(N, Ho, Wo, C, kh, kw).transpose(0, 3, 1, 2, 4, 5)  # dL/dcols = g · W
        gxp = _scatter(gcols, xp.shape, s, d)  # col2im: windows overlap, so their gradients add
        gx = gxp[:, :, p[0] : p[0] + x.shape[2], p[1] : p[1] + x.shape[3]]
        return gx, gw

    out = Tensor._make(np.ascontiguousarray(out), (x, weight), grad_fn, "conv2d")
    return out if bias is None else out + bias.reshape(1, -1, 1, 1)


def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """x (N, C, L) * weight (O, C, k) -> (N, O, Lo), via conv2d with height 1."""
    out = conv2d(x.unsqueeze(2), weight.unsqueeze(2), None, (1, stride), (0, padding), (1, dilation))
    out = out.squeeze(2)
    return out if bias is None else out + bias.reshape(1, -1, 1)


def conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, dilation=1):
    """'Deconvolution' (upsampling). x (N, Cin, H, W), weight (Cin, Cout, kh, kw).

    It is the backward pass of conv2d with respect to its input, run forwards:
    every input pixel stamps a weighted copy of the kernel onto the output.
    """
    s, p, op, d = _pair(stride), _pair(padding), _pair(output_padding), _pair(dilation)
    N, _, H, W = x.shape
    _, Cout, kh, kw = weight.shape
    Hf = (H - 1) * s[0] + d[0] * (kh - 1) + 1 + op[0]
    Wf = (W - 1) * s[1] + d[1] * (kw - 1) + 1 + op[1]
    w = weight.data
    cols = np.tensordot(x.data, w, axes=([1], [0])).transpose(0, 3, 1, 2, 4, 5)  # (N, Cout, H, W, kh, kw)
    full = _scatter(cols, (N, Cout, Hf, Wf), s, d)
    out = full[:, :, p[0] : Hf - p[0], p[1] : Wf - p[1]]

    def grad_fn(g):
        gfull = np.zeros((N, Cout, Hf, Wf), dtype=g.dtype)
        gfull[:, :, p[0] : Hf - p[0], p[1] : Wf - p[1]] = g
        gcols = _windows(gfull, kh, kw, s, d, H, W)  # (N, Cout, H, W, kh, kw)
        gx = np.tensordot(gcols, w, axes=([1, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)
        gw = np.tensordot(x.data, gcols, axes=([0, 2, 3], [0, 2, 3]))  # (Cin, Cout, kh, kw)
        return gx, gw

    out = Tensor._make(np.ascontiguousarray(out), (x, weight), grad_fn, "conv_transpose2d")
    return out if bias is None else out + bias.reshape(1, -1, 1, 1)


def max_pool2d(x, kernel_size, stride=None, padding=0):
    k = _pair(kernel_size)
    s = _pair(stride if stride is not None else kernel_size)
    p = _pair(padding)
    pw = ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1]))
    xp = np.pad(x.data, pw, constant_values=-np.inf)
    N, C = x.shape[:2]
    Ho, Wo = _out_size(x.shape[2], k[0], s[0], p[0], 1), _out_size(x.shape[3], k[1], s[1], p[1], 1)
    win = _windows(xp, k[0], k[1], s, (1, 1), Ho, Wo).reshape(N, C, Ho, Wo, -1)
    arg = win.argmax(axis=-1)
    out = np.take_along_axis(win, arg[..., None], axis=-1)[..., 0]

    def grad_fn(g):  # only the max of each window gets the gradient
        gwin = np.zeros(win.shape, dtype=g.dtype)
        np.put_along_axis(gwin, arg[..., None], g[..., None], axis=-1)
        gxp = _scatter(gwin.reshape(N, C, Ho, Wo, k[0], k[1]), xp.shape, s, (1, 1))
        return (gxp[:, :, p[0] : p[0] + x.shape[2], p[1] : p[1] + x.shape[3]],)

    return Tensor._make(out, (x,), grad_fn, "max_pool2d")


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    k = _pair(kernel_size)
    s = _pair(stride if stride is not None else kernel_size)
    p = _pair(padding)
    pw = ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1]))
    xp = np.pad(x.data, pw)
    N, C = x.shape[:2]
    Ho, Wo = _out_size(x.shape[2], k[0], s[0], p[0], 1), _out_size(x.shape[3], k[1], s[1], p[1], 1)
    out = _windows(xp, k[0], k[1], s, (1, 1), Ho, Wo).mean(axis=(-2, -1))

    def grad_fn(g):  # every pixel of a window gets 1/(kh·kw) of its gradient
        gcols = np.broadcast_to((g / (k[0] * k[1]))[..., None, None], (N, C, Ho, Wo, k[0], k[1]))
        gxp = _scatter(gcols, xp.shape, s, (1, 1))
        return (gxp[:, :, p[0] : p[0] + x.shape[2], p[1] : p[1] + x.shape[3]],)

    return Tensor._make(out, (x,), grad_fn, "avg_pool2d")


def max_pool1d(x, kernel_size, stride=None, padding=0):
    return max_pool2d(x.unsqueeze(2), (1, kernel_size), (1, stride or kernel_size), (0, padding)).squeeze(2)


def avg_pool1d(x, kernel_size, stride=None, padding=0):
    return avg_pool2d(x.unsqueeze(2), (1, kernel_size), (1, stride or kernel_size), (0, padding)).squeeze(2)


def adaptive_avg_pool2d(x, output_size=1):
    """Average-pool to a fixed output size (H and W must be multiples of it)."""
    oh, ow = _pair(output_size)
    N, C, H, W = x.shape
    assert H % oh == 0 and W % ow == 0, "adaptive_avg_pool2d needs H, W divisible by output_size"
    return x.reshape(N, C, oh, H // oh, ow, W // ow).mean(axis=(3, 5))


def scaled_dot_product_attention(q, k, v, mask=None):
    """softmax(q kᵀ / sqrt(d)) v. `mask` is True where attention is NOT allowed."""
    scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    return softmax(scores, axis=-1) @ v
