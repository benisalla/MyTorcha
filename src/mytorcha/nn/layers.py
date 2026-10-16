"""Layers: dense, convolution, pooling, normalisation, dropout, embedding, activations."""

import numpy as np

from .. import functional as F
from .. import init
from ..tensor import Tensor
from .module import Module, Parameter

# ---------------------------------------------------------------------------
# dense and convolution
# ---------------------------------------------------------------------------


class Linear(Module):
    """y = x Wᵀ + b. W: (out_features, in_features)."""

    def __init__(self, in_features, out_features, bias=True):
        self.in_features, self.out_features = in_features, out_features
        self.weight = init.default_uniform_(Parameter.empty(out_features, in_features), in_features)
        self.bias = init.default_uniform_(Parameter.empty(out_features), in_features) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class _ConvNd(Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, dilation, bias, transposed=False):
        self.in_channels, self.out_channels, self.kernel_size = in_ch, out_ch, kernel
        self.stride, self.padding, self.dilation = stride, padding, dilation
        shape = (in_ch, out_ch, *kernel) if transposed else (out_ch, in_ch, *kernel)
        fan_in = shape[1] * int(np.prod(kernel))
        self.weight = init.default_uniform_(Parameter.empty(*shape), fan_in)
        self.bias = init.default_uniform_(Parameter.empty(out_ch), fan_in) if bias else None

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class Conv1d(_ConvNd):
    """(N, C_in, L) -> (N, C_out, L_out)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, (kernel_size,), stride, padding, dilation, bias)

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class Conv2d(_ConvNd):
    """(N, C_in, H, W) -> (N, C_out, H_out, W_out)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, F._pair(kernel_size), stride, padding, dilation, bias)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class ConvTranspose2d(_ConvNd):
    """Learned upsampling (decoders, GANs, U-Nets). stride=2 roughly doubles H and W."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 dilation=1, bias=True):
        super().__init__(in_channels, out_channels, F._pair(kernel_size), stride, padding, dilation, bias,
                         transposed=True)
        self.output_padding = output_padding

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  self.output_padding, self.dilation)


# ---------------------------------------------------------------------------
# pooling and reshaping
# ---------------------------------------------------------------------------


class _Pool(Module):
    fn = None

    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, x):
        return type(self).fn(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride or self.kernel_size}"


class MaxPool1d(_Pool):
    fn = staticmethod(F.max_pool1d)


class MaxPool2d(_Pool):
    fn = staticmethod(F.max_pool2d)


class AvgPool1d(_Pool):
    fn = staticmethod(F.avg_pool1d)


class AvgPool2d(_Pool):
    fn = staticmethod(F.avg_pool2d)


class AdaptiveAvgPool2d(Module):
    """AdaptiveAvgPool2d(1) = global average pooling: (N, C, H, W) -> (N, C, 1, 1)."""

    def __init__(self, output_size=1):
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, self.output_size)


class Flatten(Module):
    """(N, C, H, W) -> (N, C·H·W)."""

    def __init__(self, start_dim=1):
        self.start_dim = start_dim

    def forward(self, x):
        return x.flatten(self.start_dim)


class Identity(Module):
    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# regularisation and normalisation
# ---------------------------------------------------------------------------


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, self.training)

    def extra_repr(self):
        return f"p={self.p}"


class _BatchNorm(Module):
    """Normalise each channel with the batch's mean/var while training, and with
    running averages of them at evaluation time (call model.eval()!)."""

    _buffers = ("running_mean", "running_var")
    axes = None  # reduction axes, set by subclasses

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.weight = Parameter(np.ones(num_features)) if affine else None
        self.bias = Parameter(np.zeros(num_features)) if affine else None
        self.running_mean = np.zeros(num_features, np.float32)
        self.running_var = np.ones(num_features, np.float32)

    def _shape(self, x):
        return (1, -1) + (1,) * (x.ndim - 2)

    def forward(self, x):
        axes = (0,) + tuple(range(2, x.ndim))
        shape = self._shape(x)
        if self.training:
            mean, var = x.mean(axis=axes, keepdims=True), x.var(axis=axes, keepdims=True)
            n = x.size / self.num_features
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mean.data.reshape(-1)
            self.running_var = (1 - m) * self.running_var + m * var.data.reshape(-1) * n / max(n - 1, 1)
            x_hat = (x - mean) / (var + self.eps).sqrt()
        else:
            mean = self.running_mean.reshape(shape).astype(x.dtype)
            std = np.sqrt(self.running_var.reshape(shape) + self.eps).astype(x.dtype)
            x_hat = (x - mean) / std
        if self.weight is not None:
            x_hat = x_hat * self.weight.reshape(shape) + self.bias.reshape(shape)
        return x_hat

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"


class BatchNorm1d(_BatchNorm):
    """Input (N, C) or (N, C, L)."""


class BatchNorm2d(_BatchNorm):
    """Input (N, C, H, W)."""


class LayerNorm(Module):
    """Normalise over the last dims of each sample (Transformers, RNNs)."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.normalized_shape, self.eps = shape, eps
        self.weight = Parameter(np.ones(shape)) if elementwise_affine else None
        self.bias = Parameter(np.zeros(shape)) if elementwise_affine else None

    def forward(self, x):
        axes = tuple(range(-len(self.normalized_shape), 0))
        mean, var = x.mean(axis=axes, keepdims=True), x.var(axis=axes, keepdims=True)
        x_hat = (x - mean) / (var + self.eps).sqrt()
        return x_hat * self.weight + self.bias if self.weight is not None else x_hat

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}"


class GroupNorm(Module):
    """Split C channels into groups and normalise each group per sample (batch-size independent)."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        assert num_channels % num_groups == 0
        self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
        self.weight = Parameter(np.ones(num_channels)) if affine else None
        self.bias = Parameter(np.zeros(num_channels)) if affine else None

    def forward(self, x):
        N = x.shape[0]
        g = x.reshape(N, self.num_groups, -1)
        g = (g - g.mean(axis=2, keepdims=True)) / (g.var(axis=2, keepdims=True) + self.eps).sqrt()
        x_hat = g.reshape(x.shape)
        if self.weight is None:
            return x_hat
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x_hat * self.weight.reshape(shape) + self.bias.reshape(shape)


class Embedding(Module):
    """A lookup table: integer ids (any shape) -> vectors (..., embedding_dim)."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        self.num_embeddings, self.embedding_dim, self.padding_idx = num_embeddings, embedding_dim, padding_idx
        self.weight = init.normal_(Parameter.empty(num_embeddings, embedding_dim))
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0

    def forward(self, ids):
        ids = ids.data if isinstance(ids, Tensor) else np.asarray(ids)
        out = self.weight[ids.astype(np.int64)]
        if self.padding_idx is not None:
            out = out * (ids != self.padding_idx)[..., None]  # the padding row never learns
        return out

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}"


# ---------------------------------------------------------------------------
# activations as modules
# ---------------------------------------------------------------------------


def _activation_module(name, fn, doc):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def forward(self, x):
        return fn(x, **self.kwargs)

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.kwargs.items())

    return type(name, (Module,), {"__init__": __init__, "forward": forward, "extra_repr": extra_repr,
                                  "__doc__": doc})


ReLU = _activation_module("ReLU", F.relu, "max(0, x)")
ReLU6 = _activation_module("ReLU6", F.relu6, "min(max(0, x), 6)")
LeakyReLU = _activation_module("LeakyReLU", F.leaky_relu, "x if x > 0 else negative_slope·x")
ELU = _activation_module("ELU", F.elu, "x if x > 0 else alpha·(eˣ - 1)")
CELU = _activation_module("CELU", F.celu, "x if x > 0 else alpha·(e^(x/alpha) - 1)")
SELU = _activation_module("SELU", F.selu, "scale·ELU(x, 1.673…), self-normalising")
GELU = _activation_module("GELU", F.gelu, "x·Φ(x); approximate='tanh' for the GPT-2 formula")
Sigmoid = _activation_module("Sigmoid", F.sigmoid, "1 / (1 + e⁻ˣ)")
LogSigmoid = _activation_module("LogSigmoid", F.logsigmoid, "log(sigmoid(x))")
Hardsigmoid = _activation_module("Hardsigmoid", F.hard_sigmoid, "clip(x/6 + 1/2, 0, 1)")
Tanh = _activation_module("Tanh", F.tanh, "tanh(x)")
Hardtanh = _activation_module("Hardtanh", F.hardtanh, "clip(x, -1, 1)")
Softplus = _activation_module("Softplus", F.softplus, "log(1 + eˣ), a smooth ReLU")
Softsign = _activation_module("Softsign", F.softsign, "x / (1 + |x|)")
SiLU = _activation_module("SiLU", F.silu, "x·sigmoid(x), a.k.a. swish")
Swish = SiLU
Hardswish = _activation_module("Hardswish", F.hardswish, "x·clip(x + 3, 0, 6) / 6 (MobileNetV3)")
Mish = _activation_module("Mish", F.mish, "x·tanh(softplus(x))")
Arctan = _activation_module("Arctan", F.arctan, "arctan(x)")
Softmax = _activation_module("Softmax", F.softmax, "probabilities along `axis` (default last)")
LogSoftmax = _activation_module("LogSoftmax", F.log_softmax, "log of softmax, computed stably")
Softmin = _activation_module("Softmin", F.softmin, "softmax(-x)")


class PReLU(Module):
    """Leaky ReLU whose negative slope is learned (one per channel, or shared)."""

    def __init__(self, num_parameters=1, init_value=0.25):
        self.weight = Parameter(np.full(num_parameters, init_value))

    def forward(self, x):
        w = self.weight if self.weight.size == 1 else self.weight.reshape((1, -1) + (1,) * (x.ndim - 2))
        return F.prelu(x, w)
