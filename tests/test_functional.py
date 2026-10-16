import numpy as np
import pytest

import mytorcha.functional as F
from mytorcha import Tensor
from mytorcha.gradcheck import gradcheck, rand

ACTS = sorted(set(F.ACTIVATIONS) - {"linear", "swish"})


@pytest.mark.parametrize("name", ACTS)
def test_activation_grads(name):
    # values away from the kinks (0, ±1, ±3, 6) where finite differences are undefined
    a = Tensor(np.array([-4.2, -2.1, -0.7, -0.3, 0.4, 0.8, 1.6, 2.5, 4.4, 7.0]), requires_grad=True)
    f = F.get_activation(name)
    assert gradcheck(lambda: f(a) * np.arange(1, 11), [a])


def test_gelu_tanh_and_prelu():
    a, w = rand(6), Tensor(np.array([0.25]), requires_grad=True)
    assert gradcheck(lambda: F.gelu(a, approximate="tanh"), [a])
    assert gradcheck(lambda: F.prelu(a, w) * np.arange(6), [a, w])


@pytest.mark.parametrize("fn", [F.softmax, F.log_softmax, F.softmin])
def test_softmax_family(fn):
    a = rand(3, 5)
    assert gradcheck(lambda: fn(a, axis=-1) * np.arange(5), [a])
    assert gradcheck(lambda: fn(a, axis=0) * np.arange(5), [a])


def test_softmax_is_stable():
    out = F.softmax(Tensor(np.array([1000.0, 1001.0])))
    assert np.isfinite(out.data).all() and out.data.sum() == pytest.approx(1)


def test_losses():
    p, y = rand(4, 3), rand(4, 3, seed=9, requires_grad=False)
    for loss in (F.mse_loss, F.l1_loss, F.huber_loss, F.smooth_l1_loss):
        assert gradcheck(lambda loss=loss: loss(p, y), [p])
    prob = Tensor(np.random.default_rng(0).uniform(0.1, 0.9, (4, 3)), requires_grad=True)
    yb = (np.random.default_rng(1).random((4, 3)) > 0.5).astype(float)
    assert gradcheck(lambda: F.binary_cross_entropy(prob, yb), [prob])
    assert gradcheck(lambda: F.binary_cross_entropy_with_logits(p, yb), [p])
    t = np.array([0, 2, 1, 2])
    assert gradcheck(lambda: F.cross_entropy(p, t), [p])
    assert gradcheck(lambda: F.cross_entropy(p, t, label_smoothing=0.1), [p])
    assert gradcheck(lambda: F.cross_entropy(p, np.array([0, -1, 1, -1]), ignore_index=-1), [p])
    assert gradcheck(lambda: F.hinge_loss(p, t), [p])
    target = F.softmax(rand(4, 3, seed=5, requires_grad=False)).data
    assert gradcheck(lambda: F.kl_div(F.log_softmax(p), target), [p])


def test_cross_entropy_on_sequences():  # (N, T, C) logits, classes last
    p = rand(2, 3, 4)
    assert gradcheck(lambda: F.cross_entropy(p, np.array([[0, 1, 2], [3, 3, 0]])), [p])


@pytest.mark.parametrize("stride,padding,dilation", [(1, 0, 1), (2, 1, 1), (1, 2, 2), ((2, 1), (0, 1), 1)])
def test_conv2d(stride, padding, dilation):
    x, w, b = rand(2, 3, 7, 6), rand(4, 3, 3, 2, seed=1), rand(4, seed=2)
    assert gradcheck(lambda: F.conv2d(x, w, b, stride, padding, dilation), [x, w, b])


def test_conv1d_and_transpose():
    x, w = rand(2, 3, 9), rand(4, 3, 3, seed=1)
    assert gradcheck(lambda: F.conv1d(x, w, None, 2, 1), [x, w])
    x, w, b = rand(2, 3, 4, 5), rand(3, 2, 3, 3, seed=1), rand(2, seed=2)
    assert gradcheck(lambda: F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1), [x, w, b])


@pytest.mark.parametrize("pool", [F.max_pool2d, F.avg_pool2d])
def test_pool2d(pool):
    x = rand(2, 3, 6, 7)
    assert gradcheck(lambda: pool(x, 2) * np.arange(3).reshape(1, 3, 1, 1), [x])
    assert gradcheck(lambda: pool(x, 3, stride=2, padding=1), [x])


def test_pool1d_adaptive_attention():
    x = rand(2, 3, 8)
    assert gradcheck(lambda: F.max_pool1d(x, 2), [x])
    assert gradcheck(lambda: F.avg_pool1d(x, 4, 2), [x])
    x4 = rand(2, 3, 4, 6)
    assert gradcheck(lambda: F.adaptive_avg_pool2d(x4, (2, 3)), [x4])
    q, k, v = rand(2, 4, 3, seed=1), rand(2, 5, 3, seed=2), rand(2, 5, 3, seed=3)
    mask = np.triu(np.ones((4, 5), bool), 1)
    assert gradcheck(lambda: F.scaled_dot_product_attention(q, k, v, mask), [q, k, v])


def test_dropout_keeps_expectation_and_is_off_in_eval():
    x = Tensor(np.ones((200, 200)))
    out = F.dropout(x, 0.3, training=True, rng=np.random.default_rng(0))
    assert abs(out.data.mean() - 1) < 0.02 and (out.data == 0).mean() == pytest.approx(0.3, abs=0.02)
    assert F.dropout(x, 0.3, training=False) is x
