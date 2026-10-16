import numpy as np
import pytest

import mytorcha as mt
from mytorcha.gradcheck import gradcheck, rand


def test_broadcast_add_sums_gradient_back():
    x = mt.Tensor(np.ones((4, 3)), requires_grad=True)
    b = mt.Tensor(np.zeros(3), requires_grad=True)
    (x + b).sum().backward()
    assert b.grad.tolist() == [4.0, 4.0, 4.0]


@pytest.mark.parametrize(
    "f",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: (a * b + a) ** 2,
        lambda a, b: a.exp() * b.abs(),
        lambda a, b: (a * a + 1).log() + b.sqrt(),
        lambda a, b: mt.where(a > 0, a, b),
        lambda a, b: mt.concat([a, b], axis=1),
        lambda a, b: mt.stack([a, b], axis=0).max(axis=0),
    ],
)
def test_elementwise(f):
    a, b = rand(3, 4, seed=1), rand(3, 4, seed=2, positive=True)
    assert gradcheck(lambda: f(a, b), [a, b])


def test_broadcasting_ops():
    a, b = rand(2, 3, 4), rand(1, 4, seed=3, positive=True)
    assert gradcheck(lambda: a * b + a / b - b, [a, b])


@pytest.mark.parametrize("shapes", [((3, 4), (4, 5)), ((2, 3, 4), (4, 2)), ((4,), (4, 3)), ((3, 4), (4,))])
def test_matmul(shapes):
    a, b = rand(*shapes[0], seed=1), rand(*shapes[1], seed=2)
    assert gradcheck(lambda: a @ b, [a, b])


@pytest.mark.parametrize("axis", [None, 0, 1, (0, 2)])
def test_reductions(axis):
    a = rand(2, 3, 4)
    assert gradcheck(lambda: a.sum(axis=axis) * 2, [a])
    assert gradcheck(lambda: a.mean(axis=axis), [a])
    assert gradcheck(lambda: a.max(axis=axis), [a])
    assert gradcheck(lambda: a.var(axis=axis), [a])


def test_shape_ops():
    a = rand(2, 3, 4)
    assert gradcheck(lambda: a.reshape(6, 4) * mt.arange(4), [a])
    assert gradcheck(lambda: a.transpose(2, 0, 1) * mt.arange(3), [a])
    assert gradcheck(lambda: a[:, 1:, ::2] * 3, [a])
    assert gradcheck(lambda: a.pad(((0, 0), (1, 1), (2, 0))) ** 2, [a])
    assert gradcheck(lambda: a.masked_fill(a.data > 0, 0.0) * 2, [a])


def test_fancy_index_accumulates_repeats():  # this is how Embedding learns
    w = mt.Tensor(np.zeros((5, 2)), requires_grad=True)
    w[np.array([1, 1, 3])].sum().backward()
    assert w.grad[:, 0].tolist() == [0, 2, 0, 1, 0]


def test_no_grad_records_nothing():
    a = rand(3)
    with mt.no_grad():
        b = a * 2
    assert not b.requires_grad and b._prev == ()


def test_grad_accumulates_until_zeroed():
    a = mt.Tensor([1.0, 2.0], requires_grad=True)
    (a * 3).sum().backward()
    (a * 3).sum().backward()
    assert a.grad.tolist() == [6.0, 6.0]


def test_numpy_on_the_left():
    a = rand(3)
    out = np.ones(3) * a + np.ones(3)
    assert isinstance(out, mt.Tensor) and out.requires_grad
