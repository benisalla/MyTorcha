import math

import pytest

from mytorcha.scalar import MLP, Node, sum_nodes

FUNCS = ["exp", "log", "relu", "leaky_relu", "elu", "sigmoid", "tanh", "softplus", "arctan", "gelu"]


@pytest.mark.parametrize("name", FUNCS)
@pytest.mark.parametrize("x0", [-1.7, 0.6, 2.3])
def test_unary_grad_matches_finite_difference(name, x0):
    if name == "log" and x0 <= 0:
        pytest.skip("log needs x > 0")
    x = Node(x0)
    getattr(x, name)().backward()
    h = 1e-6
    numeric = (getattr(Node(x0 + h), name)().data - getattr(Node(x0 - h), name)().data) / (2 * h)
    assert x.grad == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_log_grad_is_one_over_x():  # was 1/log(x) in v1
    x = Node(2.0)
    x.log().backward()
    assert x.grad == pytest.approx(0.5)


def test_equal_values_are_still_different_nodes():  # v1 merged them in sets
    a, b = Node(3.0), Node(3.0)
    y = a * b + b
    y.backward()
    assert len({a, b}) == 2
    assert (a.grad, b.grad) == (3.0, 4.0)


def test_reused_node_accumulates_gradient():
    x = Node(3.0)
    (x * x).backward()
    assert x.grad == 6.0


def test_rdiv_and_rsub():
    x = Node(4.0)
    y = 1 / x + (10 - x)
    y.backward()
    assert y.data == pytest.approx(6.25)
    assert x.grad == pytest.approx(-1 / 16 - 1)


def test_deep_graph_does_not_hit_recursion_limit():
    s = sum_nodes([Node(1.0) for _ in range(5000)])
    s.backward()
    assert s.data == 5000


def test_mlp_learns_xor():
    import random

    random.seed(0)
    X, Y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
    model = MLP(2, [8], 1, act="tanh")
    for _ in range(300):
        loss = sum_nodes([(model(x)[0] - y) ** 2 for x, y in zip(X, Y)])
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
    assert loss.data < 0.01
    assert [round(model(x)[0].data) for x in X] == Y
    assert math.isfinite(loss.data)
