import numpy as np
import pytest

import mytorcha as mt
from mytorcha import optim

torch = pytest.importorskip("torch")

CASES = [
    (dict(cls="SGD", lr=0.1), {}),
    (dict(cls="SGD", lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01), {}),
    (dict(cls="SGD", lr=0.1, momentum=0.9, dampening=0.1), {}),
    (dict(cls="Adagrad", lr=0.1, weight_decay=0.01), {}),
    (dict(cls="RMSprop", lr=0.01), {}),
    (dict(cls="RMSprop", lr=0.01, momentum=0.9, centered=True), {}),
    (dict(cls="Adadelta", lr=1.0), {}),
    (dict(cls="Adam", lr=0.01, weight_decay=0.01), {}),
    (dict(cls="Adam", lr=0.01, amsgrad=True), {}),
    (dict(cls="AdamW", lr=0.01, weight_decay=0.1), {}),
    (dict(cls="Adamax", lr=0.01), {}),
]


@pytest.mark.parametrize("cfg,_", CASES, ids=lambda c: str(c))
def test_matches_torch(cfg, _):
    cfg = dict(cfg)
    name = cfg.pop("cls")
    w0 = np.random.default_rng(0).standard_normal((3, 4))
    target = np.random.default_rng(1).standard_normal((3, 4))

    w = mt.nn.Parameter(w0)
    w.data = w0.copy()  # float64 for an exact comparison
    opt = getattr(optim, name)([w], **cfg)
    tw = torch.tensor(w0.copy(), requires_grad=True)
    topt = getattr(torch.optim, name)([tw], **cfg)
    for _ in range(10):
        opt.zero_grad()
        (((w - target) ** 4).sum() * 0.1 + (w * w).sum()).backward()
        opt.step()
        topt.zero_grad()
        (((tw - torch.tensor(target)) ** 4).sum() * 0.1 + (tw * tw).sum()).backward()
        topt.step()
    assert np.isfinite(w.data).all() and not np.allclose(w.data, w0)  # really moved, no NaN
    np.testing.assert_allclose(w.data, tw.detach().numpy(), rtol=1e-6, atol=1e-8)


def test_every_parameter_has_its_own_state():  # the v1 bug
    a, b = mt.nn.Parameter([1.0]), mt.nn.Parameter([1.0])
    opt = optim.Adam([a, b], lr=0.1)
    a.grad, b.grad = np.array([1.0], np.float32), np.array([-1.0], np.float32)
    opt.step()
    assert a.data[0] < 1 < b.data[0]


@pytest.mark.parametrize("name", ["SGD", "Adagrad", "RMSprop", "Adadelta", "Adam", "AdamW", "Adamax", "Lion"])
def test_minimises_quadratic(name):
    w = mt.nn.Parameter(np.full(5, 3.0))
    lr = {"Lion": 0.05, "Adadelta": 10.0, "SGD": 0.1}.get(name, 0.3)
    opt = getattr(optim, name)([w], lr=lr)
    for _ in range(300):
        opt.zero_grad()
        (w * w).sum().backward()
        opt.step()
    assert np.abs(w.data).max() < 0.2


def test_schedulers_and_clipping():
    w = mt.nn.Parameter([1.0])
    opt = optim.SGD([w], lr=1.0)
    s = optim.StepLR(opt, step_size=2, gamma=0.5)
    assert [s.step() for _ in range(4)] == [1.0, 0.5, 0.5, 0.25]
    c = optim.CosineAnnealingLR(optim.SGD([w], lr=1.0), T_max=10)
    lrs = [c.step() for _ in range(10)]
    assert lrs[4] == pytest.approx(0.5) and lrs[-1] == pytest.approx(0.0)
    w.grad = np.array([30.0, 40.0][:1], np.float32)
    w2 = mt.nn.Parameter([0.0])
    w2.grad = np.array([40.0], np.float32)
    w.grad = np.array([30.0], np.float32)
    assert optim.clip_grad_norm_([w, w2], 1.0) == pytest.approx(50.0)
    assert np.hypot(w.grad[0], w2.grad[0]) == pytest.approx(1.0, rel=1e-4)
