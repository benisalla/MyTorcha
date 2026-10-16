"""Our forward passes must match PyTorch's definitions (skipped if torch is absent)."""

import numpy as np
import pytest

import mytorcha.functional as F
from mytorcha import Tensor

torch = pytest.importorskip("torch")
TF = torch.nn.functional
rng = np.random.default_rng(0)


def both(*shape):
    a = rng.standard_normal(shape)
    return Tensor(a, requires_grad=True), torch.tensor(a, requires_grad=True)


PAIRS = {
    "relu": TF.relu, "relu6": TF.relu6, "leaky_relu": TF.leaky_relu, "elu": TF.elu, "celu": TF.celu,
    "selu": TF.selu, "sigmoid": torch.sigmoid, "logsigmoid": TF.logsigmoid,
    "hard_sigmoid": TF.hardsigmoid, "tanh": torch.tanh, "hardtanh": TF.hardtanh,
    "softplus": TF.softplus, "softsign": TF.softsign, "silu": TF.silu, "hardswish": TF.hardswish,
    "mish": TF.mish, "gelu": TF.gelu, "arctan": torch.atan,
}


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_activations_match_torch(name):
    m, t = both(50)
    m_out, t_out = F.get_activation(name)(m), PAIRS[name](t)
    np.testing.assert_allclose(m_out.data, t_out.detach().numpy(), atol=1e-6)
    m_out.sum().backward()
    t_out.sum().backward()
    np.testing.assert_allclose(m.grad, t.grad.numpy(), atol=1e-6)


def test_losses_match_torch():
    (m, t), y = both(6, 4), rng.integers(0, 4, 6)
    np.testing.assert_allclose(F.cross_entropy(m, y, label_smoothing=0.1).data,
                               TF.cross_entropy(t, torch.tensor(y), label_smoothing=0.1).item(), atol=1e-6)
    yb = rng.random((6, 4))
    np.testing.assert_allclose(F.binary_cross_entropy_with_logits(m, yb).data,
                               TF.binary_cross_entropy_with_logits(t, torch.tensor(yb)).item(), atol=1e-6)
    np.testing.assert_allclose(F.huber_loss(m, yb, delta=0.5).data,
                               TF.huber_loss(t, torch.tensor(yb), delta=0.5).item(), atol=1e-6)
    np.testing.assert_allclose(F.hinge_loss(m, y).data,
                               4 * TF.multi_margin_loss(t, torch.tensor(y)).item(), atol=1e-6)


def test_conv_and_pool_match_torch():
    (mx, tx), (mw, tw) = both(2, 3, 9, 8), both(5, 3, 3, 3)
    m = F.conv2d(mx, mw, stride=2, padding=1, dilation=2)
    t = TF.conv2d(tx, tw, stride=2, padding=1, dilation=2)
    np.testing.assert_allclose(m.data, t.detach().numpy(), atol=1e-10)
    (mw2, tw2) = both(3, 4, 3, 3)
    m = F.conv_transpose2d(mx, mw2, stride=2, padding=1, output_padding=1)
    t = TF.conv_transpose2d(tx, tw2, stride=2, padding=1, output_padding=1)
    np.testing.assert_allclose(m.data, t.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(F.max_pool2d(mx, 3, 2, 1).data, TF.max_pool2d(tx, 3, 2, 1).detach().numpy())
    np.testing.assert_allclose(F.avg_pool2d(mx, 3, 2, 1).data, TF.avg_pool2d(tx, 3, 2, 1).detach().numpy(),
                               atol=1e-12)


def _copy(dst_torch_param, src):
    with torch.no_grad():
        dst_torch_param.copy_(torch.tensor(src.data, dtype=torch.float64))


@pytest.mark.parametrize("name", ["RNN", "GRU", "LSTM"])
def test_recurrent_layers_match_torch(name):
    import mytorcha.nn as nn

    ours = getattr(nn, name)(3, 5, num_layers=2, bidirectional=True)
    theirs = getattr(torch.nn, name)(3, 5, num_layers=2, bidirectional=True, batch_first=True).double()
    for layer in range(2):
        for d, suffix in enumerate(["", "_reverse"]):
            cell = ours.cells[layer * 2 + d]
            for w in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
                _copy(getattr(theirs, f"{w}_l{layer}{suffix}"), getattr(cell, w))
    for p in ours.parameters():
        p.data = p.data.astype(np.float64)
    (mx, tx) = both(4, 6, 3)
    m_out, m_state = ours(mx)
    t_out, t_state = theirs(tx)
    np.testing.assert_allclose(m_out.data, t_out.detach().numpy(), atol=1e-10)
    m_h = m_state[0] if name == "LSTM" else m_state
    t_h = t_state[0] if name == "LSTM" else t_state
    np.testing.assert_allclose(m_h.data, t_h.detach().numpy(), atol=1e-10)
    m_out.sum().backward()
    t_out.sum().backward()
    np.testing.assert_allclose(mx.grad, tx.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.cells[0].weight_hh.grad, theirs.weight_hh_l0.grad.numpy(), atol=1e-10)


def test_batchnorm_matches_torch_in_train_and_eval():
    import mytorcha.nn as nn

    ours, theirs = nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3).double()
    for p in ours.parameters():
        p.data = p.data.astype(np.float64)
    for _ in range(3):
        mx, tx = both(4, 3, 5, 5)
        np.testing.assert_allclose(ours(mx).data, theirs(tx).detach().numpy(), atol=1e-6)
    np.testing.assert_allclose(ours.running_mean, theirs.running_mean.numpy(), atol=1e-6)
    np.testing.assert_allclose(ours.running_var, theirs.running_var.numpy(), atol=1e-6)
    ours.eval(), theirs.eval()
    mx, tx = both(2, 3, 5, 5)
    np.testing.assert_allclose(ours(mx).data, theirs(tx).detach().numpy(), atol=1e-5)


def test_multihead_attention_matches_torch():
    import mytorcha.nn as nn

    ours = nn.MultiheadAttention(8, 2)
    theirs = torch.nn.MultiheadAttention(8, 2, batch_first=True).double()
    w = np.concatenate([ours.q_proj.weight.data, ours.k_proj.weight.data, ours.v_proj.weight.data])
    b = np.concatenate([ours.q_proj.bias.data, ours.k_proj.bias.data, ours.v_proj.bias.data])
    _copy(theirs.in_proj_weight, mt_tensor(w))
    _copy(theirs.in_proj_bias, mt_tensor(b))
    _copy(theirs.out_proj.weight, ours.out_proj.weight)
    _copy(theirs.out_proj.bias, ours.out_proj.bias)
    for p in ours.parameters():
        p.data = p.data.astype(np.float64)
    mx, tx = both(2, 5, 8)
    mask = np.triu(np.ones((5, 5), bool), 1)
    t_out, _ = theirs(tx, tx, tx, attn_mask=torch.tensor(mask))
    np.testing.assert_allclose(ours(mx, causal=True).data, t_out.detach().numpy(), atol=1e-10)


def mt_tensor(a):
    return Tensor(a)
