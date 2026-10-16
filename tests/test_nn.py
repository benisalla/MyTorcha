import numpy as np
import pytest

import mytorcha as mt
from mytorcha import nn
from mytorcha.gradcheck import gradcheck, rand


def f64(module):
    """float64 copies of the weights, so finite differences are precise."""
    for p in module.parameters():
        p.data = p.data.astype(np.float64)
    return module


LAYERS = [
    (lambda: nn.Linear(4, 3), (2, 4)),
    (lambda: nn.Conv1d(3, 2, 3, stride=2, padding=1), (2, 3, 7)),
    (lambda: nn.Conv2d(2, 3, 3, padding=1), (2, 2, 5, 5)),
    (lambda: nn.ConvTranspose2d(2, 3, 3, stride=2, padding=1, output_padding=1), (2, 2, 3, 3)),
    (lambda: nn.BatchNorm1d(4), (6, 4)),
    (lambda: nn.BatchNorm2d(2), (3, 2, 3, 3)),
    (lambda: nn.LayerNorm(4), (2, 3, 4)),
    (lambda: nn.GroupNorm(2, 4), (2, 4, 3)),
    (lambda: nn.PReLU(), (2, 5)),
    (lambda: nn.MultiheadAttention(4, 2), (2, 3, 4)),
    (lambda: nn.TransformerEncoderLayer(4, 2, 8), (2, 3, 4)),
]


@pytest.mark.parametrize("make,shape", LAYERS)
def test_layer_gradients(make, shape):
    mt.manual_seed(0)
    layer = f64(make())
    x = rand(*shape)
    w = np.random.default_rng(1).standard_normal(layer(x).shape)  # random projection of the output
    assert gradcheck(lambda: layer(x) * w, [x] + layer.parameters())


@pytest.mark.parametrize("cls", [nn.RNN, nn.GRU, nn.LSTM])
def test_recurrent_gradients(cls):
    mt.manual_seed(0)
    layer = f64(cls(3, 4, num_layers=2, bidirectional=True))
    x = rand(2, 5, 3)
    w = np.random.default_rng(1).standard_normal((2, 5, 8))
    assert gradcheck(lambda: layer(x)[0] * w, [x] + layer.parameters()[:4])


def test_embedding_and_padding_idx():
    emb = nn.Embedding(5, 3, padding_idx=0)
    out = emb(np.array([[0, 2, 2]]))
    out.sum().backward()
    assert out.shape == (1, 3, 3)
    assert emb.weight.grad[0].tolist() == [0, 0, 0] and emb.weight.grad[2].tolist() == [2, 2, 2]


def test_parameters_found_through_lists_and_tied_weights_once():
    class Net(nn.Module):
        def __init__(self):
            self.blocks = [nn.Linear(2, 2), nn.Linear(2, 2)]
            self.head = self.blocks[0]  # tied

    assert len(Net().parameters()) == 4


def test_train_eval_dropout_and_batchnorm_running_stats():
    m = nn.Sequential(nn.BatchNorm1d(3), nn.Dropout(0.5))
    x = mt.Tensor(np.random.default_rng(0).normal(5, 2, (64, 3)))
    m(x)
    assert np.all(m[0].running_mean > 0.4)  # moved 10% towards ~5
    m.eval()
    assert not m[1].training
    np.testing.assert_array_equal(m(x).data, m(x).data)  # deterministic in eval


def test_state_dict_roundtrip(tmp_path):
    mt.manual_seed(0)
    a = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    a(mt.randn(8, 3))
    a.save(tmp_path / "m.npz")
    mt.manual_seed(1)
    b = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).load(tmp_path / "m.npz")
    for (n1, v1), (n2, v2) in zip(a.state_dict().items(), b.state_dict().items()):
        assert n1 == n2
        np.testing.assert_array_equal(v1, v2)


def test_dataloader_batches():
    ds = mt.TensorDataset(np.arange(10.0)[:, None], np.arange(10))
    batches = list(mt.DataLoader(ds, batch_size=4, shuffle=True))
    assert [len(b[0]) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate([b[1].data for b in batches]).tolist()) == list(range(10))
    assert len(mt.DataLoader(ds, batch_size=4, drop_last=True)) == 2
