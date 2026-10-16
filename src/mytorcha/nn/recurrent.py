"""Recurrent layers: RNN, LSTM, GRU (and their single-step cells).

Shapes are batch-first: input (N, T, input_size), output (N, T, hidden·dirs).
Weight names and gate order match PyTorch (LSTM: i, f, g, o; GRU: r, z, n).

A recurrent net is just the same cell applied T times; backprop then flows
backwards through those T copies ("backpropagation through time"). We precompute
x @ W_ihᵀ for all time steps in one matmul; only h @ W_hhᵀ must be sequential.
"""

import numpy as np

from .. import functional as F
from .. import init
from ..tensor import Tensor, concat, stack, zeros
from .module import Module, Parameter


class _Cell(Module):
    gates = 1

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size, self.hidden_size = input_size, hidden_size
        G = self.gates * hidden_size
        self.weight_ih = init.default_uniform_(Parameter.empty(G, input_size), hidden_size)
        self.weight_hh = init.default_uniform_(Parameter.empty(G, hidden_size), hidden_size)
        self.bias_ih = init.default_uniform_(Parameter.empty(G), hidden_size) if bias else None
        self.bias_hh = init.default_uniform_(Parameter.empty(G), hidden_size) if bias else None

    def project_input(self, x):
        """x @ W_ihᵀ + b_ih for every time step at once."""
        return F.linear(x, self.weight_ih, self.bias_ih)

    def project_hidden(self, h):
        return F.linear(h, self.weight_hh, self.bias_hh)

    def forward(self, x, state=None):
        """One time step on x: (N, input_size)."""
        if state is None:
            state = self.initial_state(x.shape[0])
        return self.step(self.project_input(x), state)

    def initial_state(self, n):
        return zeros(n, self.hidden_size)

    def extra_repr(self):
        return f"{self.input_size}, {self.hidden_size}"


class RNNCell(_Cell):
    """h' = tanh(x W_ihᵀ + b_ih + h W_hhᵀ + b_hh)."""

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, bias)
        self.nonlinearity = nonlinearity

    def step(self, xp, h):
        act = F.tanh if self.nonlinearity == "tanh" else F.relu
        return act(xp + self.project_hidden(h))


class LSTMCell(_Cell):
    """Long short-term memory: a cell state c that is changed only by gated addition,
    so gradients can flow over many steps without vanishing.

        i, f, o = σ(·)   input / forget / output gates
        g = tanh(·)      candidate values
        c' = f·c + i·g ;  h' = o·tanh(c')
    """

    gates = 4

    def initial_state(self, n):
        return zeros(n, self.hidden_size), zeros(n, self.hidden_size)

    def step(self, xp, state):
        h, c = state
        z = xp + self.project_hidden(h)
        H = self.hidden_size
        i, f = F.sigmoid(z[:, :H]), F.sigmoid(z[:, H : 2 * H])
        g, o = F.tanh(z[:, 2 * H : 3 * H]), F.sigmoid(z[:, 3 * H :])
        c = f * c + i * g
        return o * F.tanh(c), c


class GRUCell(_Cell):
    """Gated recurrent unit: an LSTM with fewer gates and no separate cell state.

        r, z = σ(·)                    reset / update gates
        n = tanh(W_in x + r·(W_hn h))  candidate
        h' = (1 - z)·n + z·h
    """

    gates = 3

    def step(self, xp, h):
        hp = self.project_hidden(h)
        H = self.hidden_size
        r = F.sigmoid(xp[:, :H] + hp[:, :H])
        z = F.sigmoid(xp[:, H : 2 * H] + hp[:, H : 2 * H])
        n = F.tanh(xp[:, 2 * H :] + r * hp[:, 2 * H :])
        return (1 - z) * n + z * h


class _Recurrent(Module):
    cell_class = None

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0,
                 bidirectional=False, **cell_kwargs):
        self.input_size, self.hidden_size, self.num_layers = input_size, hidden_size, num_layers
        self.dropout, self.bidirectional = dropout, bidirectional
        dirs = 2 if bidirectional else 1
        self.cells = [
            self.cell_class(input_size if layer == 0 else hidden_size * dirs, hidden_size, bias, **cell_kwargs)
            for layer in range(num_layers)
            for _ in range(dirs)
        ]

    def _run(self, cell, x, state, reverse):
        xp = cell.project_input(x)  # (N, T, G·H) in one matmul
        T = x.shape[1]
        outs = []
        for t in (reversed(range(T)) if reverse else range(T)):
            state = cell.step(xp[:, t], state)
            outs.append(state[0] if isinstance(state, tuple) else state)
        if reverse:
            outs.reverse()
        return stack(outs, axis=1), state

    def forward(self, x, state=None):
        """x: (N, T, input_size). state: per-layer initial states (as returned).
        Returns output (N, T, H·dirs) and the final state(s) h_n (L·dirs, N, H)."""
        dirs = 2 if self.bidirectional else 1
        finals = []
        for layer in range(self.num_layers):
            outs = []
            for d in range(dirs):
                k = layer * dirs + d
                cell = self.cells[k]
                s0 = self._initial(state, k, x.shape[0], cell)
                out, s = self._run(cell, x, s0, reverse=d == 1)
                outs.append(out)
                finals.append(s)
            x = outs[0] if dirs == 1 else concat(outs, axis=2)
            if self.dropout and layer < self.num_layers - 1:
                x = F.dropout(x, self.dropout, self.training)
        return x, self._pack(finals)

    def _initial(self, state, k, n, cell):
        return cell.initial_state(n) if state is None else state[k]

    def _pack(self, finals):
        return stack(finals, axis=0)

    def extra_repr(self):
        return (f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
                f"bidirectional={self.bidirectional}")


class RNN(_Recurrent):
    """Elman RNN. nonlinearity='tanh' | 'relu'."""

    cell_class = RNNCell


class GRU(_Recurrent):
    cell_class = GRUCell


class LSTM(_Recurrent):
    """Returns output, (h_n, c_n). Pass (h_0, c_0) of shape (L·dirs, N, H) to continue a sequence."""

    cell_class = LSTMCell

    def _initial(self, state, k, n, cell):
        return cell.initial_state(n) if state is None else (state[0][k], state[1][k])

    def _pack(self, finals):
        return stack([h for h, _ in finals], axis=0), stack([c for _, c in finals], axis=0)


def detach_state(state):
    """Cut the graph between chunks (truncated BPTT): keep the values, drop the history."""
    if isinstance(state, tuple):
        return tuple(detach_state(s) for s in state)
    return Tensor(np.array(state.data))
