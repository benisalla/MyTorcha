"""Multi-head self-attention and a Transformer encoder block (batch-first)."""

import numpy as np

from .. import functional as F
from .layers import GELU, Dropout, LayerNorm, Linear, ReLU
from .module import Module, Sequential


class MultiheadAttention(Module):
    """Each head attends with its own slice of the q/k/v projections; the heads'
    outputs are concatenated and mixed by out_proj.

    forward(query, key=None, value=None, mask=None, causal=False) -> (N, T, E).
    `mask` is True where attention is NOT allowed; causal=True hides the future.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim, self.num_heads, self.head_dim = embed_dim, num_heads, embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias)
        self.dropout = dropout

    def _heads(self, x):  # (N, T, E) -> (N, heads, T, head_dim)
        N, T, _ = x.shape
        return x.reshape(N, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, query, key=None, value=None, mask=None, causal=False):
        key = query if key is None else key
        value = key if value is None else value
        q, k, v = self._heads(self.q_proj(query)), self._heads(self.k_proj(key)), self._heads(self.v_proj(value))
        Tq, Tk = q.shape[2], k.shape[2]
        if causal:
            future = np.triu(np.ones((Tq, Tk), bool), 1)
            mask = future if mask is None else (np.asarray(mask) | future)
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = F.dropout(F.softmax(scores, axis=-1), self.dropout, self.training)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(query.shape[0], Tq, self.embed_dim)
        return self.out_proj(out)

    def extra_repr(self):
        return f"{self.embed_dim}, num_heads={self.num_heads}"


class TransformerEncoderLayer(Module):
    """x + Attention(LN(x)), then x + MLP(LN(x))  (pre-norm, as in GPT-2)."""

    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.0, activation="gelu"):
        dim_feedforward = dim_feedforward or 4 * d_model
        self.ln1, self.ln2 = LayerNorm(d_model), LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        act = GELU() if activation == "gelu" else ReLU()
        self.mlp = Sequential(Linear(d_model, dim_feedforward), act, Linear(dim_feedforward, d_model),
                              Dropout(dropout))

    def forward(self, x, mask=None, causal=False):
        x = x + self.attn(self.ln1(x), mask=mask, causal=causal)
        return x + self.mlp(self.ln2(x))
