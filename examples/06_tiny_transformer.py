"""Chapter 8: the same character task with a tiny GPT-style Transformer."""

import numpy as np

import mytorcha as mt
from mytorcha import nn, optim

TEXT = ("the deal is in the details. i do not learn things until i build them. ") * 40
mt.manual_seed(0)
chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in TEXT])
T, N, D = 32, 32, 48


class TinyGPT(nn.Module):
    def __init__(self, vocab, d=D, heads=4, layers=2):
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.blocks = [nn.TransformerEncoderLayer(d, heads) for _ in range(layers)]
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.tok(ids) + self.pos(np.arange(ids.shape[1]))
        for block in self.blocks:
            x = block(x, causal=True)  # each position may only look at the past
        return self.head(self.ln(x))


model = TinyGPT(len(chars))
print(f"{model.num_parameters():,} parameters")
opt = optim.AdamW(model.parameters(), lr=3e-3)
for step in range(201):
    starts = mt.random.get_rng().integers(0, len(data) - T - 1, N)
    x = np.stack([data[s : s + T] for s in starts])
    y = np.stack([data[s + 1 : s + T + 1] for s in starts])
    loss = mt.functional.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d} | loss {loss.item():.3f}")

ids = [stoi[c] for c in "the "]
with mt.no_grad():
    for _ in range(80):
        logits = model(np.array([ids[-T:]]))
        ids.append(int(logits.data[0, -1].argmax()))
print("greedy sample:", "".join(chars[i] for i in ids))
