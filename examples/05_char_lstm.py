"""Chapter 7: a character-level LSTM that learns to write like its training text."""

import numpy as np

import mytorcha as mt
from mytorcha import nn, optim

TEXT = (
    "the deal is in the details. i do not learn things until i build them. "
    "a tensor remembers how it was made, and the chain rule walks that memory backwards. "
    "every weight gets its own gradient, every optimizer keeps its own state. "
) * 20

mt.manual_seed(0)
chars = sorted(set(TEXT))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in TEXT])
T, N = 48, 32


class CharLSTM(nn.Module):
    def __init__(self, vocab, emb=32, hidden=128):
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids, state=None):
        out, state = self.lstm(self.emb(ids), state)
        return self.head(out), state


def sample(model, prompt, n=120, temperature=0.6):
    model.eval()
    ids, state, out = [stoi[c] for c in prompt], None, prompt
    with mt.no_grad():
        logits, state = model(np.array([ids]), state)
        for _ in range(n):
            p = mt.functional.softmax(logits[:, -1] / temperature).data[0].astype(np.float64)
            nxt = int(mt.random.get_rng().choice(len(chars), p=p / p.sum()))
            out += chars[nxt]
            logits, state = model(np.array([[nxt]]), state)
    model.train()
    return out


model = CharLSTM(len(chars))
opt = optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()  # classes on the last axis: (N, T, vocab) works directly
for step in range(301):
    starts = mt.random.get_rng().integers(0, len(data) - T - 1, N)
    x = np.stack([data[s : s + T] for s in starts])
    y = np.stack([data[s + 1 : s + T + 1] for s in starts])
    logits, _ = model(x)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    optim.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
        print(f"step {step:3d} | loss {loss.item():.3f} | {sample(model, 'the ', 60)!r}")
print("\nfinal sample:\n" + sample(model, "every "))
