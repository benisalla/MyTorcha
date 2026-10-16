# MyTorcha

**MyTorcha** is a small deep learning framework built from scratch: a baby framework that
hoped to grow into something bigger, Inshallah... and it grew. It started as a neuron, a
layer and a way to stack them. Now it has an autograd engine, CNNs, RNNs, LSTMs, GRUs,
attention, Transformers, 20 activations, 10 losses and 8 optimizers. It still depends on
**NumPy and nothing else**.

<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288189698-874ae81f-1951-4698-af3a-ee28468703ac.gif" width="800" height="400" alt="Graph Visualization"/>
</div>

The core idea is still graph theory. Every computation builds a **directed acyclic graph
(DAG)**: each node remembers its inputs and how to send gradients back to them. Calling
`backward()` walks that graph in reverse and applies the chain rule. MyTorcha has two
engines built on that one idea:

| Engine | One node holds | Use it to |
|---|---|---|
| `mytorcha.scalar.Node` | one Python number | *see* backprop, number by number (the original MyTorcha) |
| `mytorcha.Tensor` | a whole NumPy array | train real models: MLPs, CNNs, RNNs, LSTMs, Transformers |

Is this the end? No! As long as I'm breathing, none of my projects will stop (hopefully).
The ConvNets and Transformers that were "almost done, just fixing some bugs" in v1?
**Done. Bugs fixed. Tested.** (See [what v1 got wrong](#what-v1-got-wrong). It was humbling.)

<div align="center">
  <h5>MyTorcha</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288193776-a2eb7531-5305-4c5b-98d5-875bfb619a08.png" width="400" height="400" alt="MyTorcha Logo"/>
</div>

---

## Table of Contents

- [Learn it: the website](#learn-it-the-website)
- [Installation](#installation)
- [Quick start](#quick-start)
- [What's inside](#whats-inside)
- [Results](#results)
- [How do we know the gradients are right?](#how-do-we-know-the-gradients-are-right)
- [What v1 got wrong](#what-v1-got-wrong)
- [Project layout](#project-layout)
- [Motivation](#motivation)
- [Future roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Learn it: the website

The [`docs/`](docs/) folder is a small website that teaches MyTorcha chapter by chapter.
It has runnable code, "predict the output" exercises and interactive widgets: a live
computation graph, an optimizer race and a convolution visualizer.

```bash
python -m http.server 8000 -d docs     # then open http://localhost:8000
```

| Chapter | You learn |
|---|---|
| 1. Autograd from scratch | the scalar engine, the chain rule, topological sort; train XOR |
| 2. Tensors | broadcasting, `backward()`, `no_grad`, checking gradients |
| 3. Modules | `Module`, `Parameter`, `Sequential`, init, save / load |
| 4. Training | losses, optimizers, schedulers, `DataLoader`, the training loop |
| 5. CNNs | convolution as im2col, pooling, BatchNorm, digits MLP vs CNN |
| 6. RNN / LSTM / GRU | gates, backprop through time, gradient clipping, char-LSTM |
| 7. Attention | scaled dot-product, multi-head, a tiny GPT |
| Reference | every public function and class |

---

## Installation

```bash
git clone https://github.com/benisalla/MyTorcha.git
cd MyTorcha
python -m venv .venv && source .venv/bin/activate
pip install -e .              # just NumPy
pip install -e '.[dev]'       # + pytest and ruff, to run the tests
pip install -e '.[parity]'    # + PyTorch, ONLY to compare our numbers against it
```

(The v1 README said `pip install -r requirements.txt`. There was no `requirements.txt`.
We don't talk about it.)

---

## Quick start

```python
import mytorcha as mt
from mytorcha import nn, optim

mt.manual_seed(0)
X, y = mt.datasets.make_spirals(n_per_class=100, n_classes=3)   # not linearly separable
loader = mt.DataLoader(mt.TensorDataset(X, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(200):
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()       # the whole DAG, in reverse
        opt.step()
```

Prefer not to write the loop? `mt.Trainer(model, loss_fn, opt).fit(loader, epochs=200)`.
The PyTorch-shaped API is on purpose: what you learn here transfers directly.

And the original scalar engine, where every single number is a node:

```python
from mytorcha.scalar import Node

x = Node(2.0)
y = x * x + x.log()
y.backward()
print(x.grad)        # 2x + 1/x = 4.5
```

---

## What's inside

| Part | What you get |
|---|---|
| **Autograd** | `Tensor` with broadcasting, `@`, `sum/mean/max/var`, `reshape/transpose/getitem/pad`, `concat/stack/where`, `no_grad()` |
| **Activations** (20) | ReLU, ReLU6, LeakyReLU, PReLU *(learned)*, ELU, CELU, SELU, GELU (exact and tanh), Sigmoid, LogSigmoid, Hardsigmoid, Tanh, Hardtanh, Softplus, Softsign, SiLU / Swish, Hardswish, Mish, Arctan, and Softmax / LogSoftmax / Softmin |
| **Layers** | Linear, Conv1d, Conv2d, ConvTranspose2d, MaxPool1d/2d, AvgPool1d/2d, AdaptiveAvgPool2d, Flatten, Dropout, BatchNorm1d/2d, LayerNorm, GroupNorm, Embedding, Identity |
| **Recurrent** | RNN, LSTM, GRU (multi-layer, bidirectional, dropout between layers) and RNNCell / LSTMCell / GRUCell |
| **Attention** | scaled dot-product attention, MultiheadAttention (masks, causal), TransformerEncoderLayer |
| **Losses** (10) | MSE, L1, Huber, SmoothL1, BCE, BCEWithLogits, CrossEntropy (label smoothing, ignore_index), NLL, KLDiv, Hinge |
| **Optimizers** (8) | SGD (momentum, Nesterov, weight decay), Adagrad, RMSprop (centered, momentum), Adadelta, Adam (AMSGrad), AdamW, Adamax, Lion |
| **Schedulers** | StepLR, ExponentialLR, CosineAnnealingLR, LambdaLR (warm-up) and `clip_grad_norm_` |
| **Init** | Xavier / Kaiming (uniform and normal), orthogonal, normal, uniform |
| **Data** | `Dataset`, `TensorDataset`, `DataLoader`; toy datasets (spirals, moons, sine waves); the Kaggle digits CSV loader |
| **Extras** | `Trainer`, `state_dict` / `save` / `load` (`.npz`), `gradcheck`, graphviz `draw()` for the scalar graph |

Differences from PyTorch, so nothing surprises you:
- Cross-entropy takes classes on the **last** axis (`(N, C)` or `(N, T, C)`). That means
  sequence models don't need a transpose.
- RNNs are **batch-first**: `(N, T, features)`.
- `MultiheadAttention` returns just the output, not `(output, weights)`.

---

## Results

All numbers are real runs of the scripts in [`examples/`](examples/) on a CPU (seed 0).

| Example | Result |
|---|---|
| `01_xor_scalar.py`: scalar MLP on XOR | loss 1.771 → 0.0000, all 4 predictions correct |
| `02_spirals_mlp.py`: 3-class spirals | linear baseline **53.7%** accuracy, MLP **99.3%** |
| `03_digits_mlp_vs_cnn.py`: Kaggle digits, 10k train / 2k val, 3 epochs | MLP (101,770 params) **93.65%** val, CNN (9,146 params) **97.10%** val |
| `04_sine_rnn_lstm_gru.py`: next-value prediction | MSE: copy-last-value baseline 0.0361, RNN 0.0084, LSTM 0.0050, GRU 0.0050 |
| `05_char_lstm.py`: character LSTM | gibberish at step 0 → *"every weight gets its own gradient, every optimizer keeps its own state."* by step 300 |
| `06_tiny_transformer.py`: 2-layer GPT, 59,825 params | loss 3.07 → 0.059 in 200 steps; greedy sampling reproduces the training sentence |

Why two engines? For one 784 → 10 layer, the scalar engine builds **31,427 nodes per image**.
The Tensor engine does the same batch as one matrix multiply, roughly **3,000× faster**.
Same chain rule, different granularity.

On a busy machine, NumPy's multi-threaded BLAS can be slower than single-threaded for
small matrices. If training feels slow, try `OPENBLAS_NUM_THREADS=1 python examples/...`.

---

## How do we know the gradients are right?

We don't trust them. We test them. **160 tests** run in about 20 seconds:

- **Finite differences** (`mytorcha.gradcheck`): every op, activation, loss, layer, RNN, LSTM
  and GRU is checked against `(f(x + h) - f(x - h)) / 2h` in float64.
- **PyTorch parity**: activations, losses, conv / transposed conv / pooling, BatchNorm
  (including running stats), 2-layer bidirectional RNN / LSTM / GRU, causal multi-head
  attention, and 10 optimizer configurations match `torch` to about 1e-10 (skipped if
  torch isn't installed).

```bash
pytest
```

---

## What v1 got wrong

The original code is kept in [`legacy/`](legacy/) for history. When we tested it, it had real
bugs. Each one is now a regression test:

| v1 bug | Effect | Fix |
|---|---|---|
| `log` backward used `1/log(x)` instead of `1/x` | d/dx log(x) at 2 gave **1.4427** instead of 0.5 | correct local derivative |
| `elu` backward used `x·eˣ` | gradient at -1 was **-0.37** (wrong sign!) instead of +0.37 | `alpha·eˣ` |
| Nodes were equal when their **values** were equal | two weights holding 3.0 collapsed into one set entry, so `backward()` could skip a subgraph | identity hashing |
| Adam / RMSprop / Adagrad / momentum shared **one** scalar state for all weights | every weight's step mixed in everyone else's gradients | per-parameter state; matches `torch.optim` |
| `gradient_descent` took a `model` argument but was called without one | `TypeError` on the first step | fixed (and the "we are in sgd" print is gone) |
| ReLU was applied to the output logits | negative logits clipped to 0 before softmax | linear output layer |
| `main.py` swapped train and validation | trained on 10% of the data | train on the big split |
| raw `exp` in tanh / sigmoid / softmax | overflow on large inputs | stable formulas |
| recursive topological sort | recursion limit on deep graphs | iterative sort |
| weights drawn from `uniform(-1, 1)` | 784-input neurons saturated | Kaiming / Xavier / PyTorch-default init |

Moral: **The Deal is (really) in the Details.** And in the tests.

---

## Project layout

```
src/mytorcha/
├── scalar.py        the original engine: Node, Neuron, Layer, MLP
├── tensor.py        Tensor autograd (the heart)
├── functional.py    activations, losses, conv, pooling, attention, dropout
├── nn/              Module, layers, recurrent, attention, loss modules
├── optim/           optimizers and LR schedulers
├── init.py  data.py  datasets.py  trainer.py  gradcheck.py  random.py
examples/            01_xor_scalar … 06_tiny_transformer
tests/               gradchecks and PyTorch parity
docs/                the teaching website
legacy/              MyTorcha v1, untouched
data/                Kaggle "Digit Recognizer" CSVs
```

---

## Motivation

I believe complexity is one of the main challenges of modern software frameworks. Using
PyTorch and TensorFlow is efficient, but it often left me feeling disconnected from what
happens underneath. So I set out to build something where nothing is hidden. Every
gradient here is code you can read, and a test that proves it.

This project is driven by my personal belief:

> **"The Deal is in the Details"**

---

## Future roadmap

Done since v1: ~~CNNs~~, ~~Transformers~~, ~~Adam and SGD with momentum~~ (done properly
this time), ~~graph visualization~~. Next, Inshallah:

- GPU backend (CuPy has the NumPy API, so it's tempting)
- A C++ core (the original dream, still alive)
- More layers: Conv3d, depthwise / grouped convolutions, a Transformer decoder with cross-attention
- Mixed precision and gradient checkpointing

---

## Contributing

Contributions are welcome! Fork the repo, create a branch, and open a pull request.
One rule: **every new op comes with a gradcheck test**. If the finite differences don't
agree, neither do we.

---

## Contact

For any questions or suggestions, feel free to reach out:

- **Email**: [ismailbenalla52@gmail.com](mailto:ismailbenalla52@gmail.com)
- **GitHub**: [@benisalla](https://github.com/benisalla)
- **LinkedIn**: [Ismail Ben Alla](https://linkedin.com/in/ismail-ben-alla-bai)
- **Twitter**: [@ismail_ben_alla](https://twitter.com/ismail_ben_alla)
- **Instagram**: [@ismail_ben_alla](https://instagram.com/ismail_ben_alla)

Let's connect and explore the world of AI together! 🤖🌟

<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288196444-aae5a2b5-6623-4f1e-9149-be45342093c3.gif" width="500" height="300" alt="Neural Net Humor"/>
</div>
