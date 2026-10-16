"""Authoring helper: wrap page fragments in the shared chrome. Output is plain static HTML."""

import html
import pathlib
import re

SP = pathlib.Path(__file__).parent
OUT = SP.parent / "docs"
(SP / "check").mkdir(exist_ok=True)  # every snippet, dumped per page so it can be run and checked
PAGES = [
    (
        "index.html",
        "",
        "Start here",
        "MyTorcha — a tiny deep learning framework built from scratch",
        "MyTorcha is a small deep learning framework written from scratch in Python and NumPy: autograd, CNNs, RNNs, LSTMs, attention, losses and optimizers, explained chapter by chapter.",
    ),
    (
        "01-autograd.html",
        "01",
        "Autograd with single numbers",
        "Chapter 1 · Autograd with single numbers",
        "Build backpropagation by hand with MyTorcha's scalar Node engine: computation graphs, the chain rule, topological sort, and an interactive graph.",
    ),
    (
        "02-tensors.html",
        "02",
        "Tensors",
        "Chapter 2 · Tensors",
        "The MyTorcha Tensor: requires_grad, backward, gradient accumulation, no_grad, broadcasting and gradient checking.",
    ),
    (
        "03-modules.html",
        "03",
        "Modules and layers",
        "Chapter 3 · Modules and layers",
        "Build models with MyTorcha's Module, Parameter, Sequential and ModuleList; save and load weights; initialise them properly.",
    ),
    (
        "04-training.html",
        "04",
        "Losses, optimizers, training",
        "Chapter 4 · Losses, optimizers and the training loop",
        "Every MyTorcha loss and optimizer, learning-rate schedules, gradient clipping, DataLoader and Trainer, with an interactive optimizer race.",
    ),
    (
        "05-cnn.html",
        "05",
        "Convolutional networks",
        "Chapter 5 · Convolutional networks",
        "Conv2d, pooling, BatchNorm and ConvTranspose2d in MyTorcha, how im2col makes convolution a matrix multiply, and an interactive convolution visualiser.",
    ),
    (
        "06-rnn.html",
        "06",
        "RNN, LSTM, GRU",
        "Chapter 6 · Recurrent networks: RNN, LSTM, GRU",
        "Recurrent layers in MyTorcha: cell equations, shapes, backpropagation through time, gradient clipping, a sine forecaster and a character-level LSTM.",
    ),
    (
        "07-attention.html",
        "07",
        "Attention and Transformers",
        "Chapter 7 · Attention and Transformers",
        "Scaled dot-product attention, causal masks, multi-head attention and a tiny GPT-style Transformer built with MyTorcha.",
    ),
    (
        "reference.html",
        "API",
        "API reference",
        "API reference",
        "Every public class and function in MyTorcha with its exact signature.",
    ),
]

FENCE = re.compile(r"^```([\w-]+)\n(.*?)^```\s*$", re.S | re.M)


def render_fences(body, page):
    blocks = []

    def sub(m):
        lang, code = m.group(1), m.group(2).rstrip("\n")
        if lang in ("python", "python-norun"):
            if lang == "python":
                blocks.append(code)
            return f'<div class="code"><pre><code class="lang-python">{html.escape(code, quote=False)}</code></pre></div>'
        if lang == "bash":
            return f'<div class="code"><pre><code class="lang-bash">{html.escape(code, quote=False)}</code></pre></div>'
        if lang == "output":
            return (
                f'<div class="code output"><span class="label">output</span><pre><code>'
                f"{html.escape(code, quote=False)}</code></pre></div>"
            )
        if lang == "math":
            return f'<div class="math">{html.escape(code, quote=False)}</div>'
        raise ValueError(lang)

    out = FENCE.sub(sub, body)
    (SP / "check" / (page.replace(".html", ".py"))).write_text("\n\n".join(blocks) + "\n")
    return out


def sidebar(current):
    items = []
    for f, num, label, *_ in PAGES:
        cur = ' aria-current="page"' if f == current else ""
        n = f'<span class="num">{num}</span>' if num else '<span class="num">·</span>'
        items.append(f'<li><a href="{f}"{cur}>{n}{label}</a></li>')
    return "\n".join(items)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/style.css">
<script src="assets/site.js"></script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <a class="brand" href="index.html"><img src="assets/favicon.svg" alt="" width="24" height="24">MyTorcha</a>
  <button type="button" class="menu-btn" aria-controls="sidebar" aria-expanded="false">Chapters</button>
</header>
<div class="layout">
<nav class="sidebar" id="sidebar" aria-label="Chapters">
  <a class="brand" href="index.html"><img src="assets/favicon.svg" alt="" width="28" height="28">MyTorcha</a>
  <p class="tagline">A baby deep learning framework, built from scratch so nothing is magic.</p>
  <ol>
{nav}
  </ol>
  <p class="side-foot"><a href="https://github.com/benisalla/MyTorcha">Source on GitHub</a></p>
</nav>
<main id="main">
<article class="content">
{body}
{pager}
<p class="foot">MyTorcha by <a href="https://github.com/benisalla">Ismail Ben Alla</a>. MIT licensed. The deal is in the details.</p>
</article>
</main>
</div>
</body>
</html>
"""


def pager(i):
    parts = []
    if i > 0:
        f, _, label, *_ = PAGES[i - 1]
        parts.append(f'<a class="prev" href="{f}" rel="prev"><span>← Previous</span>{label}</a>')
    if i < len(PAGES) - 1:
        f, _, label, *_ = PAGES[i + 1]
        parts.append(f'<a class="next" href="{f}" rel="next"><span>Next →</span>{label}</a>')
    return '<nav class="pager" aria-label="Chapter navigation">' + "".join(parts) + "</nav>"


for i, (f, _num, _label, title, desc) in enumerate(PAGES):
    src = SP / "pages" / f
    if not src.exists():
        print("skip", f)
        continue
    body = render_fences(src.read_text(), f)
    full_title = title if f == "index.html" else f"{title} · MyTorcha"
    (OUT / f).write_text(
        TEMPLATE.format(
            title=html.escape(full_title),
            desc=html.escape(desc),
            nav=sidebar(f),
            body=body,
            pager=pager(i),
        )
    )
    print("wrote", f)
