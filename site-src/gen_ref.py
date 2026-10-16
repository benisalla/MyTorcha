import html
import inspect
import pathlib

import mytorcha as mt
from mytorcha import datasets, gradcheck, init, nn, optim, scalar
from mytorcha import functional as F

SP = pathlib.Path(__file__).parent
D = {  # one-liners for things without a docstring (kept factual)
    "tensor": "Create a Tensor (same as the Tensor constructor).",
    "zeros": "float32 tensor of zeros.",
    "ones": "float32 tensor of ones.",
    "randn": "float32 tensor of standard-normal samples (global generator unless rng is given).",
    "arange": "Like np.arange, as a Tensor.",
    "concat": "Join tensors along an existing axis (differentiable).",
    "cat": "Alias of concat.",
    "stack": "Join tensors along a new axis (differentiable).",
    "where": "Pick from a where cond is True, else from b (differentiable in a and b).",
    "is_grad_enabled": "False inside a no_grad() block.",
    "manual_seed": "Seed the global generator used by init, randn, dropout and DataLoader shuffling.",
    "Tensor": "n-dimensional array with autograd. Python floats become float32; NumPy arrays keep their dtype.",
    "backward": "Backpropagate from this tensor; grad defaults to 1 for a scalar.",
    "zero_grad": "Set .grad to None.",
    "detach": "A new Tensor with the same data and no history.",
    "numpy": "The underlying ndarray.",
    "item": "The value of a one-element tensor as a Python number.",
    "astype": "Cast to another dtype (differentiable).",
    "argmax": "Index of the max (no gradient).",
    "__matmul__": "a @ b: matrix product, batched and broadcast like NumPy.",
    "exp": "Elementwise eˣ.",
    "log": "Elementwise natural log.",
    "sqrt": "Elementwise square root.",
    "abs": "Elementwise |x|.",
    "clip": "Clamp to [lo, hi]; zero gradient outside.",
    "sum": "Sum over axis (all axes by default).",
    "mean": "Mean over axis.",
    "max": "Max over axis; gradient goes to the max (ties share it).",
    "min": "Min over axis.",
    "reshape": "New shape, same data.",
    "view": "Alias of reshape.",
    "flatten": "Flatten all axes from start on.",
    "transpose": "Permute axes (reverse them by default).",
    "permute": "Alias of transpose.",
    "swapaxes": "Swap two axes.",
    "unsqueeze": "Insert an axis of size 1.",
    "squeeze": "Remove axes of size 1.",
    "__getitem__": "t[idx]: slicing and integer-array indexing; gradient scattered with np.add.at.",
    "train_epoch": "One pass over a loader; returns the mean training loss.",
    "fit": "Train for epochs, optionally evaluating each epoch; returns history.",
    "predict": "Model outputs as a NumPy array, in batches, under no_grad.",
    "Trainer": "A small training loop: fit, evaluate, predict.",
    "identity": "f(x) = x.",
    "linear_act": "Alias of identity.",
    "relu": "max(0, x).",
    "relu6": "min(max(0, x), 6).",
    "leaky_relu": "x if x > 0 else negative_slope·x.",
    "elu": "x if x > 0 else alpha·(eˣ − 1).",
    "celu": "x if x > 0 else alpha·(e^(x/alpha) − 1).",
    "selu": "Scaled ELU with self-normalising constants.",
    "sigmoid": "1 / (1 + e⁻ˣ), computed without overflow.",
    "logsigmoid": "log(sigmoid(x)), stable.",
    "hard_sigmoid": "clip(x/6 + 1/2, 0, 1).",
    "tanh": "Hyperbolic tangent.",
    "hardtanh": "clip(x, lo, hi).",
    "softplus": "log(1 + e^(βx)) / β, stable.",
    "softsign": "x / (1 + |x|).",
    "hardswish": "x·clip(x + 3, 0, 6) / 6.",
    "arctan": "arctan(x).",
    "softmax": "Probabilities along axis (max-subtracted, stable).",
    "log_softmax": "log of softmax along axis, stable.",
    "softmin": "softmax(−x).",
    "get_activation": "Look up an activation function by name (e.g. 'gelu').",
    "mse_loss": "Mean squared error.",
    "l1_loss": "Mean absolute error.",
    "smooth_l1_loss": "Huber loss divided by beta.",
    "cosine_similarity": "Cosine of the angle between a and b along axis.",
    "max_pool2d": "2-D max pooling (NCHW); stride defaults to kernel_size.",
    "avg_pool2d": "2-D average pooling (zero padding counted).",
    "max_pool1d": "1-D max pooling on (N, C, L).",
    "avg_pool1d": "1-D average pooling on (N, C, L).",
    "ExponentialLR": "Multiply lr by gamma every epoch.",
    "uniform_": "Fill with U(a, b).",
    "normal_": "Fill with N(mean, std²).",
    "zeros_": "Fill with 0.",
    "ones_": "Fill with 1.",
    "xavier_normal_": "Glorot normal: std = gain·√(2/(fan_in + fan_out)).",
    "kaiming_normal_": "He normal: std = √(2/((1 + a²)·fan_in)).",
    "orthogonal_": "(Semi-)orthogonal matrix, scaled by gain.",
    "make_moons": "Two interleaving half-circles (binary classification).",
    "Layer": "A list of scalar Neurons; always returns a list.",
    "Dropout": "Zero activations with probability p while training (inverted dropout).",
    "Identity": "Returns its input.",
    "MaxPool1d": "Module form of max_pool1d.",
    "MaxPool2d": "Module form of max_pool2d.",
    "AvgPool1d": "Module form of avg_pool1d.",
    "AvgPool2d": "Module form of avg_pool2d.",
    "GRU": "Multi-layer (bi)directional GRU; returns output, h_n.",
    "ModuleList": "A list of modules (a plain Python list works too).",
    "Module": "Base class: subclass it, create layers/Parameters in __init__, write forward().",
    "Module.parameters": "All trainable Parameters (tied ones once).",
    "Module.named_parameters": "(dotted name, Parameter) pairs.",
    "Module.modules": "This module and every sub-module.",
    "Module.named_modules": "(prefix, module) pairs.",
    "Module.num_parameters": "Total number of trainable scalars.",
    "Module.train": "Set training mode on every sub-module.",
    "Module.eval": "Set evaluation mode (Dropout off, BatchNorm uses running stats).",
    "Module.zero_grad": "Set every parameter's .grad to None.",
    "Module.state_dict": "Copies of parameters and buffers, by name.",
    "Module.load_state_dict": "Load values from a state_dict (shapes must match).",
    "Module.save": "Save the state_dict to a .npz file.",
    "Module.load": "Load a .npz file saved by save(); returns the module.",
    "Module.extra_repr": "Override to add details to print(module).",
    "Node.exp": "eˣ.",
    "Node.log": "Natural log.",
    "Node.abs": "|x|.",
    "Node.relu": "max(0, x).",
    "Node.leaky_relu": "Leaky ReLU.",
    "Node.elu": "ELU.",
    "Node.sigmoid": "Stable sigmoid.",
    "Node.tanh": "tanh.",
    "Node.softplus": "Stable softplus.",
    "Node.arctan": "arctan.",
    "Node.gelu": "GELU (tanh approximation).",
    "Node.linear": "Identity (for a linear output layer).",
}


def first(o, key=None):
    if key and key in D:
        return D[key]
    d = inspect.getdoc(o) or ""
    if d.startswith("Base class: subclass it") and key not in ("Module",):
        d = ""
    line = d.split("\n\n")[0].replace("\n", " ").strip()
    return line or D.get(key or getattr(o, "__name__", ""), "")


def sig(o):
    try:
        return str(inspect.signature(o))
    except (TypeError, ValueError):
        return "(…)"


rows_out = []


def section(title, anchor, intro, items):
    rows_out.append(f'<h2 id="{anchor}">{title}</h2>')
    if intro:
        rows_out.append(f"<p>{intro}</p>")
    rows_out.append(
        '<div class="table-wrap"><table class="ref"><thead><tr><th>Name</th><th>Description</th></tr></thead><tbody>'
    )
    for name, obj, key in items:
        desc = first(obj, key)
        assert desc, f"no description for {name}"
        s = sig(obj) if not isinstance(obj, property) else ""
        rows_out.append(
            f'<tr><td><code>{html.escape(name)}</code><div class="sig">{html.escape(s)}</div></td><td>{html.escape(desc)}</td></tr>'
        )
    rows_out.append("</tbody></table></div>")


def mod_items(mod, names, prefix=""):
    return [(prefix + n, getattr(mod, n), n) for n in names]


section(
    "mytorcha",
    "top",
    "Import with <code>import mytorcha as mt</code>.",
    mod_items(
        mt,
        [
            "tensor",
            "zeros",
            "ones",
            "randn",
            "arange",
            "concat",
            "cat",
            "stack",
            "where",
            "no_grad",
            "is_grad_enabled",
            "manual_seed",
        ],
    ),
)
T = mt.Tensor
tm = [("Tensor", T, "Tensor")]
for n, o in vars(T).items():
    if isinstance(o, property) or (n.startswith("_") and n not in ("__getitem__", "__matmul__")):
        continue
    if callable(o):
        tm.append((f"Tensor.{n}", o, n))
section(
    "Tensor",
    "tensor",
    "Properties: <code>shape</code>, <code>ndim</code>, <code>size</code>, <code>dtype</code>, <code>is_float</code>, <code>T</code>, <code>data</code> (the ndarray), <code>grad</code>, <code>requires_grad</code>. Operators <code>+ - * / ** @</code>, unary <code>-</code> and comparisons (non-differentiable, boolean) are supported, also with NumPy arrays or numbers on either side.",
    tm,
)
section(
    "Data and training",
    "data",
    "",
    [
        ("Dataset", mt.Dataset, None),
        ("TensorDataset", mt.TensorDataset, None),
        ("DataLoader", mt.DataLoader, None),
        ("Trainer", mt.Trainer, "Trainer"),
    ]
    + [
        (f"Trainer.{n}", getattr(mt.Trainer, n), n)
        for n in ["train_epoch", "evaluate", "fit", "predict"]
    ],
)
fn = [
    n
    for n, o in vars(F).items()
    if callable(o)
    and not n.startswith("_")
    and getattr(o, "__module__", "") == "mytorcha.functional"
]
section(
    "mytorcha.functional",
    "functional",
    'Stateless functions: <code>import mytorcha.functional as F</code>. Losses take <code>reduction="mean" | "sum" | "none"</code>; classification losses expect classes on the <strong>last</strong> axis.',
    mod_items(F, fn, "F."),
)
nn_names = sorted(
    [n for n in dir(nn) if not n.startswith("_") and n[0].isupper()], key=str.lower
) + ["detach_state"]
section(
    "mytorcha.nn",
    "nn",
    "Activation and loss modules accept keyword arguments that are passed to their function, e.g. <code>nn.LeakyReLU(negative_slope=0.2)</code>, <code>nn.CrossEntropyLoss(label_smoothing=0.1)</code>. Recurrent layers are batch-first.",
    mod_items(nn, nn_names, "nn."),
)
section(
    "nn.Module methods",
    "module",
    "",
    [
        (f"Module.{n}", getattr(nn.Module, n), f"Module.{n}")
        for n in [
            "parameters",
            "named_parameters",
            "modules",
            "named_modules",
            "num_parameters",
            "train",
            "eval",
            "zero_grad",
            "state_dict",
            "load_state_dict",
            "save",
            "load",
            "extra_repr",
        ]
    ],
)
section(
    "mytorcha.optim",
    "optim",
    "Every optimizer has <code>step()</code>, <code>zero_grad()</code>, <code>lr</code> (read/write), <code>params</code>, per-parameter <code>state</code>, and <code>state_dict()</code> / <code>load_state_dict()</code>. Schedulers have <code>step()</code>, which returns the new lr.",
    mod_items(
        optim,
        [
            "SGD",
            "Adagrad",
            "RMSprop",
            "Adadelta",
            "Adam",
            "AdamW",
            "Adamax",
            "Lion",
            "clip_grad_norm_",
            "StepLR",
            "ExponentialLR",
            "CosineAnnealingLR",
            "LambdaLR",
        ],
        "optim.",
    ),
)
ini = [
    n
    for n, o in vars(init).items()
    if callable(o) and not n.startswith("_") and getattr(o, "__module__", "") == "mytorcha.init"
]
section(
    "mytorcha.init",
    "init",
    "In-place initialisers: they fill <code>tensor.data</code> and return the tensor.",
    mod_items(init, ini, "init."),
)
section(
    "mytorcha.datasets",
    "datasets",
    "Small built-in datasets; nothing is downloaded.",
    mod_items(
        datasets, ["make_spirals", "make_moons", "sine_sequences", "load_digits_csv"], "datasets."
    ),
)
section(
    "mytorcha.gradcheck",
    "gradcheck",
    "",
    mod_items(gradcheck, ["gradcheck", "numerical_grad", "rand"], "gradcheck."),
)
sc = mod_items(scalar, ["Node", "sum_nodes", "draw", "Neuron", "Layer", "MLP"], "scalar.")
sc += [
    (f"Node.{n}", o, f"Node.{n}")
    for n, o in vars(scalar.Node).items()
    if not n.startswith("_") and callable(o)
]
section(
    "mytorcha.scalar",
    "scalar",
    "The one-number-per-node engine of chapter 1. <code>Node</code> attributes: <code>data</code>, <code>grad</code>, <code>label</code>; supports <code>+ - * / **</code> with numbers on either side.",
    sc,
)

head = (
    """<p class="kicker">Reference</p>
<h1>API reference</h1>
<p class="lead">Every public class and function, with its exact signature (taken from the code with <code>inspect.signature</code>).</p>
<p>Jump to: """
    + " · ".join(
        f'<a href="#{a}">{t}</a>'
        for a, t in [
            ("top", "mytorcha"),
            ("tensor", "Tensor"),
            ("data", "data &amp; Trainer"),
            ("functional", "functional"),
            ("nn", "nn"),
            ("module", "Module"),
            ("optim", "optim"),
            ("init", "init"),
            ("datasets", "datasets"),
            ("gradcheck", "gradcheck"),
            ("scalar", "scalar"),
        ]
    )
    + "</p>"
)
(SP / "pages" / "reference.html").write_text(head + "\n" + "\n".join(rows_out) + "\n")
print("rows:", sum(r.startswith("<tr><td>") for r in rows_out))
