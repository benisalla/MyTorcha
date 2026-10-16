"""Module, Parameter and containers."""

import numpy as np

from ..tensor import DEFAULT_DTYPE, Tensor


class Parameter(Tensor):
    """A Tensor that a Module should train (requires_grad=True by default)."""

    def __init__(self, data, requires_grad=True):
        super().__init__(np.asarray(data, dtype=DEFAULT_DTYPE), requires_grad=requires_grad)

    @classmethod
    def empty(cls, *shape):
        return cls(np.zeros(shape, DEFAULT_DTYPE))


class Module:
    """Base class: subclass it, create layers/Parameters in __init__, write forward().

    Parameters are found by walking attributes (Parameters, Modules, and lists or
    tuples of them), so there is nothing to register by hand. Non-trainable state
    such as BatchNorm running statistics goes in names listed in `_buffers`.
    """

    training = True
    _buffers = ()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # ---- traversal ------------------------------------------------------------
    def _items(self):
        for name, value in vars(self).items():
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, (Module, Parameter)):
                        yield f"{name}.{i}", v
            elif isinstance(value, (Module, Parameter)):
                yield name, value

    def named_parameters(self, prefix=""):
        seen = set()
        for name, v in self._items():
            items = [(prefix + name, v)] if isinstance(v, Parameter) else v.named_parameters(f"{prefix}{name}.")
            for n, p in items:
                if id(p) not in seen:  # shared (tied) weights are listed once
                    seen.add(id(p))
                    yield n, p

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def named_modules(self, prefix=""):
        yield prefix, self
        for name, v in self._items():
            if isinstance(v, Module):
                yield from v.named_modules(f"{prefix}{name}." if prefix else f"{name}.")

    def modules(self):
        return [m for _, m in self.named_modules()]

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    # ---- modes and gradients ---------------------------------------------------
    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    # ---- saving ------------------------------------------------------------------
    def state_dict(self):
        state = {name: p.data.copy() for name, p in self.named_parameters()}
        for prefix, m in self.named_modules():
            for b in m._buffers:
                state[prefix + b] = np.array(getattr(m, b), copy=True)
        return state

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        for name, value in state.items():
            if name in params:
                assert params[name].shape == value.shape, f"shape mismatch for {name}"
                params[name].data[...] = value
            else:
                module_name, _, buf = name.rpartition(".")
                module = dict(self.named_modules())[module_name + "." if module_name else ""]
                setattr(module, buf, np.array(value, copy=True))
        return self

    def save(self, path):
        np.savez(path, **self.state_dict())

    def load(self, path):
        with np.load(path) as f:
            return self.load_state_dict({k: f[k] for k in f.files})

    # ---- printing ----------------------------------------------------------------
    def extra_repr(self):
        return ""

    def __repr__(self):
        children = [(n, v) for n, v in self._items() if isinstance(v, Module)]
        if not children:
            return f"{type(self).__name__}({self.extra_repr()})"
        lines = [f"  ({n}): " + repr(m).replace("\n", "\n  ") for n, m in children]
        return f"{type(self).__name__}(\n" + "\n".join(lines) + "\n)"


class Sequential(Module):
    """Run modules one after another: Sequential(Linear(2, 8), ReLU(), Linear(8, 1))."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def append(self, layer):
        self.layers.append(layer)
        return self


class ModuleList(Module):
    def __init__(self, modules=()):
        self.items = list(modules)

    def __getitem__(self, i):
        return self.items[i]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def append(self, m):
        self.items.append(m)
        return self
