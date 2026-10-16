"""The scalar engine: one Python number per node.

This is the original MyTorcha idea (a computation DAG of single numbers),
kept because it is the clearest way to *see* backpropagation. It is slow by
design. For real models use `mytorcha.Tensor`, which applies exactly the same
chain rule to whole NumPy arrays at once.

    >>> x = Node(2.0); y = x * x + x.log()
    >>> y.backward(); round(x.grad, 4)      # 2x + 1/x = 4.5
    4.5
"""

import math


class Node:
    """A single number that remembers how it was computed.

    `data` is the value, `grad` is d(final output)/d(this node), `_prev` are the
    nodes it was computed from and `_backward` pushes `grad` into them.

    Hashing and equality are the default *identity* ones on purpose: two
    different weights that happen to hold the same value are still two nodes.
    """

    __slots__ = ("_backward", "_op", "_prev", "data", "grad", "label")

    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Node(data={self.data:.4g}, grad={self.grad:.4g})"

    # ---- arithmetic -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents are supported"
        out = Node(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**-1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Node(other) * self**-1

    # ---- unary functions (value, local derivative) -------------------------
    def _unary(self, value, local_grad, op):
        out = Node(value, (self,), op)

        def _backward():
            self.grad += local_grad() * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(min(self.data, 700.0))  # avoid OverflowError
        return self._unary(e, lambda: e, "exp")

    def log(self):
        return self._unary(math.log(self.data), lambda: 1.0 / self.data, "log")

    def abs(self):
        return self._unary(abs(self.data), lambda: math.copysign(1.0, self.data), "abs")

    __abs__ = abs

    def relu(self):
        return self._unary(max(self.data, 0.0), lambda: float(self.data > 0), "relu")

    def leaky_relu(self, slope=0.01):
        x = self.data
        return self._unary(x if x > 0 else slope * x, lambda: 1.0 if x > 0 else slope, "leaky_relu")

    def elu(self, alpha=1.0):
        x = self.data
        e = math.exp(min(x, 0.0))
        return self._unary(x if x > 0 else alpha * (e - 1), lambda: 1.0 if x > 0 else alpha * e, "elu")

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        return self._unary(s, lambda: s * (1 - s), "sigmoid")

    def tanh(self):
        t = math.tanh(self.data)
        return self._unary(t, lambda: 1 - t * t, "tanh")

    def softplus(self):
        x = self.data
        sp = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
        return self._unary(sp, lambda: 0.5 * (1 + math.tanh(x / 2)), "softplus")

    def arctan(self):
        x = self.data
        return self._unary(math.atan(x), lambda: 1 / (1 + x * x), "arctan")

    def gelu(self):
        x = self.data
        c = math.sqrt(2 / math.pi)
        u = c * (x + 0.044715 * x**3)
        t = math.tanh(u)
        return self._unary(
            0.5 * x * (1 + t),
            lambda: 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x),
            "gelu",
        )

    def linear(self):
        return self._unary(self.data, lambda: 1.0, "linear")

    # ---- the graph ----------------------------------------------------------
    def topo(self):
        """Nodes in topological order (inputs before outputs), iteratively.

        A recursive version hits Python's recursion limit: one neuron with 784
        inputs already builds a chain of ~800 additions.
        """
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((p, False) for p in node._prev if p not in seen)
        return order

    def backward(self):
        """Fill `.grad` of every node this one depends on (d self / d node)."""
        order = self.topo()
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


def sum_nodes(nodes):
    """Sum a list of nodes with one node per addition (no hidden Node(0) start)."""
    nodes = list(nodes)
    total = nodes[0]
    for n in nodes[1:]:
        total = total + n
    return total


def draw(root, fmt="svg"):
    """Render the graph with graphviz (optional dependency: `pip install graphviz`)."""
    from graphviz import Digraph

    dot = Digraph(format=fmt, graph_attr={"rankdir": "LR"})
    for n in root.topo():
        uid = str(id(n))
        dot.node(uid, f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}", shape="record")
        if n._op:
            dot.node(uid + n._op, n._op)
            dot.edge(uid + n._op, uid)
        for p in n._prev:
            dot.edge(str(id(p)), uid + n._op)
    return dot


# ---- the original building blocks (now with a linear output layer) ----------
import random  # noqa: E402


class Neuron:
    """w·x + b followed by an activation (any Node method name, e.g. "tanh")."""

    def __init__(self, n_in, act="relu"):
        bound = 1 / math.sqrt(n_in)  # keep w·x ~ O(1); uniform(-1, 1) saturates
        self.w = [Node(random.uniform(-bound, bound), label="w") for _ in range(n_in)]
        self.b = Node(0.0, label="b")
        self.act = act

    def __call__(self, x):
        z = sum_nodes([wi * xi for wi, xi in zip(self.w, x)]) + self.b
        return getattr(z, self.act)()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, act="relu"):
        self.neurons = [Neuron(n_in, act) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]  # always a list, even for one unit

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """MLP(2, [8, 8], 1): hidden layers use `act`, the last layer is linear
    (activating the logits would clip them before softmax / the loss)."""

    def __init__(self, n_in, hidden, n_out, act="relu"):
        sizes = [n_in] + list(hidden) + [n_out]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], act if i < len(sizes) - 2 else "linear")
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
