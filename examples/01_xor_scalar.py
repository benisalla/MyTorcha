"""Chapter 1: backprop with single numbers (the scalar engine). Run: python examples/01_xor_scalar.py"""

import random

from mytorcha.scalar import MLP, sum_nodes

random.seed(0)
X, Y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
model = MLP(2, [8], 1, act="tanh")
print("parameters:", len(model.parameters()))

for step in range(301):
    loss = sum_nodes([(model(x)[0] - y) ** 2 for x, y in zip(X, Y)])
    model.zero_grad()
    loss.backward()  # chain rule over every single number in the graph
    for p in model.parameters():
        p.data -= 0.05 * p.grad
    if step % 50 == 0:
        print(f"step {step:3d} | loss {loss.data:.4f}")

print("predictions:", [round(model(x)[0].data, 3) for x in X], "targets:", Y)
print("graph size for one forward pass:", len(model(X[0])[0].topo()), "nodes")
