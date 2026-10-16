"""Chapter 2-4: a linear baseline vs an MLP on the spiral dataset."""

import mytorcha as mt
from mytorcha import nn, optim

mt.manual_seed(0)
X, y = mt.datasets.make_spirals(n_per_class=100, n_classes=3)
loader = mt.DataLoader(mt.TensorDataset(X, y), batch_size=32, shuffle=True)


def train(model, epochs=200):
    trainer = mt.Trainer(model, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=1e-2))
    trainer.fit(loader, epochs=epochs, verbose=False)
    return trainer.evaluate(mt.DataLoader(mt.TensorDataset(X, y), batch_size=300))


baseline = nn.Linear(2, 3)  # baseline first: can a straight line do it?
mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
print("linear :", train(baseline))
print("MLP    :", train(mlp))
