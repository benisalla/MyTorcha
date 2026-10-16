"""Chapter 5: MLP vs CNN on the Kaggle digits CSV in data/train.csv (42k images, 28x28).

python examples/03_digits_mlp_vs_cnn.py            # 10k train / 2k val, a few minutes on a CPU
"""

import time

import mytorcha as mt
from mytorcha import nn, optim

mt.manual_seed(0)
t0 = time.time()
X, y = mt.datasets.load_digits_csv("data/train.csv", limit=12000)
print(f"loaded {X.shape} in {time.time() - t0:.1f}s")
Xtr, ytr, Xva, yva = X[:10000], y[:10000], X[10000:], y[10000:]  # train on the BIG part (v1 swapped them)
train_loader = mt.DataLoader(mt.TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
val_loader = mt.DataLoader(mt.TensorDataset(Xva, yva), batch_size=500)

mlp = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
cnn = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),   # 8 x 14 x 14
    nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # 16 x 7 x 7
    nn.Flatten(), nn.Dropout(0.2), nn.Linear(16 * 7 * 7, 10),
)
for name, model in [("MLP", mlp), ("CNN", cnn)]:
    print(f"\n{name}: {model.num_parameters():,} parameters")
    opt = optim.Adam(model.parameters(), lr=2e-3)
    trainer = mt.Trainer(model, nn.CrossEntropyLoss(), opt, scheduler=optim.CosineAnnealingLR(opt, T_max=3))
    trainer.fit(train_loader, epochs=3, val_loader=val_loader)
cnn.save("checkpoints_digits_cnn.npz")
