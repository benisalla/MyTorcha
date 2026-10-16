"""Chapter 6: predict the next value of a sine wave with RNN, LSTM and GRU."""

import mytorcha as mt
from mytorcha import nn, optim

mt.manual_seed(0)
X, Y = mt.datasets.sine_sequences(n=256, T=30)
loader = mt.DataLoader(mt.TensorDataset(X, Y), batch_size=32, shuffle=True)


class Forecaster(nn.Module):
    def __init__(self, cell):
        self.rnn = cell(1, 32)          # (N, T, 1) -> (N, T, 32)
        self.head = nn.Linear(32, 1)    # one prediction per time step

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out)


naive = float(((X[:, :, 0] - Y[:, :, 0]) ** 2).mean())  # baseline: "next value = current value"
print(f"baseline (copy last value) MSE: {naive:.5f}")
for cell in (nn.RNN, nn.LSTM, nn.GRU):
    model = Forecaster(cell)
    trainer = mt.Trainer(model, nn.MSELoss(), optim.Adam(model.parameters(), lr=1e-2), clip_grad=1.0,
                         metric=None)
    hist = trainer.fit(loader, epochs=15, verbose=False)
    print(f"{cell.__name__:5s} params {model.num_parameters():5d} | final MSE {hist[-1]['train_loss']:.5f} "
          f"| {sum(h['time'] for h in hist):.1f}s")
