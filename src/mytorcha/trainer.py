"""A small, readable training loop. You can always write the loop yourself:

    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""

import time

import numpy as np

from .optim import clip_grad_norm_
from .tensor import no_grad


class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler=None, clip_grad=None, metric="accuracy"):
        self.model, self.loss_fn, self.optimizer = model, loss_fn, optimizer
        self.scheduler, self.clip_grad, self.metric = scheduler, clip_grad, metric
        self.history = []

    def train_epoch(self, loader):
        self.model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            loss = self.loss_fn(self.model(xb), yb)
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                clip_grad_norm_(self.optimizer.params, self.clip_grad)
            self.optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
        return total / n

    def evaluate(self, loader):
        """Average loss (and accuracy for classification) without building a graph."""
        self.model.eval()
        total, correct, n = 0.0, 0, 0
        with no_grad():
            for xb, yb in loader:
                out = self.model(xb)
                total += self.loss_fn(out, yb).item() * len(xb)
                if self.metric == "accuracy":
                    correct += int((out.data.argmax(-1) == yb.data).sum())
                n += len(xb)
        result = {"loss": total / n}
        if self.metric == "accuracy":
            result["acc"] = correct / n
        return result

    def fit(self, train_loader, epochs=10, val_loader=None, verbose=True):
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            row = {"epoch": epoch, "train_loss": self.train_epoch(train_loader)}
            if val_loader is not None:
                row.update({f"val_{k}": v for k, v in self.evaluate(val_loader).items()})
            if self.scheduler is not None:
                self.scheduler.step()
            row["time"] = time.time() - t0
            self.history.append(row)
            if verbose:
                print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        return self.history

    def predict(self, x, batch_size=256):
        self.model.eval()
        outs = []
        with no_grad():
            for i in range(0, len(x), batch_size):
                outs.append(self.model(x[i : i + batch_size]).data)
        return np.concatenate(outs)
