"""Learning-rate schedules. Call scheduler.step() once per epoch (or per step)."""

import math


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.epoch = 0

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.get_lr()
        return self.optimizer.lr


class StepLR(LRScheduler):
    """Multiply lr by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size, self.gamma = step_size, gamma

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.epoch // self.step_size)


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma**self.epoch


class CosineAnnealingLR(LRScheduler):
    """Smoothly decay from base_lr to eta_min over T_max epochs along a half cosine."""

    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max, self.eta_min = T_max, eta_min

    def get_lr(self):
        t = min(self.epoch, self.T_max)
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2


class LambdaLR(LRScheduler):
    """lr = base_lr · fn(epoch). E.g. linear warm-up: LambdaLR(opt, lambda e: min(1, (e + 1) / 10))."""

    def __init__(self, optimizer, fn):
        super().__init__(optimizer)
        self.fn = fn
        optimizer.lr = self.base_lr * fn(0)

    def get_lr(self):
        return self.base_lr * self.fn(self.epoch)
