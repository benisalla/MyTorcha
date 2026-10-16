"""MyTorcha: a tiny deep learning framework built from scratch (NumPy only)."""

from . import datasets, functional, init, nn, optim
from .data import DataLoader, Dataset, TensorDataset
from .random import manual_seed
from .tensor import (
    Tensor,
    arange,
    cat,
    concat,
    is_grad_enabled,
    no_grad,
    ones,
    randn,
    stack,
    tensor,
    where,
    zeros,
)
from .trainer import Trainer

__version__ = "0.2.0"
