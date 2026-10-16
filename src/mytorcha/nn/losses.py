"""Loss functions as modules: loss_fn = CrossEntropyLoss(); loss = loss_fn(logits, y)."""

from .. import functional as F
from .module import Module


def _loss_module(name, fn, doc):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def forward(self, pred, target):
        return fn(pred, target, **self.kwargs)

    return type(name, (Module,), {"__init__": __init__, "forward": forward, "__doc__": doc})


MSELoss = _loss_module("MSELoss", F.mse_loss, "mean((pred - y)²): regression")
L1Loss = _loss_module("L1Loss", F.l1_loss, "mean(|pred - y|): regression, robust to outliers")
HuberLoss = _loss_module("HuberLoss", F.huber_loss, "quadratic near 0, linear beyond `delta`")
SmoothL1Loss = _loss_module("SmoothL1Loss", F.smooth_l1_loss, "Huber / beta (object detection)")
BCELoss = _loss_module("BCELoss", F.binary_cross_entropy, "binary cross-entropy on probabilities")
BCEWithLogitsLoss = _loss_module("BCEWithLogitsLoss", F.binary_cross_entropy_with_logits,
                                 "sigmoid + binary cross-entropy, numerically stable")
CrossEntropyLoss = _loss_module("CrossEntropyLoss", F.cross_entropy,
                                "softmax + negative log-likelihood; logits (..., C), int targets (...)")
NLLLoss = _loss_module("NLLLoss", F.nll_loss, "negative log-likelihood on log-probabilities")
KLDivLoss = _loss_module("KLDivLoss", F.kl_div, "KL(target || exp(input)); input = log-probabilities")
HingeLoss = _loss_module("HingeLoss", F.hinge_loss, "multi-class SVM margin loss")
