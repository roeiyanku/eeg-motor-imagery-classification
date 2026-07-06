"""Shared evaluation-result record used across decoders, the CNN, and benchmarks.

Every training/evaluation path (classical models, the neural decoders, and the
competition benchmark) produces the same core record: which model, which
subject, its accuracy, and a 4x4 confusion matrix. The optional fields carry
the extras a particular path also reports -- ``macro_f1`` for the within-subject
split, ``kappa`` for the competition protocol, and ``history`` for neural
training curves -- so a single type can flow into ``reporting`` unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalResult:
    model: str
    subject: str
    accuracy: float
    confusion: np.ndarray
    macro_f1: float | None = None
    kappa: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)
