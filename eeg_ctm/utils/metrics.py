"""
Metrics: accuracy, Cohen's kappa, macro-F1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    kappa: float
    macro_f1: float

    def to_dict(self) -> dict:
        return {"accuracy": self.accuracy, "kappa": self.kappa, "macro_f1": self.macro_f1}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
    )

