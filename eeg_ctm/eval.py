"""
Evaluation utilities for EEG-CTM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from eeg_ctm.models.aggregation import CertaintyWeightedConfig, aggregate_logits
from eeg_ctm.utils.metrics import Metrics, compute_metrics


@dataclass(frozen=True)
class EvalConfig:
    readout: str = "certainty_weighted"
    certainty_weighted_alpha: float = 5.0
    certainty_weighted_detach_certainty: bool = False


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    eval_cfg: EvalConfig,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        logits_ticks, certainty, _ = model(x)
        logits = aggregate_logits(
            logits_ticks,
            certainty,
            mode=eval_cfg.readout,  # type: ignore[arg-type]
            cw=CertaintyWeightedConfig(
                alpha=float(eval_cfg.certainty_weighted_alpha),
                detach_certainty=bool(eval_cfg.certainty_weighted_detach_certainty),
            ),
        )
        pred = logits.argmax(dim=-1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    eval_cfg: EvalConfig,
) -> Metrics:
    y_true, y_pred = predict(model, loader, device=device, eval_cfg=eval_cfg)
    return compute_metrics(y_true, y_pred)
