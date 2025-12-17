"""
Loss functions for EEG-CTM.

Design doc mapping:
  - design.md "训练时的 tick loss（mean_ce / ctm_t1t2 / hybrid）"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


TickLossType = Literal["mean_ce", "ctm_t1t2", "hybrid"]


@dataclass(frozen=True)
class TickLossConfig:
    type: TickLossType = "mean_ce"
    hybrid_lambda: float = 0.5  # weight for mean_ce in hybrid


def per_sample_tick_ce(logits_ticks: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    logits_ticks: [B,C,T]
    y: [B]
    Returns per-sample per-tick CE: [B,T]
    """
    if logits_ticks.ndim != 3:
        raise ValueError(f"Expected logits_ticks [B,C,T], got {tuple(logits_ticks.shape)}")
    if y.ndim != 1:
        raise ValueError("y must be [B]")
    B, C, T = logits_ticks.shape
    if y.shape[0] != B:
        raise ValueError("y length mismatch")
    logits_bt = logits_ticks.permute(0, 2, 1).reshape(B * T, C)
    y_bt = y.repeat_interleave(T)
    ce = F.cross_entropy(logits_bt, y_bt, reduction="none").reshape(B, T)
    return ce


def tick_classification_loss(
    logits_ticks: torch.Tensor,
    certainty: torch.Tensor,
    y: torch.Tensor,
    *,
    cfg: TickLossConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Returns (loss, details).
    """
    ce_bt = per_sample_tick_ce(logits_ticks, y)  # [B,T]
    mean_ce = ce_bt.mean()

    if cfg.type == "mean_ce":
        return mean_ce, {"loss_mean_ce": float(mean_ce.detach().cpu())}

    if certainty.ndim != 2 or certainty.shape != ce_bt.shape:
        raise ValueError("certainty must be [B,T] and match logits")

    # CTM t1/t2: min-loss tick + max-certainty tick.
    t_min = ce_bt.argmin(dim=-1)  # [B]
    t_cert = certainty.argmax(dim=-1)  # [B]
    b = torch.arange(ce_bt.shape[0], device=ce_bt.device)
    loss_t1 = ce_bt[b, t_min]
    loss_t2 = ce_bt[b, t_cert]
    loss_ctm = 0.5 * (loss_t1 + loss_t2).mean()

    if cfg.type == "ctm_t1t2":
        return loss_ctm, {"loss_ctm_t1t2": float(loss_ctm.detach().cpu()), "loss_mean_ce": float(mean_ce.detach().cpu())}

    if cfg.type == "hybrid":
        lam = float(cfg.hybrid_lambda)
        loss = lam * mean_ce + (1.0 - lam) * loss_ctm
        return loss, {
            "loss_hybrid": float(loss.detach().cpu()),
            "loss_mean_ce": float(mean_ce.detach().cpu()),
            "loss_ctm_t1t2": float(loss_ctm.detach().cpu()),
        }

    raise ValueError(cfg.type)

