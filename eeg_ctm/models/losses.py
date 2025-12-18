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

    details: dict[str, float] = {"loss_mean_ce": float(mean_ce.detach().cpu())}
    # design.md + theory notes: tick-gap statistics for stability diagnostics.
    # Δ_CE(x) = CE_(2) - CE_(1)  (2nd best - best) over ticks
    if ce_bt.shape[1] >= 2:
        ce_sorted = torch.sort(ce_bt, dim=-1).values
        gap_ce = (ce_sorted[:, 1] - ce_sorted[:, 0]).mean()
        details["tick_gap_ce"] = float(gap_ce.detach().cpu())

    if cfg.type == "mean_ce":
        # Certainty gap is defined only when certainty is available and valid.
        if certainty.ndim == 2 and certainty.shape == ce_bt.shape and certainty.shape[1] >= 2:
            cert_sorted = torch.sort(certainty, dim=-1, descending=True).values
            gap_cert = (cert_sorted[:, 0] - cert_sorted[:, 1]).mean()
            details["tick_gap_certainty"] = float(gap_cert.detach().cpu())
        return mean_ce, details

    if certainty.ndim != 2 or certainty.shape != ce_bt.shape:
        raise ValueError("certainty must be [B,T] and match logits")

    if certainty.shape[1] >= 2:
        cert_sorted = torch.sort(certainty, dim=-1, descending=True).values
        gap_cert = (cert_sorted[:, 0] - cert_sorted[:, 1]).mean()
        details["tick_gap_certainty"] = float(gap_cert.detach().cpu())

    # CTM t1/t2: min-loss tick + max-certainty tick.
    t_min = ce_bt.argmin(dim=-1)  # [B]
    t_cert = certainty.argmax(dim=-1)  # [B]
    b = torch.arange(ce_bt.shape[0], device=ce_bt.device)
    loss_t1 = ce_bt[b, t_min]
    loss_t2 = ce_bt[b, t_cert]
    loss_ctm = 0.5 * (loss_t1 + loss_t2).mean()

    if cfg.type == "ctm_t1t2":
        details["loss_ctm_t1t2"] = float(loss_ctm.detach().cpu())
        return loss_ctm, details

    if cfg.type == "hybrid":
        lam = float(cfg.hybrid_lambda)
        loss = lam * mean_ce + (1.0 - lam) * loss_ctm
        details["loss_ctm_t1t2"] = float(loss_ctm.detach().cpu())
        details["loss_hybrid"] = float(loss.detach().cpu())
        return loss, details

    raise ValueError(cfg.type)
