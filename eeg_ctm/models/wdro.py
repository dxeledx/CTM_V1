"""
Wasserstein-DRO style robustification (feature-space PGD).

This implements a practical approximation of feature-space Wasserstein-DRO by
solving an inner maximization problem over small perturbations in representation
space, using projected gradient ascent (PGD) under an L2-ball constraint.

Design / theory mapping (user notes, 2025-12-18):
  - "Wasserstein-DRO (feature-space) as cross-subject main innovation"
  - Inner problem: max_{||delta||<=rho} CE(head(rep+delta), y)
  - Outer problem: mix clean/robust losses with warmup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F


WDROLevel = Literal["rep", "tick"]
WDRONorm = Literal["l2"]


@dataclass(frozen=True)
class WDROConfig:
    enabled: bool = False
    level: WDROLevel = "rep"  # "rep" (v1) or "tick" (v2 extension)
    norm: WDRONorm = "l2"

    # L2 radius for perturbations.
    rho: float = 0.5
    steps: int = 3
    step_size: float = 0.2

    # Total robust objective = (1-mix_clean)*clean + mix_clean*robust
    mix_clean: float = 0.5
    lambda_wdro: float = 1.0  # scale factor applied to the WDRO objective

    normalize_rep: bool = True  # layer-norm rep before perturbation
    warmup_epochs: int = 10  # rho and step_size are linearly warmed-up from 0 -> target


def warmup_scale(epoch: int, warmup_epochs: int) -> float:
    """
    Linear warmup over epochs: scale in [0,1].

    epoch is 1-based.
    """
    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    # epoch=1 -> 1/warmup, epoch=warmup -> 1
    return float(min(1.0, max(0.0, epoch / warmup_epochs)))


def _project_l2(delta: torch.Tensor, rho: float) -> torch.Tensor:
    rho = float(rho)
    if rho <= 0.0:
        return torch.zeros_like(delta)
    # per-sample projection
    dn = delta.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    factor = torch.clamp(rho / dn, max=1.0)
    return delta * factor


def pgd_l2_delta(
    rep_detached: torch.Tensor,
    y: torch.Tensor,
    *,
    head: torch.nn.Module,
    rho: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    """
    Compute approx argmax_{||delta||_2 <= rho} CE(head(rep + delta), y)
    using PGD with projection to the L2 ball.

    Important: rep_detached must be detached from the backbone to avoid second-order gradients.
    Returned delta is detached.
    """
    if rep_detached.ndim != 2:
        raise ValueError(f"Expected rep_detached [B,D], got shape={tuple(rep_detached.shape)}")
    if y.ndim != 1 or y.shape[0] != rep_detached.shape[0]:
        raise ValueError("y must be [B] and match rep_detached")

    rho = float(rho)
    steps = int(steps)
    step_size = float(step_size)
    if rho <= 0.0 or steps <= 0 or step_size <= 0.0:
        return torch.zeros_like(rep_detached)

    delta = torch.zeros_like(rep_detached)
    for _ in range(steps):
        delta.requires_grad_(True)
        logits = head(rep_detached + delta)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta, only_inputs=True, create_graph=False)[0]
        # L2-normalized ascent step (per sample)
        gn = grad.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        delta = (delta + step_size * grad / gn).detach()
        delta = _project_l2(delta, rho).detach()
    return delta.detach()


def wdro_rep_objective(
    rep: torch.Tensor,
    y: torch.Tensor,
    *,
    head: torch.nn.Module,
    cfg: WDROConfig,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute feature-space WDRO objective on a representation (rep).

    Returns (loss, details) where details contains robust diagnostics.
    """
    if rep.ndim != 2:
        raise ValueError(f"Expected rep [B,D], got shape={tuple(rep.shape)}")
    if y.ndim != 1 or y.shape[0] != rep.shape[0]:
        raise ValueError("y must be [B] and match rep")

    scale = warmup_scale(epoch, int(cfg.warmup_epochs))
    rho_eff = float(cfg.rho) * scale
    step_eff = float(cfg.step_size) * scale

    rep_base = rep
    if cfg.normalize_rep:
        rep_base = F.layer_norm(rep_base, rep_base.shape[-1:])

    logits_clean = head(rep_base)
    loss_clean = F.cross_entropy(logits_clean, y)

    rep_det = rep_base.detach()
    delta_star = pgd_l2_delta(rep_det, y, head=head, rho=rho_eff, steps=int(cfg.steps), step_size=step_eff)

    logits_robust = head(rep_base + delta_star)
    loss_robust = F.cross_entropy(logits_robust, y)

    w_robust = float(cfg.mix_clean)
    if not (0.0 <= w_robust <= 1.0):
        raise ValueError("wdro.mix_clean must be in [0,1]")
    loss = (1.0 - w_robust) * loss_clean + w_robust * loss_robust

    with torch.no_grad():
        delta_norm = delta_star.norm(p=2, dim=-1).mean()
        rep_norm = rep_base.detach().norm(p=2, dim=-1).mean()
        gap = (loss_robust.detach() - loss_clean.detach()).mean()

    details = {
        "wdro_loss_clean": float(loss_clean.detach().cpu()),
        "wdro_loss_robust": float(loss_robust.detach().cpu()),
        "wdro_gap": float(gap.detach().cpu()),
        "wdro_delta_norm": float(delta_norm.detach().cpu()),
        "wdro_rep_norm": float(rep_norm.detach().cpu()),
        "wdro_rho_eff": float(rho_eff),
        "wdro_step_size_eff": float(step_eff),
        "wdro_warmup_scale": float(scale),
    }
    return loss, details

