"""
Subject-adversarial constraint via Gradient Reversal Layer (GRL).

Design doc mapping:
  - design.md "Subject-adversarial（域对抗 + GRL）"
  - design.md "λ_adv warm-up"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # noqa: N805
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: N805
        return -ctx.lambd * grad_output, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverseFn.apply(x, float(lambd))


@dataclass(frozen=True)
class AdvConfig:
    enabled: bool = False
    lambda_max: float = 0.1
    warmup: float = 0.3  # fraction of total steps
    head_hidden: int = 128


class SubjectHead(nn.Module):
    def __init__(self, in_dim: int, n_subjects: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(n_subjects)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def linear_warmup(progress: float, warmup: float) -> float:
    """
    progress: 0..1
    warmup: fraction in (0,1]
    """
    if warmup <= 0:
        return 1.0
    if progress <= 0:
        return 0.0
    if progress >= warmup:
        return 1.0
    return float(progress / warmup)

