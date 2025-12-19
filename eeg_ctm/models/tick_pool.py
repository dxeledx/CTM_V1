"""
Learnable tick pooling for CTM multi-tick outputs.

Motivation (user feedback, 2025-12-19):
  - The existing certainty-weighted aggregation is a heuristic.
  - Cross-subject MI often has subject-specific "useful time windows"; a learnable
    pooling over ticks is a simple, low-cost way to let the model learn which ticks
    to trust for each sample.

Design doc mapping:
  - design.md "多 tick 聚合策略（Inference readout）" (upgrade from heuristic to learnable)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TickPoolConfig:
    enabled: bool = False
    n_heads: int = 4
    dropout: float = 0.0
    layernorm: bool = True
    temperature: float = 1.0  # softmax temperature on attention scores


class LearnedTickAttentionPool(nn.Module):
    """
    Learnable attention pooling over ticks.

    Inputs:
      logits_ticks: [B,C,T]
      z_ticks:      [B,T,D]
    Output:
      logits: [B,C]
    """

    def __init__(self, d_rep: int, *, n_heads: int = 4, dropout: float = 0.0, layernorm: bool = True, temperature: float = 1.0) -> None:
        super().__init__()
        d_rep = int(d_rep)
        n_heads = int(n_heads)
        if d_rep <= 0:
            raise ValueError("d_rep must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")

        self.temperature = float(temperature)
        self.ln = nn.LayerNorm(d_rep) if bool(layernorm) else nn.Identity()
        self.drop = nn.Dropout(float(dropout))
        self.scorer = nn.Linear(d_rep, n_heads, bias=True)

        if n_heads == 1:
            self.head_weights = None
        else:
            # Class-independent head mixing, initialized to uniform.
            self.head_weights = nn.Parameter(torch.zeros(n_heads))

    def forward(self, logits_ticks: torch.Tensor, z_ticks: torch.Tensor) -> torch.Tensor:
        if logits_ticks.ndim != 3:
            raise ValueError(f"Expected logits_ticks [B,C,T], got {tuple(logits_ticks.shape)}")
        if z_ticks.ndim != 3:
            raise ValueError(f"Expected z_ticks [B,T,D], got {tuple(z_ticks.shape)}")
        B, C, T = logits_ticks.shape
        if z_ticks.shape[0] != B or z_ticks.shape[1] != T:
            raise ValueError("Shape mismatch between logits_ticks and z_ticks")

        z = self.drop(self.ln(z_ticks))
        scores = self.scorer(z)  # [B,T,H]
        temp = float(self.temperature)
        if temp <= 0.0:
            raise ValueError("temperature must be > 0")
        w = torch.softmax(scores / temp, dim=1)  # over ticks => [B,T,H]

        # Pool logits per head: [B,C,T] x [B,T,H] -> [B,C,H]
        pooled = torch.einsum("bct,bth->bch", logits_ticks, w)
        if self.head_weights is None:
            return pooled.squeeze(-1)
        mix = torch.softmax(self.head_weights, dim=0)  # [H]
        return (pooled * mix.view(1, 1, -1)).sum(dim=-1)

