"""
Tick aggregation utilities (logits + representations).

Design doc mapping:
  - design.md "多 tick 聚合策略（Inference readout）"
  - design.md "aggregate_rep(z_ticks, logits_ticks)" for adv/contrastive heads
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ReadoutType = Literal["last", "most_certain", "mean_logits", "certainty_weighted"]
RepAggType = Literal["last", "mean", "certainty_weighted"]


@dataclass(frozen=True)
class CertaintyWeightedConfig:
    alpha: float = 5.0  # softmax temperature on certainty


def aggregate_logits(
    logits_ticks: torch.Tensor,  # [B,C,T]
    certainty: torch.Tensor,  # [B,T]
    *,
    mode: ReadoutType,
    cw: CertaintyWeightedConfig = CertaintyWeightedConfig(),
) -> torch.Tensor:
    if logits_ticks.ndim != 3:
        raise ValueError(f"Expected logits_ticks [B,C,T], got {tuple(logits_ticks.shape)}")
    if certainty.ndim != 2:
        raise ValueError(f"Expected certainty [B,T], got {tuple(certainty.shape)}")
    B, C, T = logits_ticks.shape
    if certainty.shape != (B, T):
        raise ValueError("certainty shape mismatch")

    if mode == "last":
        return logits_ticks[:, :, -1]
    if mode == "mean_logits":
        return logits_ticks.mean(dim=-1)
    if mode == "most_certain":
        idx = certainty.argmax(dim=-1)  # [B]
        out = logits_ticks.permute(0, 2, 1)  # [B,T,C]
        return out[torch.arange(B, device=logits_ticks.device), idx, :]
    if mode == "certainty_weighted":
        w = torch.softmax(float(cw.alpha) * certainty, dim=-1)  # [B,T]
        return torch.einsum("bct,bt->bc", logits_ticks, w)
    raise ValueError(mode)


def aggregate_rep(
    z_ticks: torch.Tensor,  # [B,T,D]
    certainty: torch.Tensor,  # [B,T]
    *,
    mode: RepAggType = "certainty_weighted",
    cw: CertaintyWeightedConfig = CertaintyWeightedConfig(),
) -> torch.Tensor:
    if z_ticks.ndim != 3:
        raise ValueError(f"Expected z_ticks [B,T,D], got {tuple(z_ticks.shape)}")
    if certainty.ndim != 2:
        raise ValueError(f"Expected certainty [B,T], got {tuple(certainty.shape)}")
    B, T, D = z_ticks.shape
    if certainty.shape != (B, T):
        raise ValueError("certainty shape mismatch")

    if mode == "last":
        return z_ticks[:, -1, :]
    if mode == "mean":
        return z_ticks.mean(dim=1)
    if mode == "certainty_weighted":
        w = torch.softmax(float(cw.alpha) * certainty, dim=-1)  # [B,T]
        return torch.einsum("btd,bt->bd", z_ticks, w)
    raise ValueError(mode)

