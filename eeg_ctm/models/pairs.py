"""
Pair sampling utilities for CTM synchronisation (fixed + cached).

Design doc mapping:
  - design.md "Pair 采样策略（固定随机 + 少量 self-pairs）"
  - design.md "pair 采样固定并缓存"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class PairBankConfig:
    D: int = 256
    K_action: int = 256
    K_out: int = 256
    n_self: int = 16
    seed: int = 0


def sample_pairs(D: int, Dsub: int, nself: int, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random neuron index pairs with a small fraction of self-pairs.

    Returns:
      left_idx: [Dsub] long
      right_idx: [Dsub] long
    """
    D = int(D)
    Dsub = int(Dsub)
    nself = int(nself)
    if not (0 <= nself <= Dsub):
        raise ValueError("Expected 0 <= nself <= Dsub")
    if D <= 0 or Dsub <= 0:
        raise ValueError("D and Dsub must be positive")

    if nself > 0:
        self_idx = torch.randperm(D, generator=generator)[:nself]
        left_self = self_idx
        right_self = self_idx
    else:
        left_self = torch.empty((0,), dtype=torch.long)
        right_self = torch.empty((0,), dtype=torch.long)

    n_rand = Dsub - nself
    left = torch.randint(0, D, (n_rand,), generator=generator, dtype=torch.long)
    right = torch.randint(0, D, (n_rand,), generator=generator, dtype=torch.long)

    idx_left = torch.cat([left_self, left], dim=0)
    idx_right = torch.cat([right_self, right], dim=0)
    return idx_left, idx_right


class PairBank(torch.nn.Module):
    """
    Holds fixed action/out pair indices as buffers for reproducibility.
    """

    def __init__(self, cfg: PairBankConfig) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(int(cfg.seed))
        act_left, act_right = sample_pairs(cfg.D, cfg.K_action, cfg.n_self, generator=g)
        out_left, out_right = sample_pairs(cfg.D, cfg.K_out, cfg.n_self, generator=g)

        self.register_buffer("act_left", act_left)
        self.register_buffer("act_right", act_right)
        self.register_buffer("out_left", out_left)
        self.register_buffer("out_right", out_right)

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "act_left": self.act_left.detach().cpu(),
            "act_right": self.act_right.detach().cpu(),
            "out_left": self.out_left.detach().cpu(),
            "out_right": self.out_right.detach().cpu(),
        }

    @classmethod
    def from_dict(cls, data: dict, *, device: Optional[torch.device] = None) -> "PairBank":
        # Minimal validation.
        required = {"act_left", "act_right", "out_left", "out_right"}
        if not required.issubset(set(data.keys())):
            raise ValueError(f"Pair dict missing keys: {required - set(data.keys())}")
        cfg = PairBankConfig(
            D=int(max(data["act_left"].max(), data["act_right"].max(), data["out_left"].max(), data["out_right"].max()).item()) + 1,
            K_action=int(data["act_left"].numel()),
            K_out=int(data["out_left"].numel()),
            n_self=0,
            seed=0,
        )
        bank = cls(cfg)
        bank.act_left = data["act_left"].long().to(device=device)
        bank.act_right = data["act_right"].long().to(device=device)
        bank.out_left = data["out_left"].long().to(device=device)
        bank.out_right = data["out_right"].long().to(device=device)
        return bank


def load_or_create_pairbank(cfg: PairBankConfig, *, cache_path: Path) -> PairBank:
    """
    Create a PairBank deterministically and cache it to disk. If already present, load it.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        bank = PairBank.from_dict(data)
        return bank

    bank = PairBank(cfg)
    torch.save(bank.to_dict(), cache_path)
    return bank

