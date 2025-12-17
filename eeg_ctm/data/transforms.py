"""
Data transforms for EEG-CTM.

Design doc mapping:
  - design.md §1 "Epoch / 标准化（Trial-wise）"
  - design.md §2 "Trial-wise 标准化"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TrialWiseStandardize:
    """
    Trial-wise standardization (no transductive leakage).

    For each trial X[C, T], compute per-channel statistics over time and standardize:
        X'[c, t] = (X[c, t] - mu[c]) / (sigma[c] + eps)

    Modes:
      - "zscore": mean/std
      - "robust": median / MAD (scaled)
    """

    mode: str = "zscore"  # "zscore" | "robust"
    eps: float = 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Expected x as [C,T], got shape={x.shape}")
        if self.mode == "zscore":
            mu = x.mean(axis=1, keepdims=True)
            sigma = x.std(axis=1, keepdims=True)
            return (x - mu) / (sigma + self.eps)
        if self.mode == "robust":
            med = np.median(x, axis=1, keepdims=True)
            mad = np.median(np.abs(x - med), axis=1, keepdims=True)
            # 1.4826 makes MAD consistent with std for Gaussian
            sigma = 1.4826 * mad
            return (x - med) / (sigma + self.eps)
        raise ValueError(f"Unknown mode={self.mode}")


def trialwise_standardize_torch(x: torch.Tensor, *, mode: str = "zscore", eps: float = 1e-6) -> torch.Tensor:
    """
    Torch batch version for x shaped [B,C,T] or [C,T].
    """
    if x.ndim == 2:
        x_ = x.unsqueeze(0)
        squeeze = True
    elif x.ndim == 3:
        x_ = x
        squeeze = False
    else:
        raise ValueError(f"Expected x as [C,T] or [B,C,T], got shape={tuple(x.shape)}")

    if mode == "zscore":
        mu = x_.mean(dim=-1, keepdim=True)
        sigma = x_.std(dim=-1, keepdim=True)
        y = (x_ - mu) / (sigma + eps)
    elif mode == "robust":
        med = x_.median(dim=-1, keepdim=True).values
        mad = (x_ - med).abs().median(dim=-1, keepdim=True).values
        sigma = 1.4826 * mad
        y = (x_ - med) / (sigma + eps)
    else:
        raise ValueError(f"Unknown mode={mode}")

    return y.squeeze(0) if squeeze else y
