"""
Token-level fusion/modulation for train-time augmentation injection.

Motivation (user feedback, 2025-12-19):
  - Plain "concat" injection of S&R-augmented samples can be unstable for cross-subject MI:
      * OOD artifacts: model may learn augmentation traces instead of MI patterns
      * Donor pool bias: cross-subject donors may leak subject signatures into the objective
  - A more stable alternative is to treat the augmented view as a *conditioning signal*
    and fuse/modulate the main tokens, rather than treating the augmented view as an
    additional independent sample.

Design doc mapping:
  - design.md "增强层骨架（Train only）：S&R 模块" (injection strategy upgrade)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TokenFusionType = Literal["adain", "film", "xattn"]


@dataclass(frozen=True)
class TokenFusionConfig:
    """
    Train-time token fusion to inject an augmented view into the main view.

    All modes are designed to be *shape-preserving*: [B,N,D] -> [B,N,D].
    """

    mode: TokenFusionType = "adain"

    # FiLM
    film_hidden: int = 128

    # Cross-attention fusion
    xattn_heads: int = 4
    xattn_dropout: float = 0.0
    xattn_layernorm: bool = True


class AdaINTokenFusion(nn.Module):
    """
    AdaIN-style token modulation (parameter-free).

    For each sample, match the token-wise feature mean/std of `tokens` to `cond_tokens`.
    If tokens == cond_tokens, this is exactly identity.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or cond_tokens.ndim != 3:
            raise ValueError("Expected tokens/cond_tokens as [B,N,D]")
        if tokens.shape != cond_tokens.shape:
            raise ValueError(f"Shape mismatch: tokens={tuple(tokens.shape)} cond_tokens={tuple(cond_tokens.shape)}")

        mu_x = tokens.mean(dim=1, keepdim=True)
        mu_y = cond_tokens.mean(dim=1, keepdim=True)
        sig_x = tokens.std(dim=1, keepdim=True).clamp_min(self.eps)
        sig_y = cond_tokens.std(dim=1, keepdim=True).clamp_min(self.eps)
        x_norm = (tokens - mu_x) / sig_x
        return x_norm * sig_y + mu_y


class FiLMTokenFusion(nn.Module):
    """
    FiLM-style token modulation with a small MLP producing (gamma, beta).

    We use an identity-friendly parameterization:
      out = tokens * (1 + gamma) + beta
    and initialize the last layer to 0 so that gamma,beta start near 0.
    """

    def __init__(self, d_model: int, hidden: int = 128) -> None:
        super().__init__()
        d_model = int(d_model)
        hidden = int(hidden)
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )
        # Identity init: last layer -> zeros so gamma=beta=0 at start.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or cond_tokens.ndim != 3:
            raise ValueError("Expected tokens/cond_tokens as [B,N,D]")
        if tokens.shape != cond_tokens.shape:
            raise ValueError(f"Shape mismatch: tokens={tuple(tokens.shape)} cond_tokens={tuple(cond_tokens.shape)}")

        pooled = cond_tokens.mean(dim=1)  # [B,D]
        gamma_beta = self.mlp(pooled)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B,D], [B,D]
        return tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class CrossAttentionTokenFusion(nn.Module):
    """
    Cross-attention fusion: main tokens are Query; augmented tokens are Key/Value.

    Uses a learnable residual gate initialized to 0 for stability:
      out = LN(tokens + tanh(gate) * attn(tokens, cond_tokens))
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 4,
        dropout: float = 0.0,
        layernorm: bool = True,
    ) -> None:
        super().__init__()
        d_model = int(d_model)
        n_heads = int(n_heads)
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by n_heads({n_heads})")

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=float(dropout), batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))
        self.ln = nn.LayerNorm(d_model) if bool(layernorm) else nn.Identity()

    def forward(self, tokens: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or cond_tokens.ndim != 3:
            raise ValueError("Expected tokens/cond_tokens as [B,N,D]")
        if tokens.shape != cond_tokens.shape:
            raise ValueError(f"Shape mismatch: tokens={tuple(tokens.shape)} cond_tokens={tuple(cond_tokens.shape)}")

        out, _ = self.attn(tokens, cond_tokens, cond_tokens, need_weights=False)
        g = torch.tanh(self.gate)
        return self.ln(tokens + g * out)


def build_token_fuser(cfg: TokenFusionConfig, *, d_model: int) -> nn.Module:
    """
    Factory for token fusion modules.
    """
    if cfg.mode == "adain":
        return AdaINTokenFusion()
    if cfg.mode == "film":
        return FiLMTokenFusion(d_model=int(d_model), hidden=int(cfg.film_hidden))
    if cfg.mode == "xattn":
        return CrossAttentionTokenFusion(
            d_model=int(d_model),
            n_heads=int(cfg.xattn_heads),
            dropout=float(cfg.xattn_dropout),
            layernorm=bool(cfg.xattn_layernorm),
        )
    raise ValueError(cfg.mode)

