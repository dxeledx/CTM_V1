"""
EEG Tokenizer (v1): Conv-Patch Tokenizer producing ~20 tokens for CTM cross-attention.

Design doc mapping:
  - design.md §3 "Tokenizer v1（强烈建议这样起步）"
  - design.md "Tokenizer v1 总体接口" and "默认超参（v1 baseline）"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math

import torch
import torch.nn as nn


TokenizerPoolType = Literal["avg", "max", "conv_stride"]
TokenizerPosEmb = Literal["learnable", "sinusoidal", "none"]
TokenizerNorm = Literal["BN", "GN", "none"]
TokenizerAct = Literal["ELU", "GELU"]
TokenizerBranchFusion = Literal["concat", "gated_concat"]


@dataclass(frozen=True)
class TokenizerV1Config:
    # Input shape (fixed for BCI-IV-2a)
    C: int = 22
    T: int = 1000

    # Token shape
    d_kv: int = 128
    # Multi-scale temporal conv
    temporal_kernels: tuple[int, ...] = (25, 63)
    F_per_branch: int = 8
    branch_fusion: TokenizerBranchFusion = "concat"

    # Spatial mixing
    spatial_depth_multiplier: int = 2
    spatial_mixer: Literal["dw_conv(C,1)", "none"] = "dw_conv(C,1)"

    # Tokenization/pooling
    token_pool_type: TokenizerPoolType = "avg"
    token_pool_kernel: int = 50
    token_stride: int = 50

    # Norm/act/dropout
    conv_norm: TokenizerNorm = "BN"
    token_norm: Literal["LN", "none"] = "LN"
    act: TokenizerAct = "ELU"
    dropout_p: float = 0.25

    # Positional encoding
    pos_emb: TokenizerPosEmb = "learnable"


def _make_act(act: TokenizerAct) -> nn.Module:
    if act == "ELU":
        return nn.ELU()
    if act == "GELU":
        return nn.GELU()
    raise ValueError(act)


def _make_conv_norm(norm: TokenizerNorm, num_channels: int) -> nn.Module:
    if norm == "BN":
        return nn.BatchNorm2d(num_channels)
    if norm == "GN":
        # default: 8 groups or fallback to 1
        groups = 8 if num_channels % 8 == 0 else 1
        return nn.GroupNorm(groups, num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(norm)


def _make_token_norm(norm: str, d_kv: int) -> nn.Module:
    if norm == "LN":
        return nn.LayerNorm(d_kv)
    if norm == "none":
        return nn.Identity()
    raise ValueError(norm)


def _num_tokens(T: int, kernel: int, stride: int) -> int:
    if T <= 0:
        raise ValueError("T must be positive")
    if kernel <= 0 or stride <= 0:
        raise ValueError("kernel/stride must be positive")
    # For pooling without padding: floor((T - kernel)/stride) + 1
    if T < kernel:
        raise ValueError(f"T({T}) < kernel({kernel})")
    return (T - kernel) // stride + 1


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequences.
    """

    def __init__(self, n_positions: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(n_positions, d_model)
        position = torch.arange(0, n_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1,N,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D]
        return self.pe[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)


class ConvPatchTokenizerV1(nn.Module):
    """
    Tokenizer v1 (Conv-Patch).

    Input:  x [B,C,T]
    Output: tokens [B,N,d_kv]
    """

    def __init__(self, cfg: TokenizerV1Config) -> None:
        super().__init__()
        self.cfg = cfg

        C = int(cfg.C)
        F_per = int(cfg.F_per_branch)
        kernels = list(cfg.temporal_kernels)
        if len(kernels) < 1:
            raise ValueError("temporal_kernels must be non-empty")

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, F_per, kernel_size=(1, int(k)), stride=(1, 1), padding=(0, int(k) // 2), bias=False),
                )
                for k in kernels
            ]
        )
        self.branch_fusion = cfg.branch_fusion
        if self.branch_fusion == "gated_concat":
            self.branch_gates = nn.Parameter(torch.zeros(len(kernels)))
        else:
            self.branch_gates = None

        F_total = F_per * len(kernels)
        spatial_mult = int(cfg.spatial_depth_multiplier)
        if cfg.spatial_mixer == "dw_conv(C,1)":
            self.spatial = nn.Conv2d(
                F_total,
                F_total * spatial_mult,
                kernel_size=(C, 1),
                groups=F_total,
                bias=False,
            )
            F_after = F_total * spatial_mult
        elif cfg.spatial_mixer == "none":
            self.spatial = nn.Identity()
            F_after = F_total
        else:
            raise ValueError(cfg.spatial_mixer)

        self.proj = nn.Conv2d(F_after, int(cfg.d_kv), kernel_size=(1, 1), bias=False)
        self.conv_norm = _make_conv_norm(cfg.conv_norm, int(cfg.d_kv))
        self.act = _make_act(cfg.act)
        self.dropout = nn.Dropout(float(cfg.dropout_p))

        if cfg.token_pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=(1, int(cfg.token_pool_kernel)), stride=(1, int(cfg.token_stride)))
        elif cfg.token_pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=(1, int(cfg.token_pool_kernel)), stride=(1, int(cfg.token_stride)))
        elif cfg.token_pool_type == "conv_stride":
            self.pool = nn.Conv2d(
                int(cfg.d_kv),
                int(cfg.d_kv),
                kernel_size=(1, int(cfg.token_pool_kernel)),
                stride=(1, int(cfg.token_stride)),
                groups=int(cfg.d_kv),
                bias=False,
            )
        else:
            raise ValueError(cfg.token_pool_type)

        self.N = _num_tokens(int(cfg.T), int(cfg.token_pool_kernel), int(cfg.token_stride))
        self.token_norm = _make_token_norm(cfg.token_norm, int(cfg.d_kv))

        if cfg.pos_emb == "learnable":
            self.pos_emb = nn.Parameter(torch.zeros(1, self.N, int(cfg.d_kv)))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            self.pos_enc = None
        elif cfg.pos_emb == "sinusoidal":
            self.pos_emb = None
            self.pos_enc = SinusoidalPositionalEncoding(self.N, int(cfg.d_kv))
        elif cfg.pos_emb == "none":
            self.pos_emb = None
            self.pos_enc = None
        else:
            raise ValueError(cfg.pos_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x as [B,C,T], got shape={tuple(x.shape)}")
        B, C, T = x.shape
        if C != int(self.cfg.C):
            raise ValueError(f"Expected C={self.cfg.C}, got C={C}")
        if T != int(self.cfg.T):
            raise ValueError(f"Expected T={self.cfg.T}, got T={T}")

        x2 = x.unsqueeze(1)  # [B,1,C,T]

        outs = []
        if self.branch_fusion == "gated_concat":
            gates = torch.sigmoid(self.branch_gates)  # [n_branches]
            for i, br in enumerate(self.branches):
                outs.append(br(x2) * gates[i])
        else:
            for br in self.branches:
                outs.append(br(x2))
        f = torch.cat(outs, dim=1)  # [B,F_total,C,T]

        f = self.spatial(f)  # [B,F_after,1,T] (if dw conv)
        f = self.proj(f)  # [B,d_kv,1,T]
        f = self.conv_norm(f)
        f = self.act(f)
        f = self.dropout(f)

        f = self.pool(f)  # [B,d_kv,1,N]
        tokens = f.squeeze(2).transpose(1, 2).contiguous()  # [B,N,d_kv]
        if tokens.shape[1] != self.N:
            raise RuntimeError(f"Unexpected N={tokens.shape[1]} (expected {self.N})")

        if self.pos_emb is not None:
            tokens = tokens + self.pos_emb.to(dtype=tokens.dtype, device=tokens.device)
        elif self.pos_enc is not None:
            tokens = tokens + self.pos_enc(tokens)

        tokens = self.token_norm(tokens)
        return tokens

