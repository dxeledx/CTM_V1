"""
End-to-end EEG-CTM model: Tokenizer v1 + CTM core.

Design doc mapping:
  - design.md "Tokenizer v1 总体接口"
  - design.md "连接到 CTM：kv_projector"
  - design.md "CTM core 的 tick 顺序"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from eeg_ctm.models.ctm_core import CTMCore, CTMCoreConfig
from eeg_ctm.models.tokenizer import ConvPatchTokenizerV1, TokenizerV1Config


@dataclass(frozen=True)
class EEGCTMConfig:
    tokenizer: TokenizerV1Config = TokenizerV1Config()
    ctm: CTMCoreConfig = CTMCoreConfig()


class EEGCTM(nn.Module):
    def __init__(self, cfg: EEGCTMConfig, *, pair_bank: "torch.nn.Module") -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ConvPatchTokenizerV1(cfg.tokenizer)
        self.ctm = CTMCore(cfg.ctm, d_kv=int(cfg.tokenizer.d_kv), pair_bank=pair_bank)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,C,T] (expected trial-wise standardized upstream)
        Returns:
          logits_ticks: [B,num_classes,T_internal]
          certainty:   [B,T_internal]
          z_ticks:     [B,T_internal,D]
        """
        tokens = self.tokenizer(x)
        return self.ctm(tokens)

