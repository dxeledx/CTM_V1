"""
Supervised contrastive learning (cross-subject positives + same-instance positives).

Design doc mapping:
  - design.md "跨被试 supervised contrastive"
  - design.md positives:
      * same class AND different subject
      * same instance (two augmented views)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SupConConfig:
    enabled: bool = False
    tau: float = 0.07
    lambda_con: float = 0.1
    proj_dim: int = 128
    proj_hidden: int = 256
    include_same_instance: bool = True


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.nn.functional.normalize(x, dim=-1)


def supervised_contrastive_loss(
    emb: torch.Tensor,  # [N,d], already normalized
    labels: torch.Tensor,  # [N]
    subjects: torch.Tensor,  # [N]
    *,
    tau: float,
    include_same_instance: bool,
) -> torch.Tensor:
    """
    SupCon with cross-subject positives:
      pos(i,j) if labels equal and subjects differ.
    Optionally includes same-instance positives for two-view batches (N=2B):
      i <-> i+B

    If an anchor has no positives, it is ignored (unless include_same_instance=True, which should prevent that).
    """
    if emb.ndim != 2:
        raise ValueError(f"Expected emb [N,d], got {tuple(emb.shape)}")
    N = emb.shape[0]
    if labels.shape != (N,) or subjects.shape != (N,):
        raise ValueError("labels/subjects shape mismatch")

    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be > 0")

    # Similarity logits
    logits = emb @ emb.t() / tau  # [N,N]
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # Mask out self in denominators
    eye = torch.eye(N, device=emb.device, dtype=torch.bool)
    logits_mask = ~eye

    # Positive mask: same class, different subject
    same_class = labels.view(N, 1).eq(labels.view(1, N))
    diff_subj = subjects.view(N, 1).ne(subjects.view(1, N))
    pos_mask = same_class & diff_subj & logits_mask

    if include_same_instance:
        if N % 2 != 0:
            raise ValueError("include_same_instance=True requires N=2B")
        B = N // 2
        idx = torch.arange(B, device=emb.device)
        pos_mask[idx, idx + B] = True
        pos_mask[idx + B, idx] = True

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)  # [N]
    # Avoid NaNs: only average anchors with at least 1 positive.
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.zeros((), device=emb.device, dtype=emb.dtype)

    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_count.float() + 1e-12)
    loss = -mean_log_prob_pos[valid].mean()
    return loss

