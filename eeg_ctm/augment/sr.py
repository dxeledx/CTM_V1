"""
Train-only EEG augmentation: Segmentation & Recombination (S&R).

Design doc mapping:
  - design.md "增强层骨架（Train only）：S&R 模块"
  - design.md "LOSO 场景下不会踩坑的实现细则" (no leakage into donor pool)
  - design.md "拼接边界 cross-fade（余弦窗平滑）"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SRAugmentConfig:
    enabled: bool = True
    k_segments: int = 8
    # How S&R is injected into the batch is handled by the training loop. This module only generates x_aug.
    cross_fade: bool = True
    cross_fade_len: int = 20  # samples

    # If True, attempt to avoid sampling the same trial as donor.
    avoid_self: bool = True


class ClasswiseMemoryBank:
    """
    Class-wise donor pool for S&R.

    Stores only training trials (caller responsibility) to avoid leakage.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self._x: list[list[torch.Tensor]] = [[] for _ in range(self.num_classes)]
        self._uid: list[list[int]] = [[] for _ in range(self.num_classes)]

    def add(self, x: torch.Tensor, y: int, uid: int) -> None:
        # x: [C,T] on CPU recommended
        cls = int(y)
        self._x[cls].append(x)
        self._uid[cls].append(int(uid))

    def size(self, cls: int) -> int:
        return len(self._x[int(cls)])

    def sample(self, cls: int, *, exclude_uid: Optional[int], generator: torch.Generator) -> torch.Tensor:
        """
        Sample a donor trial tensor for the given class.
        """
        cls = int(cls)
        n = self.size(cls)
        if n == 0:
            raise RuntimeError(f"Empty memory bank for class={cls}")

        if exclude_uid is None or n == 1:
            j = int(torch.randint(0, n, (1,), generator=generator).item())
            return self._x[cls][j]

        # Try a few times to avoid self, then fall back.
        for _ in range(10):
            j = int(torch.randint(0, n, (1,), generator=generator).item())
            if self._uid[cls][j] != int(exclude_uid):
                return self._x[cls][j]
        return self._x[cls][j]


def _cosine_fade_weights(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 1:
        return torch.ones((length,), device=device, dtype=dtype)
    t = torch.arange(length, device=device, dtype=dtype)
    denom = float(length - 1)
    # w goes from 1 -> 0
    return 0.5 * (1.0 + torch.cos(torch.pi * (t / denom)))


class SegmentationRecombinationAugmenter:
    """
    Baseline S&R augmentation (Conformer/CTNet style).

    Input x: [B,C,T] (assumed already trial-wise standardized)
    Output x_aug: [B,C,T]
    """

    def __init__(self, cfg: SRAugmentConfig, *, num_classes: int) -> None:
        self.cfg = cfg
        self.num_classes = int(num_classes)

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        uid: Optional[torch.Tensor],
        *,
        bank: ClasswiseMemoryBank,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if not self.cfg.enabled:
            return x

        if x.ndim != 3:
            raise ValueError(f"Expected x as [B,C,T], got shape={tuple(x.shape)}")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("y must be [B]")

        B, C, T = x.shape
        K = int(self.cfg.k_segments)
        if K <= 1:
            raise ValueError("k_segments must be > 1 for S&R")

        # Segment boundaries (equal split; last gets remainder).
        base = T // K
        seg_lens = [base] * K
        seg_lens[-1] = T - base * (K - 1)
        boundaries = [0]
        for L in seg_lens:
            boundaries.append(boundaries[-1] + L)
        assert boundaries[-1] == T

        x_aug = torch.empty_like(x)
        for b in range(B):
            cls = int(y[b].item())
            exclude = int(uid[b].item()) if (uid is not None and self.cfg.avoid_self) else None

            # Build recombined trial.
            out = []
            for k in range(K):
                donor = bank.sample(cls, exclude_uid=exclude, generator=generator)
                s, e = boundaries[k], boundaries[k + 1]
                out.append(donor[:, s:e].to(device=x.device, dtype=x.dtype))
            x_new = torch.cat(out, dim=-1)  # [C,T]

            # Optional cross-fade to reduce boundary discontinuities.
            if self.cfg.cross_fade and self.cfg.cross_fade_len > 0:
                Lb = int(self.cfg.cross_fade_len)
                w = _cosine_fade_weights(Lb, device=x.device, dtype=x.dtype).view(1, -1)  # [1,Lb]
                for k in range(1, K):
                    idx = boundaries[k]
                    if idx - Lb < 0 or idx + Lb > T:
                        continue
                    prev = x_new[:, idx - Lb : idx]
                    nxt = x_new[:, idx : idx + Lb]
                    x_new[:, idx - Lb : idx] = w * prev + (1.0 - w) * nxt

            x_aug[b] = x_new

        return x_aug

