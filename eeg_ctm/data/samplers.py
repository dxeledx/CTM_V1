"""
Batch samplers for (subject, class)-balanced training.

Design doc mapping:
  - design.md "你需要的 Sampler（否则 SupCon 很难工作）"
  - design.md sampler: pick S subjects, m samples/class/subject => batch = S*C*m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SamplerConfig:
    type: Literal["none", "subject_class_balanced"] = "none"
    subjects_per_batch: int = 4  # S
    samples_per_class: int = 2  # m


class SubjectClassBalancedBatchSampler(Sampler[List[int]]):
    """
    Yields batches with approximately balanced subject×class composition.
    """

    def __init__(
        self,
        *,
        subjects: Sequence[int],
        labels: Sequence[int],
        subjects_per_batch: int,
        samples_per_class: int,
        num_classes: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        if self.subjects.shape[0] != self.labels.shape[0]:
            raise ValueError("subjects/labels length mismatch")

        self.subjects_per_batch = int(subjects_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.unique_subjects = np.unique(self.subjects).tolist()

        # Build index pools: (subj, cls) -> indices
        self.pools: dict[tuple[int, int], np.ndarray] = {}
        for subj in self.unique_subjects:
            for cls in range(self.num_classes):
                idx = np.where((self.subjects == subj) & (self.labels == cls))[0]
                if idx.size > 0:
                    self.pools[(int(subj), int(cls))] = idx

        self.batch_size = self.subjects_per_batch * self.num_classes * self.samples_per_class
        if self.batch_size <= 0:
            raise ValueError("Invalid batch size from sampler config")

        self._n_batches = self.subjects.shape[0] // self.batch_size if self.drop_last else int(np.ceil(self.subjects.shape[0] / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self._n_batches)

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        n_subj_total = len(self.unique_subjects)
        S = min(self.subjects_per_batch, n_subj_total)

        for _ in range(len(self)):
            # Choose subjects for this batch (without replacement)
            perm = torch.randperm(n_subj_total, generator=g).tolist()
            chosen_subjects = [self.unique_subjects[i] for i in perm[:S]]

            batch: list[int] = []
            for subj in chosen_subjects:
                for cls in range(self.num_classes):
                    key = (int(subj), int(cls))
                    pool = self.pools.get(key)
                    if pool is None or pool.size == 0:
                        continue
                    # Sample with replacement if needed.
                    idx = torch.randint(0, pool.size, (self.samples_per_class,), generator=g).tolist()
                    batch.extend(pool[i] for i in idx)

            # If some (subj,cls) missing, fill randomly from full dataset to keep batch_size stable.
            if len(batch) < self.batch_size:
                filler = torch.randint(0, self.subjects.shape[0], (self.batch_size - len(batch),), generator=g).tolist()
                batch.extend(int(i) for i in filler)
            elif len(batch) > self.batch_size:
                batch = batch[: self.batch_size]

            yield batch

