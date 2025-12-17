"""
Train/validation split helpers.

This is used to avoid "single held-out subject as validation", which can be unstable in LOSO.
Instead, we split validation trials *within* training subjects (subject-stratified, optionally class-stratified).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WithinSubjectValSplitConfig:
    val_fraction: float = 0.3
    seed: int = 0
    stratify_by_class: bool = True


def split_within_subjects(
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    cfg: WithinSubjectValSplitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_idx, val_idx) indices into the given arrays.

    For each subject, we sample `val_fraction` trials into validation.
    If `stratify_by_class=True`, the split is stratified by y *within each subject*.
    """
    y = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.int64)
    if y.shape[0] != subjects.shape[0]:
        raise ValueError("y/subjects length mismatch")

    val_fraction = float(cfg.val_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1)")

    rng = np.random.RandomState(int(cfg.seed))
    train_parts = []
    val_parts = []

    unique_subjects = np.unique(subjects)
    for s in unique_subjects:
        idx = np.where(subjects == s)[0]
        if idx.size == 0:
            continue

        if cfg.stratify_by_class:
            y_s = y[idx]
            # Group indices by class, sample per class.
            classes = np.unique(y_s)
            # Fallback to random split if not enough variety.
            if classes.size < 2:
                perm = rng.permutation(idx)
                n_val = max(1, int(round(idx.size * val_fraction)))
                val_parts.append(perm[:n_val])
                train_parts.append(perm[n_val:])
                continue

            # Stratified sampling per class.
            val_idx_s = []
            train_idx_s = []
            for c in classes:
                idx_c = idx[y_s == c]
                perm_c = rng.permutation(idx_c)
                n_val_c = max(1, int(round(idx_c.size * val_fraction)))
                n_val_c = min(n_val_c, idx_c.size - 1)  # keep at least 1 train sample if possible
                val_idx_s.append(perm_c[:n_val_c])
                train_idx_s.append(perm_c[n_val_c:])
            val_parts.append(np.concatenate(val_idx_s))
            train_parts.append(np.concatenate(train_idx_s))
        else:
            perm = rng.permutation(idx)
            n_val = max(1, int(round(idx.size * val_fraction)))
            val_parts.append(perm[:n_val])
            train_parts.append(perm[n_val:])

    train_idx = np.concatenate(train_parts).astype(np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64)
    return train_idx, val_idx

