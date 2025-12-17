"""
BCI Competition IV-2a (MOABB BNCI2014001) data loading for EEG-CTM.

Hard requirements:
  - LOSO (subject-wise) splits with strict leakage prevention.
  - Epoch alignment to cue-onset equivalent window (default: 0–4s @ cue, i.e., 2–6s of each trial).
  - Trial-wise standardization (no train/val/test statistic sharing).

Design doc mapping:
  - design.md §1 "Epoch（trial）截取：cue-onset，0–4s 等价 2–6s"
  - design.md §2 "Trial-wise 标准化：真·Trial-wise Z-score"
  - design.md "LOSO 场景下不会踩坑的实现细则"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BCIIV2aWindow:
    """
    Defines the epoch window to crop from each trial.

    Note: MOABB BNCI2014001 uses MotorImagery events; in practice, setting tmin=0,tmax=4
    matches the common "cue-onset aligned [0,4]s" window described in design.md.
    """

    tmin_s: float = 0.0
    tmax_s: float = 4.0
    drop_last_sample: bool = True  # MOABB epochs often have 1001 samples for 4s@250Hz


@dataclass(frozen=True)
class BCIIV2aClasses:
    """
    Label mapping used throughout the project.
    """

    # Fixed canonical order for class indices.
    names: tuple[str, ...] = ("left_hand", "right_hand", "feet", "tongue")

    @property
    def name_to_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}


class EEGTrialsDataset(Dataset):
    """
    Simple in-memory dataset of fixed-length EEG trials.

    Each item:
      x: torch.FloatTensor [C,T]
      y: torch.LongTensor []
      subject: torch.LongTensor []
      uid: torch.LongTensor []  (unique per trial within the loaded set)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject: np.ndarray,
        uid: Optional[np.ndarray] = None,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if x.ndim != 3:
            raise ValueError(f"Expected x as [N,C,T], got shape={x.shape}")
        if y.shape[0] != x.shape[0] or subject.shape[0] != x.shape[0]:
            raise ValueError("x/y/subject first dimension mismatch")
        if uid is None:
            uid = np.arange(x.shape[0], dtype=np.int64)
        if uid.shape[0] != x.shape[0]:
            raise ValueError("uid first dimension mismatch")

        # Store as torch tensors to speed up training/augmentation.
        self.x = torch.as_tensor(x, dtype=dtype)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.subject = torch.as_tensor(subject, dtype=torch.long)
        self.uid = torch.as_tensor(uid, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "subject": self.subject[idx],
            "uid": self.uid[idx],
        }


def _unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted({int(v) for v in values})


def make_loso_splits(
    subjects: Iterable[int],
    *,
    val_strategy: str = "next",  # "next" | "none" | "fixed" | "within_subject"
    fixed_val_subject: Optional[int] = None,
) -> list[dict]:
    """
    Create LOSO folds.

    Returns list of dicts with keys: fold_idx, train_subjects, val_subject, test_subject.

    Leakage prevention rule:
      - test_subject is never used for training/augmentation/model selection.
      - val_subject (if any) is excluded from training augmentation donor pool.
    """
    subjects_sorted = _unique_sorted(subjects)
    folds: list[dict] = []
    for fold_idx, test_subject in enumerate(subjects_sorted):
        remaining = [s for s in subjects_sorted if s != test_subject]
        if val_strategy in ("none", "within_subject"):
            val_subject = None
            train_subjects = remaining
        elif val_strategy == "fixed":
            if fixed_val_subject is None:
                raise ValueError("fixed_val_subject is required when val_strategy='fixed'")
            if fixed_val_subject == test_subject:
                raise ValueError("fixed_val_subject cannot equal the test subject in a fold")
            if fixed_val_subject not in remaining:
                raise ValueError("fixed_val_subject must be in the subject list")
            val_subject = int(fixed_val_subject)
            train_subjects = [s for s in remaining if s != val_subject]
        elif val_strategy == "next":
            # Deterministic: pick the next subject ID (wrap around).
            higher = [s for s in remaining if s > test_subject]
            val_subject = int(higher[0] if higher else remaining[0])
            train_subjects = [s for s in remaining if s != val_subject]
        else:
            raise ValueError(f"Unknown val_strategy={val_strategy}")

        folds.append(
            {
                "fold_idx": fold_idx,
                "train_subjects": train_subjects,
                "val_subject": val_subject,
                "test_subject": int(test_subject),
            }
        )
    return folds


def load_bciiv2a_moabb(
    subjects: Iterable[int],
    *,
    window: BCIIV2aWindow,
    classes: Optional[BCIIV2aClasses] = None,
    resample_sfreq: Optional[float] = None,
    return_meta: bool = False,
) -> Any:
    """
    Load BCI-IV-2a using MOABB.

    Returns:
      x: [N, C, T] float32
      y: [N] int64  (0..3)
      subject: [N] int64  (original subject ids)
    """
    # Local imports to keep module import-time light.
    from moabb.datasets import BNCI2014001
    from moabb.paradigms import MotorImagery

    classes = classes or BCIIV2aClasses()
    dataset = BNCI2014001()
    paradigm = MotorImagery(
        n_classes=4,
        tmin=float(window.tmin_s),
        tmax=float(window.tmax_s),
        resample=resample_sfreq,
    )

    # MOABB uses local caches; do NOT force downloads here.
    X, y_str, meta = paradigm.get_data(dataset=dataset, subjects=_unique_sorted(subjects))

    # Map labels -> indices
    name_to_idx = classes.name_to_index
    y = np.asarray([name_to_idx[str(lbl)] for lbl in y_str], dtype=np.int64)
    subj = meta["subject"].to_numpy(dtype=np.int64)

    X = np.asarray(X, dtype=np.float32)  # [N,C,T]
    if window.drop_last_sample and X.shape[-1] > 0:
        # 4 seconds @ 250Hz yields 1001 samples in MNE (inclusive endpoint); we want 1000.
        X = X[..., : X.shape[-1] - 1]
    if return_meta:
        return X, y, subj, meta
    return X, y, subj


def subset_by_subjects(
    x: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    subjects_keep: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter trials by subject ids.

    Returns (x_sub, y_sub, subject_sub, uid_sub) where uid_sub are indices into the original arrays.
    """
    subjects_keep_set = {int(s) for s in subjects_keep}
    mask = np.isin(subject, np.fromiter(subjects_keep_set, dtype=np.int64))
    uid = np.flatnonzero(mask).astype(np.int64)
    return x[mask], y[mask], subject[mask], uid


def build_fold_datasets(
    *,
    all_subjects: Iterable[int],
    fold: dict,
    window: BCIIV2aWindow,
    classes: Optional[BCIIV2aClasses] = None,
    resample_sfreq: Optional[float] = None,
) -> dict[str, EEGTrialsDataset]:
    """
    Convenience: load all requested subjects once, then split into train/val/test datasets.
    """
    X, y, subj = load_bciiv2a_moabb(
        all_subjects,
        window=window,
        classes=classes,
        resample_sfreq=resample_sfreq,
    )

    x_tr, y_tr, s_tr, uid_tr = subset_by_subjects(X, y, subj, fold["train_subjects"])
    train_ds = EEGTrialsDataset(x_tr, y_tr, s_tr, uid=uid_tr)

    val_subject = fold.get("val_subject")
    if val_subject is None:
        val_ds = None
    else:
        x_va, y_va, s_va, uid_va = subset_by_subjects(X, y, subj, [val_subject])
        val_ds = EEGTrialsDataset(x_va, y_va, s_va, uid=uid_va)

    x_te, y_te, s_te, uid_te = subset_by_subjects(X, y, subj, [fold["test_subject"]])
    test_ds = EEGTrialsDataset(x_te, y_te, s_te, uid=uid_te)

    return {"train": train_ds, "val": val_ds, "test": test_ds}
