"""
Self-check script:
  - Loads a small subset of BCI-IV-2a (from MOABB cache)
  - Runs a forward pass and 1 training epoch (no accuracy expectations)

Usage:
  python -m eeg_ctm.self_check --config configs/bciiv2a_loso.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_ctm.augment.sr import ClasswiseMemoryBank, SRAugmentConfig, SegmentationRecombinationAugmenter
from eeg_ctm.data.bciiv2a import BCIIV2aClasses, BCIIV2aWindow, EEGTrialsDataset, subset_by_subjects
from eeg_ctm.models.ctm_core import CTMCoreConfig
from eeg_ctm.models.eeg_ctm_model import EEGCTM, EEGCTMConfig
from eeg_ctm.models.losses import TickLossConfig
from eeg_ctm.models.pairs import PairBankConfig, load_or_create_pairbank
from eeg_ctm.models.adversarial import AdvConfig
from eeg_ctm.models.supcon import SupConConfig
from eeg_ctm.models.tokenizer import TokenizerV1Config
from eeg_ctm.training import InjectionConfig, train_one_epoch_with_constraints
from eeg_ctm.utils.config import deep_update, load_config_file
from eeg_ctm.utils.seed import seed_everything


def _standardize_trials_np(X: np.ndarray, *, mode: str, eps: float) -> np.ndarray:
    if mode == "zscore":
        mu = X.mean(axis=-1, keepdims=True)
        sigma = X.std(axis=-1, keepdims=True)
        return (X - mu) / (sigma + eps)
    if mode == "robust":
        med = np.median(X, axis=-1, keepdims=True)
        mad = np.median(np.abs(X - med), axis=-1, keepdims=True)
        sigma = 1.4826 * mad
        return (X - med) / (sigma + eps)
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/bciiv2a_loso.yaml")
    ap.add_argument("--subjects", type=str, default="1,2")
    ap.add_argument("--trials_per_subject", type=int, default=8)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    cfg = load_config_file(args.config)
    # Override for fast self-check.
    deep_update(
        cfg,
        {
            "exp_name": "self_check",
            "train": {"epochs": 1, "batch_size": 8, "num_workers": 0, "grad_clip": 1.0, "amp": False},
            "early_stop": {"enabled": False},
            "optional": {"supcon": {"enabled": False}, "adversarial": {"enabled": False}, "sampler": {"type": "none"}},
            "augment": {"injection": {"mode": "concat"}},
        },
    )

    seed_everything(int(cfg.get("seed", 0)), deterministic=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    subjects = [int(s) for s in str(args.subjects).split(",") if s.strip()]
    window = BCIIV2aWindow(**cfg.get("data", {}).get("window", asdict(BCIIV2aWindow())))

    from eeg_ctm.data.bciiv2a import load_bciiv2a_moabb

    X, y, subj = load_bciiv2a_moabb(subjects, window=window, classes=BCIIV2aClasses(), resample_sfreq=None)
    std_cfg = cfg.get("data", {}).get("standardize", {"mode": "zscore", "eps": 1e-6})
    X = _standardize_trials_np(X, mode=std_cfg.get("mode", "zscore"), eps=float(std_cfg.get("eps", 1e-6))).astype(np.float32)

    # Keep a tiny subset per subject.
    keep = []
    for s in subjects:
        idx = np.where(subj == s)[0][: int(args.trials_per_subject)]
        keep.append(idx)
    keep_idx = np.concatenate(keep, axis=0)
    X = X[keep_idx]
    y = y[keep_idx]
    subj = subj[keep_idx]

    # Simple split: train on first subject, test on second.
    train_subjects = [subjects[0]]
    test_subject = subjects[-1]

    x_tr, y_tr, s_tr, uid_tr = subset_by_subjects(X, y, subj, train_subjects)
    x_te, y_te, s_te, uid_te = subset_by_subjects(X, y, subj, [test_subject])

    train_ds = EEGTrialsDataset(x_tr, y_tr, s_tr, uid=uid_tr)
    test_ds = EEGTrialsDataset(x_te, y_te, s_te, uid=uid_te)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0)

    exp_dir = Path(cfg.get("runs_dir", "runs")) / str(cfg.get("exp_name", "self_check"))
    default_pair = PairBankConfig()
    pair_cfg = PairBankConfig(
        **{k: cfg.get("pairs", {}).get(k, getattr(default_pair, k)) for k in ("D", "K_action", "K_out", "n_self", "seed")}
    )
    pair_bank = load_or_create_pairbank(pair_cfg, cache_path=exp_dir / "pairs.pt")

    model_cfg = EEGCTMConfig(
        tokenizer=TokenizerV1Config(**cfg.get("tokenizer", {})),
        ctm=CTMCoreConfig(**cfg.get("ctm", {})),
    )
    model = EEGCTM(model_cfg, pair_bank=pair_bank).to(device)

    # Forward sanity check.
    batch = next(iter(train_loader))
    logits_ticks, certainty, z_ticks = model(batch["x"].to(device))
    assert logits_ticks.ndim == 3 and certainty.ndim == 2 and z_ticks.ndim == 3

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 123)

    # Train 1 epoch.
    sr_cfg = SRAugmentConfig(**cfg.get("augment", {}).get("sr", asdict(SRAugmentConfig())))
    inj_cfg = InjectionConfig(**cfg.get("augment", {}).get("injection", asdict(InjectionConfig())))
    bank = ClasswiseMemoryBank(num_classes=int(cfg.get("ctm", {}).get("num_classes", 4)))
    for i in range(len(train_ds)):
        sample = train_ds[i]
        bank.add(sample["x"].cpu(), int(sample["y"].item()), int(sample["uid"].item()))
    augmenter = SegmentationRecombinationAugmenter(sr_cfg, num_classes=int(cfg.get("ctm", {}).get("num_classes", 4)))

    train_one_epoch_with_constraints(
        model,
        train_loader,
        device=device,
        optimizer=optimizer,
        tick_loss_cfg=TickLossConfig(**cfg.get("loss", {}).get("tick_loss", asdict(TickLossConfig()))),
        augmenter=augmenter,
        bank=bank,
        injection=inj_cfg,
        rng=rng,
        grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
        rep_mode="certainty_weighted",
        rep_cw_alpha=5.0,
        supcon_cfg=SupConConfig(enabled=False),
        proj_head=None,
        adv_cfg=AdvConfig(enabled=False),
        subject_head=None,
        subject_id_map=None,
        global_step=0,
        total_steps=max(1, len(train_loader)),
    )

    # One eval pass (no metric expectations).
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            _ = model(batch["x"].to(device))
            break

    print("Self-check OK: forward + 1 epoch completed.")


if __name__ == "__main__":
    main()
