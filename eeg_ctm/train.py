"""
CLI entry: `python -m eeg_ctm.train --config configs/bciiv2a_loso.yaml`

Implements LOSO training/evaluation on BCI Competition IV-2a using:
  - Trial-wise standardization (no leakage)
  - Train-only S&R augmentation (no leakage donor pool)
  - Tokenizer v1 + CTM core with multi-tick outputs
  - Per-fold metrics (accuracy/kappa/macro-F1) + mean±std summary

Design doc mapping:
  - design.md "模块链路（建议 v0 先最小闭环）"
  - design.md "LOSO 场景下不会踩坑的实现细则"
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_ctm.augment.sr import ClasswiseMemoryBank, SRAugmentConfig, SegmentationRecombinationAugmenter
from eeg_ctm.data.bciiv2a import BCIIV2aClasses, BCIIV2aWindow, EEGTrialsDataset, make_loso_splits, subset_by_subjects
from eeg_ctm.data.samplers import SamplerConfig, SubjectClassBalancedBatchSampler
from eeg_ctm.eval import EvalConfig, evaluate
from eeg_ctm.models.aggregation import CertaintyWeightedConfig
from eeg_ctm.models.adversarial import AdvConfig, SubjectHead
from eeg_ctm.models.ctm_core import CTMCoreConfig
from eeg_ctm.models.eeg_ctm_model import EEGCTM, EEGCTMConfig
from eeg_ctm.models.losses import TickLossConfig
from eeg_ctm.models.pairs import PairBankConfig, load_or_create_pairbank
from eeg_ctm.models.supcon import ProjectionHead, SupConConfig
from eeg_ctm.models.tokenizer import TokenizerV1Config
from eeg_ctm.training import InjectionConfig, TrainConfig, train_one_epoch_with_constraints
from eeg_ctm.utils.config import deep_update, load_config_file
from eeg_ctm.utils.logging import setup_logger
from eeg_ctm.utils.seed import seed_everything


def _default_cfg() -> dict[str, Any]:
    return {
        "exp_name": "bciiv2a_loso_ctm_v1",
        "seed": 0,
        "device": "auto",  # auto|cpu|cuda
        "deterministic": True,
        "runs_dir": "runs",
        "data": {
            "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "window": asdict(BCIIV2aWindow()),
            "resample_sfreq": None,
            "standardize": {"mode": "zscore", "eps": 1e-6},
        },
        "split": {"val_strategy": "next", "fixed_val_subject": None},
        "augment": {
            "sr": asdict(SRAugmentConfig()),
            "injection": asdict(InjectionConfig()),
        },
        "tokenizer": asdict(TokenizerV1Config()),
        "ctm": asdict(CTMCoreConfig()),
        "pairs": {**asdict(PairBankConfig()), "cache_name": "pairs.pt"},
        "loss": {"tick_loss": asdict(TickLossConfig())},
        "readout": {"mode": "certainty_weighted", "certainty_weighted": asdict(CertaintyWeightedConfig())},
        "opt": {"lr": 3e-4, "weight_decay": 1e-2},
        "train": asdict(TrainConfig()),
        "early_stop": {"enabled": True, "patience": 20, "metric": "kappa"},
        "optional": {
            "sampler": asdict(SamplerConfig()),
            "supcon": asdict(SupConConfig()),
            "adversarial": asdict(AdvConfig()),
            "rep_agg": {"mode": "certainty_weighted", "certainty_weighted": asdict(CertaintyWeightedConfig())},
        },
    }


def _pick_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _standardize_trials_np(X: np.ndarray, *, mode: str, eps: float) -> np.ndarray:
    # Design.md: trial-wise standardization, per-channel over time.
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


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main() -> None:
    # Silence verbose MOABB warnings (design requirement: ignore, do not print).
    warnings.filterwarnings("ignore", category=UserWarning, message=r"warnEpochs.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=r"The current default of copy=False.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"moabb\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"moabb\..*")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = _default_cfg()
    cfg_user = load_config_file(args.config)
    deep_update(cfg, cfg_user)

    exp_dir = Path(cfg["runs_dir"]) / str(cfg["exp_name"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    _save_json(exp_dir / "config_resolved.json", cfg)

    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))
    device = _pick_device(str(cfg["device"]))

    # Pair sampling: fixed + cached per experiment.
    pairs_cfg = PairBankConfig(**{k: cfg["pairs"][k] for k in ("D", "K_action", "K_out", "n_self", "seed")})
    pair_cache = exp_dir / str(cfg["pairs"].get("cache_name", "pairs.pt"))
    pair_bank = load_or_create_pairbank(pairs_cfg, cache_path=pair_cache)

    subjects = cfg["data"]["subjects"]
    folds = make_loso_splits(subjects, val_strategy=cfg["split"]["val_strategy"], fixed_val_subject=cfg["split"]["fixed_val_subject"])

    # Load full dataset once (MOABB uses local caches; no forced downloads).
    classes = BCIIV2aClasses()
    window = BCIIV2aWindow(**cfg["data"]["window"])
    from eeg_ctm.data.bciiv2a import load_bciiv2a_moabb

    X_all, y_all, subj_all = load_bciiv2a_moabb(
        subjects,
        window=window,
        classes=classes,
        resample_sfreq=cfg["data"]["resample_sfreq"],
    )
    std_cfg = cfg["data"]["standardize"]
    X_all = _standardize_trials_np(X_all, mode=std_cfg["mode"], eps=float(std_cfg.get("eps", 1e-6))).astype(np.float32)

    fold_results = []

    for fold in folds:
        fold_name = f"fold_{fold['test_subject']}"
        fold_dir = exp_dir / fold_name
        logger = setup_logger(fold_dir, name=f"eeg_ctm.{fold_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Fold: test={fold['test_subject']} val={fold['val_subject']} train={fold['train_subjects']}")

        x_tr, y_tr, s_tr, uid_tr = subset_by_subjects(X_all, y_all, subj_all, fold["train_subjects"])
        train_ds = EEGTrialsDataset(x_tr, y_tr, s_tr, uid=uid_tr)

        if fold["val_subject"] is None:
            val_ds = None
        else:
            x_va, y_va, s_va, uid_va = subset_by_subjects(X_all, y_all, subj_all, [fold["val_subject"]])
            val_ds = EEGTrialsDataset(x_va, y_va, s_va, uid=uid_va)

        x_te, y_te, s_te, uid_te = subset_by_subjects(X_all, y_all, subj_all, [fold["test_subject"]])
        test_ds = EEGTrialsDataset(x_te, y_te, s_te, uid=uid_te)

        train_cfg = TrainConfig(**cfg["train"])
        sampler_cfg = SamplerConfig(**cfg["optional"]["sampler"])
        if sampler_cfg.type == "subject_class_balanced":
            batch_sampler = SubjectClassBalancedBatchSampler(
                subjects=train_ds.subject.cpu().numpy(),
                labels=train_ds.y.cpu().numpy(),
                subjects_per_batch=int(sampler_cfg.subjects_per_batch),
                samples_per_class=int(sampler_cfg.samples_per_class),
                num_classes=int(cfg["ctm"]["num_classes"]),
                seed=int(cfg["seed"]) + 100 * int(fold["test_subject"]),
                drop_last=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=int(train_cfg.num_workers),
                pin_memory=(device.type == "cuda"),
            )
        else:
            batch_sampler = None
            dl_gen = torch.Generator().manual_seed(int(cfg["seed"]) + 10 * int(fold["test_subject"]))
            train_loader = DataLoader(
                train_ds,
                batch_size=int(train_cfg.batch_size),
                shuffle=True,
                num_workers=int(train_cfg.num_workers),
                pin_memory=(device.type == "cuda"),
                generator=dl_gen,
            )
        val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=int(train_cfg.batch_size), shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=int(train_cfg.batch_size), shuffle=False, num_workers=0)

        # Augmentation donor pool: training subjects only (strictly excludes val/test).
        sr_cfg = SRAugmentConfig(**cfg["augment"]["sr"])
        injection_cfg = InjectionConfig(**cfg["augment"]["injection"])
        supcon_cfg = SupConConfig(**cfg["optional"]["supcon"])
        adv_cfg = AdvConfig(**cfg["optional"]["adversarial"])

        need_aug = sr_cfg.enabled and (injection_cfg.mode != "none" or supcon_cfg.enabled)
        if need_aug:
            bank = ClasswiseMemoryBank(num_classes=int(cfg["ctm"]["num_classes"]))
            for i in range(len(train_ds)):
                sample = train_ds[i]
                bank.add(sample["x"].cpu(), int(sample["y"].item()), int(sample["uid"].item()))
            augmenter = SegmentationRecombinationAugmenter(sr_cfg, num_classes=int(cfg["ctm"]["num_classes"]))
        else:
            bank = None
            augmenter = None

        # Model
        tokenizer_cfg = TokenizerV1Config(**cfg["tokenizer"])
        ctm_cfg = CTMCoreConfig(**cfg["ctm"])
        if int(ctm_cfg.D) != int(pairs_cfg.D):
            raise ValueError(f"Config mismatch: ctm.D={ctm_cfg.D} vs pairs.D={pairs_cfg.D}")
        model_cfg = EEGCTMConfig(tokenizer=tokenizer_cfg, ctm=ctm_cfg)
        model = EEGCTM(model_cfg, pair_bank=pair_bank).to(device)

        # Optional heads
        rep_agg_cfg = cfg["optional"]["rep_agg"]
        rep_mode = str(rep_agg_cfg.get("mode", "certainty_weighted"))
        rep_cw_alpha = float(rep_agg_cfg.get("certainty_weighted", {}).get("alpha", 5.0))

        proj_head = None
        if supcon_cfg.enabled:
            proj_head = ProjectionHead(int(ctm_cfg.D), int(supcon_cfg.proj_dim), hidden=int(supcon_cfg.proj_hidden)).to(device)

        subject_head = None
        subject_id_map = None
        if adv_cfg.enabled:
            train_subjects_sorted = sorted(int(s) for s in fold["train_subjects"])
            subject_id_map = {sid: i for i, sid in enumerate(train_subjects_sorted)}
            subject_head = SubjectHead(int(ctm_cfg.D), n_subjects=len(train_subjects_sorted), hidden=int(adv_cfg.head_hidden)).to(device)

        opt_cfg = cfg["opt"]
        optim_params = list(model.parameters())
        if proj_head is not None:
            optim_params += list(proj_head.parameters())
        if subject_head is not None:
            optim_params += list(subject_head.parameters())
        optimizer = torch.optim.AdamW(optim_params, lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"]))

        tick_loss_cfg = TickLossConfig(**cfg["loss"]["tick_loss"])
        eval_cfg = EvalConfig(
            readout=str(cfg["readout"]["mode"]),
            certainty_weighted_alpha=float(cfg["readout"]["certainty_weighted"]["alpha"]),
        )

        best_metric = -1e9
        best_epoch = 0
        best_val = None
        patience = 0
        es_cfg = cfg["early_stop"]
        metric_name = str(es_cfg.get("metric", "kappa"))
        if metric_name not in ("accuracy", "kappa", "macro_f1"):
            raise ValueError(f"early_stop.metric must be one of accuracy|kappa|macro_f1, got {metric_name}")

        history = []
        global_step = 0
        total_steps = int(train_cfg.epochs) * max(1, len(train_loader))

        for epoch in range(1, int(train_cfg.epochs) + 1):
            if batch_sampler is not None:
                batch_sampler.set_epoch(epoch)
            # Deterministic per-epoch RNG for augmentation sampling.
            rng = torch.Generator().manual_seed(int(cfg["seed"]) + 1000 * int(fold["test_subject"]) + epoch)

            train_loss, train_details, global_step = train_one_epoch_with_constraints(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                tick_loss_cfg=tick_loss_cfg,
                augmenter=augmenter,
                bank=bank,
                injection=injection_cfg,
                rng=rng,
                grad_clip=float(train_cfg.grad_clip) if train_cfg.grad_clip is not None else None,
                rep_mode=rep_mode,
                rep_cw_alpha=rep_cw_alpha,
                supcon_cfg=supcon_cfg,
                proj_head=proj_head,
                adv_cfg=adv_cfg,
                subject_head=subject_head,
                subject_id_map=subject_id_map,
                global_step=global_step,
                total_steps=total_steps,
            )

            entry = {"epoch": epoch, "train_loss": train_loss}
            entry.update({k: v for k, v in train_details.items() if k not in entry})

            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device=device, eval_cfg=eval_cfg)
                entry.update({f"val_{k}": v for k, v in val_metrics.to_dict().items()})
                current = float(getattr(val_metrics, metric_name))
                if current > best_metric:
                    best_metric = current
                    best_epoch = epoch
                    best_val = val_metrics.to_dict()
                    patience = 0
                    torch.save({"model": model.state_dict(), "epoch": epoch}, fold_dir / "best.pt")
                else:
                    patience += 1

                if bool(es_cfg.get("enabled", True)) and patience >= int(es_cfg.get("patience", 20)):
                    logger.info(f"Early stop at epoch={epoch} (best_epoch={best_epoch}, best_{metric_name}={best_metric:.4f})")
                    break
            else:
                # No val set: always save last checkpoint.
                torch.save({"model": model.state_dict(), "epoch": epoch}, fold_dir / "last.pt")

            history.append(entry)
            logger.info(
                " | ".join(
                    [
                        f"{k}={float(v):.4f}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
                        for k, v in entry.items()
                    ]
                )
            )

        # Load best checkpoint (val-based) if available; else keep current.
        ckpt_path = fold_dir / "best.pt" if (fold_dir / "best.pt").exists() else (fold_dir / "last.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])

        test_metrics = evaluate(model, test_loader, device=device, eval_cfg=eval_cfg)
        logger.info(f"Test metrics: {test_metrics.to_dict()}")

        fold_metrics = {
            "fold": fold_name,
            "train_subjects": fold["train_subjects"],
            "val_subject": fold["val_subject"],
            "test_subject": fold["test_subject"],
            "best_epoch": best_epoch,
            "best_val": best_val,
            "test": test_metrics.to_dict(),
        }
        _save_json(fold_dir / "metrics.json", fold_metrics)
        _save_json(fold_dir / "history.json", history)

        fold_results.append(fold_metrics)

    # Summary: mean±std over folds (test metrics).
    tests = [fr["test"] for fr in fold_results]
    keys = ["accuracy", "kappa", "macro_f1"]
    summary = {}
    for k in keys:
        vals = np.array([t[k] for t in tests], dtype=np.float64)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

    out = {"exp_name": cfg["exp_name"], "n_folds": len(fold_results), "test_summary": summary, "folds": fold_results}
    _save_json(exp_dir / "metrics_summary.json", out)


if __name__ == "__main__":
    main()
