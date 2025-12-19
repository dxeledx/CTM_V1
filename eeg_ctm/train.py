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
from eeg_ctm.augment.token_fusion import TokenFusionConfig, build_token_fuser
from eeg_ctm.adapt.fewshot import FewShotConfig, run_few_shot
from eeg_ctm.data.bciiv2a import (
    BCIIV2aClasses,
    BCIIV2aWindow,
    EEGTrialsDataset,
    FilterBankConfig,
    make_loso_splits,
    subset_by_subjects,
)
from eeg_ctm.data.samplers import SamplerConfig, SubjectClassBalancedBatchSampler
from eeg_ctm.data.splits import WithinSubjectValSplitConfig, split_within_subjects
from eeg_ctm.eval import EvalConfig, evaluate
from eeg_ctm.models.aggregation import CertaintyWeightedConfig
from eeg_ctm.models.adversarial import AdvConfig, SubjectHead
from eeg_ctm.models.ctm_core import CTMCoreConfig
from eeg_ctm.models.eeg_ctm_model import EEGCTM, EEGCTMConfig
from eeg_ctm.models.losses import PoolCELossConfig, TickLossConfig
from eeg_ctm.models.pairs import PairBankConfig, load_or_create_pairbank
from eeg_ctm.models.supcon import ProjectionHead, SupConConfig
from eeg_ctm.models.tokenizer import TokenizerV1Config
from eeg_ctm.models.wdro import WDROConfig
from eeg_ctm.training import InjectionConfig, TrainConfig, train_one_epoch_with_constraints
from eeg_ctm.utils.config import deep_update, load_config_file
from eeg_ctm.utils.logging import setup_logger
from eeg_ctm.utils.seed import seed_everything


def _default_cfg() -> dict[str, Any]:
    return {
        "exp_name": "bciiv2a_loso_ctm_v1",
        "overwrite": False,
        "seed": 0,
        "device": "auto",  # auto|cpu|cuda
        "deterministic": True,
        "runs_dir": "runs",
        "data": {
            "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            # Use all sessions by default. MOABB session naming differs across versions
            # (e.g. "0train"/"1test" or "session_T"/"session_E").
            "sessions": None,
            "filterbank": asdict(FilterBankConfig()),
            "window": asdict(BCIIV2aWindow()),
            "resample_sfreq": None,
            "standardize": {"mode": "zscore", "eps": 1e-6},
        },
        "split": {
            # "next"|"fixed"|"none"|"within_subject"
            "val_strategy": "within_subject",
            "fixed_val_subject": None,
            # within_subject settings (CTNet-style, non-transductive)
            "within_subject_val_fraction": 0.3,
            "within_subject_stratify_by_class": True,
        },
        "augment": {
            "sr": asdict(SRAugmentConfig()),
            "injection": asdict(InjectionConfig()),
        },
        "tokenizer": asdict(TokenizerV1Config()),
        "ctm": asdict(CTMCoreConfig()),
        "pairs": {**asdict(PairBankConfig()), "cache_name": "pairs.pt"},
        "loss": {"tick_loss": asdict(TickLossConfig()), "pool_ce": asdict(PoolCELossConfig())},
        "readout": {"mode": "certainty_weighted", "certainty_weighted": asdict(CertaintyWeightedConfig())},
        "wdro": asdict(WDROConfig()),
        "opt": {"lr": 3e-4, "weight_decay": 1e-2},
        "train": asdict(TrainConfig()),
        "early_stop": {"enabled": True, "patience": 20, "metric": "kappa"},
        "optional": {
            "sampler": asdict(SamplerConfig()),
            "supcon": asdict(SupConConfig()),
            "adversarial": asdict(AdvConfig()),
            "rep_agg": {"mode": "certainty_weighted", "certainty_weighted": asdict(CertaintyWeightedConfig())},
        },
        # Optional test-time few-shot adaptation on the held-out subject.
        "few_shot": asdict(FewShotConfig()),
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


def _meta_breakdown_table(meta_df) -> str:
    """
    Format subject×session×run counts for logging.
    """
    # meta_df is a pandas DataFrame from MOABB.
    g = meta_df.groupby(["subject", "session", "run"]).size().reset_index(name="n")
    pivot = g.pivot_table(index=["subject", "session"], columns="run", values="n", fill_value=0, aggfunc="sum")
    # Stable column order: run_0..run_5 if present.
    cols = sorted(pivot.columns.tolist())
    pivot = pivot.reindex(cols, axis=1)
    pivot["total"] = pivot.sum(axis=1)
    return pivot.to_string()


def _log_split_counts(logger, name: str, dataset: EEGTrialsDataset | None, meta_all) -> dict[str, Any]:
    if dataset is None:
        logger.info(f"{name}: None")
        return {f"{name}_n": 0}

    n = len(dataset)
    uid = dataset.uid.detach().cpu().numpy().astype(int)
    meta_sub = meta_all.iloc[uid]
    subj_list = sorted({int(s) for s in meta_sub["subject"].unique().tolist()})
    sess_list = sorted({str(s) for s in meta_sub["session"].unique().tolist()})
    logger.info(f"{name}: n={n} subjects={subj_list} sessions={sess_list}")
    logger.info(f"{name} breakdown (subject×session×run):\n{_meta_breakdown_table(meta_sub)}")
    return {f"{name}_n": n, f"{name}_subjects": subj_list, f"{name}_sessions": sess_list}


def _is_train_session_name(name: str) -> bool:
    n = str(name).strip().lower()
    if n in {"0", "train", "training", "t", "session_t", "0train"}:
        return True
    if "train" in n:
        return True
    # Be careful: "test" ends with "t", so exclude it.
    return n.endswith("t") and ("test" not in n) and ("eval" not in n)


def _is_test_session_name(name: str) -> bool:
    n = str(name).strip().lower()
    if n in {"1", "test", "testing", "eval", "evaluation", "e", "session_e", "1test"}:
        return True
    return ("test" in n) or ("eval" in n) or n.endswith("e")


def _resolve_sessions_to_keep(*, available: list[str], requested: Any) -> list[str]:
    """
    Resolve config `data.sessions` into concrete session names present in MOABB `meta['session']`.

    Supports common aliases across MOABB versions:
      - train: "session_T"/"T"/"train"/"0train"/"0"
      - test:  "session_E"/"E"/"test"/"1test"/"1"
      - all:   null (handled upstream) or "all"/"both" to keep everything
    """
    if requested is None:
        return list(available)
    if isinstance(requested, (str, int)):
        req_list = [requested]
    else:
        req_list = list(requested)

    available_list = [str(s) for s in available]
    available_set = set(available_list)

    req_str = [str(s) for s in req_list]
    req_norm = [s.strip().lower() for s in req_str]
    req_set = set(req_str)

    if any(s in {"all", "both", "*"} for s in req_norm):
        return list(available_list)

    if req_set.issubset(available_set):
        # Preserve the order in `available_list` for stability.
        return [s for s in available_list if s in req_set]

    want_train = any(_is_train_session_name(s) for s in req_norm)
    want_test = any(_is_test_session_name(s) for s in req_norm)
    if not (want_train or want_test):
        raise ValueError(
            f"data.sessions={req_str} does not match available sessions={sorted(available_set)}. "
            "Set data.sessions: null to use all sessions."
        )

    # If the user specified something we don't understand (neither alias nor exact), fail loudly.
    for raw, norm in zip(req_str, req_norm):
        if raw in available_set:
            continue
        if norm in {"all", "both", "*"}:
            continue
        if _is_train_session_name(norm) or _is_test_session_name(norm):
            continue
        raise ValueError(
            f"data.sessions contains unknown value {raw!r}. "
            f"Available sessions={sorted(available_set)}. Use data.sessions: null to keep all."
        )

    selected: list[str] = []
    for s in available_list:
        if want_train and _is_train_session_name(s):
            selected.append(s)
        if want_test and _is_test_session_name(s):
            selected.append(s)

    # De-duplicate while preserving order.
    selected = list(dict.fromkeys(selected))
    if len(selected) == 0:
        raise ValueError(
            f"data.sessions={req_str} resolved to empty on available sessions={sorted(available_set)}. "
            "Set data.sessions: null to use all sessions."
        )
    return selected


def _resolve_exp_dir(*, runs_dir: str | Path, exp_name: str, overwrite: bool) -> Path:
    """
    Avoid overwriting previous results by default.

    If `runs/<exp_name>/` already exists and is non-empty, we create:
      runs/<exp_name>__run01/
      runs/<exp_name>__run02/
      ...
    """
    runs_dir = Path(runs_dir)
    base = runs_dir / exp_name
    if overwrite or (not base.exists()):
        return base
    try:
        is_empty = not any(base.iterdir())
    except Exception:
        is_empty = False
    if is_empty:
        return base

    i = 1
    while True:
        cand = runs_dir / f"{exp_name}__run{i:02d}"
        if not cand.exists():
            return cand
        i += 1


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

    exp_dir = _resolve_exp_dir(
        runs_dir=str(cfg["runs_dir"]),
        exp_name=str(cfg["exp_name"]),
        overwrite=bool(cfg.get("overwrite", False)),
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(exp_dir)
    _save_json(exp_dir / "config_resolved.json", cfg)
    print(f"[eeg_ctm] Output directory: {exp_dir}")

    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))
    device = _pick_device(str(cfg["device"]))

    # Pair sampling: fixed + cached per experiment.
    pairs_cfg = PairBankConfig(**{k: cfg["pairs"][k] for k in ("D", "K_action", "K_out", "n_self", "seed")})
    pair_cache = exp_dir / str(cfg["pairs"].get("cache_name", "pairs.pt"))
    pair_bank = load_or_create_pairbank(pairs_cfg, cache_path=pair_cache)

    subjects = cfg["data"]["subjects"]
    val_strategy = str(cfg["split"]["val_strategy"])
    folds = make_loso_splits(subjects, val_strategy=val_strategy, fixed_val_subject=cfg["split"]["fixed_val_subject"])

    # Load full dataset once (MOABB uses local caches; no forced downloads).
    classes = BCIIV2aClasses()
    window = BCIIV2aWindow(**cfg["data"]["window"])
    fb_cfg = FilterBankConfig(**cfg["data"].get("filterbank", {}))
    from eeg_ctm.data.bciiv2a import load_bciiv2a_moabb

    X_all, y_all, subj_all, meta_all = load_bciiv2a_moabb(
        subjects,
        window=window,
        classes=classes,
        resample_sfreq=cfg["data"]["resample_sfreq"],
        filterbank=fb_cfg,
        return_meta=True,
    )
    # Optional session filtering (default: use both sessions).
    sessions_keep = cfg["data"].get("sessions")
    if sessions_keep is not None:
        available_sessions = sorted({str(s) for s in meta_all["session"].unique().tolist()})
        resolved_sessions = _resolve_sessions_to_keep(available=available_sessions, requested=sessions_keep)
        mask = meta_all["session"].isin(resolved_sessions).to_numpy()
        if mask.sum() == 0:
            raise ValueError(
                f"Session filter produced 0 trials. requested={sessions_keep!r} "
                f"resolved={resolved_sessions!r} available={available_sessions!r}"
            )
        meta_all = meta_all.iloc[mask].reset_index(drop=True)
        X_all = X_all[mask]
        y_all = y_all[mask]
        subj_all = subj_all[mask]

    # Auto-resolve tokenizer input shape from the loaded data (useful for filterbank).
    C_eff = int(X_all.shape[1])
    T_eff = int(X_all.shape[2])
    tok_cfg = dict(cfg.get("tokenizer", {}))
    if int(tok_cfg.get("C", C_eff)) != C_eff or int(tok_cfg.get("T", T_eff)) != T_eff:
        print(f"[eeg_ctm] Tokenizer input shape override: C {tok_cfg.get('C')} -> {C_eff}, T {tok_cfg.get('T')} -> {T_eff}")
    tok_cfg["C"] = C_eff
    tok_cfg["T"] = T_eff
    cfg["tokenizer"] = tok_cfg
    _save_json(exp_dir / "config_resolved.json", cfg)

    std_cfg = cfg["data"]["standardize"]
    X_all = _standardize_trials_np(X_all, mode=std_cfg["mode"], eps=float(std_cfg.get("eps", 1e-6))).astype(np.float32)

    fold_results = []

    for fold in folds:
        fold_name = f"fold_{fold['test_subject']}"
        fold_dir = exp_dir / fold_name
        logger = setup_logger(fold_dir, name=f"eeg_ctm.{fold_name}")
        logger.info(f"Device: {device}")
        logger.info(
            f"Fold: test={fold['test_subject']} val={fold['val_subject']} train={fold['train_subjects']} (val_strategy={val_strategy})"
        )

        x_tr_all, y_tr_all, s_tr_all, uid_tr_all = subset_by_subjects(X_all, y_all, subj_all, fold["train_subjects"])
        if val_strategy == "within_subject":
            vs_cfg = WithinSubjectValSplitConfig(
                val_fraction=float(cfg["split"].get("within_subject_val_fraction", 0.3)),
                seed=int(cfg["seed"]) + 1000 * int(fold["test_subject"]),
                stratify_by_class=bool(cfg["split"].get("within_subject_stratify_by_class", True)),
            )
            tr_idx, va_idx = split_within_subjects(y_tr_all, s_tr_all, cfg=vs_cfg)
            train_ds = EEGTrialsDataset(x_tr_all[tr_idx], y_tr_all[tr_idx], s_tr_all[tr_idx], uid=uid_tr_all[tr_idx])
            val_ds = EEGTrialsDataset(x_tr_all[va_idx], y_tr_all[va_idx], s_tr_all[va_idx], uid=uid_tr_all[va_idx])
        else:
            train_ds = EEGTrialsDataset(x_tr_all, y_tr_all, s_tr_all, uid=uid_tr_all)
            if fold["val_subject"] is None:
                val_ds = None
            else:
                x_va, y_va, s_va, uid_va = subset_by_subjects(X_all, y_all, subj_all, [fold["val_subject"]])
                val_ds = EEGTrialsDataset(x_va, y_va, s_va, uid=uid_va)

        x_te, y_te, s_te, uid_te = subset_by_subjects(X_all, y_all, subj_all, [fold["test_subject"]])
        test_ds = EEGTrialsDataset(x_te, y_te, s_te, uid=uid_te)

        # Sanity logging: exact sample counts and subject×session×run breakdown.
        split_info = {}
        split_info.update(_log_split_counts(logger, "train", train_ds, meta_all))
        split_info.update(_log_split_counts(logger, "val", val_ds, meta_all))
        split_info.update(_log_split_counts(logger, "test", test_ds, meta_all))

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
        wdro_dict = dict(cfg.get("wdro", {}))
        # Backward compat: `mix_clean` was renamed to `mix_robust` (robust weight).
        if "mix_clean" in wdro_dict and "mix_robust" not in wdro_dict:
            wdro_dict["mix_robust"] = wdro_dict.pop("mix_clean")
        wdro_cfg = WDROConfig(**wdro_dict)
        logger.info(
            f"Augment: sr_enabled={sr_cfg.enabled} injection_mode={injection_cfg.mode} | "
            f"Optional: supcon_enabled={supcon_cfg.enabled} adversarial_enabled={adv_cfg.enabled} "
            f"wdro_enabled={wdro_cfg.enabled} sampler={sampler_cfg.type}"
        )

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

        token_fuser = None
        if str(injection_cfg.mode) in ("adain", "film", "xattn"):
            tf_cfg = TokenFusionConfig(
                mode=str(injection_cfg.mode),
                film_hidden=int(injection_cfg.film_hidden),
                xattn_heads=int(injection_cfg.xattn_heads),
                xattn_dropout=float(injection_cfg.xattn_dropout),
                xattn_layernorm=bool(injection_cfg.xattn_layernorm),
            )
            token_fuser = build_token_fuser(tf_cfg, d_model=int(tokenizer_cfg.d_kv)).to(device)

        # Optional heads
        rep_agg_cfg = cfg["optional"]["rep_agg"]
        rep_mode = str(rep_agg_cfg.get("mode", "certainty_weighted"))
        rep_cw_alpha = float(rep_agg_cfg.get("certainty_weighted", {}).get("alpha", 5.0))
        rep_cw_detach = bool(rep_agg_cfg.get("certainty_weighted", {}).get("detach_certainty", False))

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
        if token_fuser is not None:
            optim_params += list(token_fuser.parameters())
        if proj_head is not None:
            optim_params += list(proj_head.parameters())
        if subject_head is not None:
            optim_params += list(subject_head.parameters())
        optimizer = torch.optim.AdamW(optim_params, lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"]))

        tick_loss_cfg = TickLossConfig(**cfg["loss"]["tick_loss"])
        pool_ce_cfg = PoolCELossConfig(**cfg.get("loss", {}).get("pool_ce", {}))
        eval_cfg = EvalConfig(
            readout=str(cfg["readout"]["mode"]),
            certainty_weighted_alpha=float(cfg["readout"]["certainty_weighted"]["alpha"]),
            certainty_weighted_detach_certainty=bool(cfg["readout"]["certainty_weighted"].get("detach_certainty", False)),
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
                pool_ce_cfg=pool_ce_cfg,
                augmenter=augmenter,
                bank=bank,
                injection=injection_cfg,
                token_fuser=token_fuser,
                rng=rng,
                grad_clip=float(train_cfg.grad_clip) if train_cfg.grad_clip is not None else None,
                rep_mode=rep_mode,
                rep_cw_alpha=rep_cw_alpha,
                rep_cw_detach=rep_cw_detach,
                acc_readout=str(eval_cfg.readout),
                acc_cw_alpha=float(eval_cfg.certainty_weighted_alpha),
                acc_cw_detach=bool(eval_cfg.certainty_weighted_detach_certainty),
                wdro_cfg=wdro_cfg,
                epoch=epoch,
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

        few_shot_cfg = FewShotConfig(**cfg.get("few_shot", {}))
        few_shot_out: dict[str, Any] = {"enabled": False}
        if few_shot_cfg.enabled:
            # Few-shot adaptation happens strictly after model selection (no leakage into training).
            meta_test = meta_all.iloc[uid_te].reset_index(drop=True)
            few_shot_out = run_few_shot(
                model,
                test_ds=test_ds,
                meta_sub=meta_test,
                device=device,
                num_classes=int(cfg["ctm"]["num_classes"]),
                tick_loss_cfg=tick_loss_cfg,
                eval_cfg=eval_cfg,
                cfg=few_shot_cfg,
                fold_seed=int(cfg["seed"]) + 1000 * int(fold["test_subject"]),
            )
            for k_str, res in few_shot_out.get("results", {}).items():
                m = res["mean"]
                s = res["std"]
                logger.info(
                    f"Few-shot K={k_str} (mean±std over {res['n_resamples']} resamples): "
                    f"acc={m['accuracy']:.4f}±{s['accuracy']:.4f} "
                    f"kappa={m['kappa']:.4f}±{s['kappa']:.4f} "
                    f"macro_f1={m['macro_f1']:.4f}±{s['macro_f1']:.4f}"
                )

        fold_metrics = {
            "fold": fold_name,
            "train_subjects": fold["train_subjects"],
            "val_subject": fold["val_subject"],
            "test_subject": fold["test_subject"],
            "val_strategy": val_strategy,
            "data_sessions": sessions_keep if sessions_keep is not None else "all",
            "split_counts": split_info,
            "best_epoch": best_epoch,
            "best_val": best_val,
            "test": test_metrics.to_dict(),
            "few_shot": few_shot_out,
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

    # Optional: few-shot summary (mean±std over folds, using each fold's mean over resamples).
    few_shot_summary = {}
    few_shot_folds = [fr.get("few_shot", {}) for fr in fold_results if bool(fr.get("few_shot", {}).get("enabled", False))]
    if len(few_shot_folds) > 0:
        # Collect all K values present in every fold.
        k_sets = [set(fs.get("results", {}).keys()) for fs in few_shot_folds]
        common_ks = sorted(set.intersection(*k_sets)) if k_sets else []
        for k_str in common_ks:
            per_fold = [fs["results"][k_str]["mean"] for fs in few_shot_folds if k_str in fs.get("results", {})]
            if len(per_fold) == 0:
                continue
            few_shot_summary[k_str] = {}
            for metric in keys:
                vals = np.array([float(d[metric]) for d in per_fold], dtype=np.float64)
                few_shot_summary[k_str][metric] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1 if vals.size >= 2 else 0)),
                }

    out = {
        "exp_name": cfg["exp_name"],
        "n_folds": len(fold_results),
        "test_summary": summary,
        "few_shot_summary": few_shot_summary,
        "folds": fold_results,
    }
    _save_json(exp_dir / "metrics_summary.json", out)


if __name__ == "__main__":
    main()
