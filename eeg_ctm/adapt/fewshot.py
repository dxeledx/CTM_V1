"""
Few-shot test-time adaptation (subject-specific) utilities.

Design / theory mapping:
  - design.md: optional extensions (test-time adaptation is *not* part of the strict LOSO baseline)
  - User theory notes (2025-12-18): bilevel view of subject adaptation; head-only is the default setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import torch

from eeg_ctm.eval import EvalConfig, evaluate
from eeg_ctm.models.losses import TickLossConfig, tick_classification_loss


FewShotProtocol = Literal["within_subject", "cross_session"]


@dataclass(frozen=True)
class FewShotConfig:
    enabled: bool = False
    protocol: FewShotProtocol = "within_subject"

    # Session-wise split (recommended for BCI-IV-2a): adapt on "0train", evaluate on "1test".
    # When protocol="within_subject", these are optional filters for sampling support/query.
    support_session: Optional[str] = None
    query_session: Optional[str] = None

    # K-shot per class; total support size is K * num_classes.
    k_shots: tuple[int, ...] = (1, 5, 10, 20)
    n_resamples: int = 5
    seed: int = 0

    # Inner solver: head-only fine-tuning by default.
    trainable_prefixes: tuple[str, ...] = ("ctm.head",)
    lr: float = 1e-2
    weight_decay: float = 0.0
    steps: int = 200
    batch_size: int = 16
    grad_clip: Optional[float] = None


def _match_prefix(name: str, prefix: str) -> bool:
    if name == prefix:
        return True
    if name.startswith(prefix + "."):
        return True
    return False


def snapshot_params_by_prefix(model: torch.nn.Module, prefixes: Sequence[str]) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if any(_match_prefix(name, pref) for pref in prefixes):
            snap[name] = p.detach().clone()
    return snap


def restore_params(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name].to(device=p.device, dtype=p.dtype))


def set_trainable_by_prefix(model: torch.nn.Module, prefixes: Sequence[str]) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        is_trainable = any(_match_prefix(name, pref) for pref in prefixes)
        p.requires_grad = bool(is_trainable)
        if is_trainable:
            trainable.append(p)
    return trainable


def _balanced_sample_support(
    *,
    y: np.ndarray,
    candidate_mask: np.ndarray,
    k: int,
    num_classes: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if y.ndim != 1 or candidate_mask.ndim != 1 or y.shape[0] != candidate_mask.shape[0]:
        raise ValueError("y/mask shape mismatch")
    if k <= 0:
        raise ValueError("k must be > 0")

    support_parts = []
    for c in range(int(num_classes)):
        idx_c = np.where(candidate_mask & (y == c))[0]
        if idx_c.size < k:
            raise ValueError(f"Not enough samples for class={c}: need k={k}, have {idx_c.size}")
        pick = rng.choice(idx_c, size=int(k), replace=False)
        support_parts.append(pick)
    support_idx = np.concatenate(support_parts, axis=0).astype(np.int64)
    rng.shuffle(support_idx)
    return support_idx


def split_support_query(
    *,
    y: np.ndarray,
    sessions: Optional[np.ndarray],
    cfg: FewShotConfig,
    k: int,
    num_classes: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (support_idx, query_idx) as indices into the test-subject arrays.
    """
    y = np.asarray(y, dtype=np.int64)
    N = int(y.shape[0])
    if sessions is None:
        sess = None
    else:
        sess = np.asarray(sessions).astype(str)
        if sess.shape[0] != N:
            raise ValueError("sessions length mismatch")

    if cfg.protocol == "cross_session":
        if sess is None:
            raise ValueError("cross_session protocol requires sessions array")
        if cfg.support_session is None or cfg.query_session is None:
            raise ValueError("cross_session protocol requires support_session and query_session")
        support_mask = sess == str(cfg.support_session)
        query_mask = sess == str(cfg.query_session)
        if support_mask.sum() == 0:
            raise ValueError(f"support_session={cfg.support_session!r} produced 0 samples")
        if query_mask.sum() == 0:
            raise ValueError(f"query_session={cfg.query_session!r} produced 0 samples")
        support_idx = _balanced_sample_support(y=y, candidate_mask=support_mask, k=k, num_classes=num_classes, rng=rng)
        query_idx = np.where(query_mask)[0].astype(np.int64)
        return support_idx, query_idx

    if cfg.protocol == "within_subject":
        candidate_mask = np.ones((N,), dtype=bool)
        if sess is not None and cfg.support_session is not None:
            candidate_mask = candidate_mask & (sess == str(cfg.support_session))
        support_idx = _balanced_sample_support(y=y, candidate_mask=candidate_mask, k=k, num_classes=num_classes, rng=rng)

        query_mask = np.ones((N,), dtype=bool)
        if sess is not None and cfg.query_session is not None:
            query_mask = query_mask & (sess == str(cfg.query_session))
        query_mask[support_idx] = False
        query_idx = np.where(query_mask)[0].astype(np.int64)
        if query_idx.size == 0:
            raise ValueError("query is empty after support selection")
        return support_idx, query_idx

    raise ValueError(cfg.protocol)


def _loader_from_subset(ds, idx: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> torch.utils.data.DataLoader:
    idx = np.asarray(idx, dtype=np.int64)
    subset = torch.utils.data.Subset(ds, idx.tolist())
    g = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.DataLoader(subset, batch_size=int(batch_size), shuffle=bool(shuffle), generator=g, num_workers=0)


def finetune_head_on_support(
    model: torch.nn.Module,
    support_loader,
    *,
    device: torch.device,
    trainable_prefixes: Sequence[str],
    tick_loss_cfg: TickLossConfig,
    lr: float,
    weight_decay: float,
    steps: int,
    grad_clip: Optional[float],
) -> None:
    """
    Head-only (or small subset) adaptation on the support set.

    Recommended default: trainable_prefixes=("ctm.head",) and keep model in eval() mode to avoid
    BN/stat updates and dropout noise under tiny K.
    """
    # Save current flags/mode to avoid leaving the model in a surprising state.
    prev_training = bool(model.training)
    prev_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    try:
        # Freeze everything except the requested prefixes.
        trainable_params = set_trainable_by_prefix(model, trainable_prefixes)
        if len(trainable_params) == 0:
            raise ValueError(f"No trainable parameters matched prefixes={list(trainable_prefixes)}")

        # Use eval() to disable dropout and BN/stat updates under tiny K.
        model.eval()
        optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))

        it = iter(support_loader)
        for _ in range(int(steps)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(support_loader)
                batch = next(it)

            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits_ticks, certainty, _ = model(x)
            loss, _ = tick_classification_loss(logits_ticks, certainty, y, cfg=tick_loss_cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip))
            optimizer.step()
    finally:
        # Restore original requires_grad flags and mode.
        for name, p in model.named_parameters():
            if name in prev_requires_grad:
                p.requires_grad = bool(prev_requires_grad[name])
        model.train(prev_training)


def run_few_shot(
    model: torch.nn.Module,
    *,
    test_ds,
    meta_sub,  # pandas DataFrame subset for this test subject (same order as test_ds uid)
    device: torch.device,
    num_classes: int,
    tick_loss_cfg: TickLossConfig,
    eval_cfg: EvalConfig,
    cfg: FewShotConfig,
    fold_seed: int,
) -> dict:
    """
    Returns nested dict with per-K results (mean/std over resamples + each resample metrics).
    """
    if not cfg.enabled:
        return {"enabled": False}

    # Extract arrays for splitting.
    y = test_ds.y.detach().cpu().numpy().astype(np.int64)
    sessions = None
    if meta_sub is not None and "session" in meta_sub.columns:
        sessions = meta_sub["session"].astype(str).to_numpy()

    prefixes = tuple(str(p) for p in cfg.trainable_prefixes)
    base_snapshot = snapshot_params_by_prefix(model, prefixes)

    results: dict[str, dict] = {}
    for k in cfg.k_shots:
        k_int = int(k)
        per_run = []
        for r in range(int(cfg.n_resamples)):
            rng = np.random.RandomState(int(cfg.seed) + int(fold_seed) + 1000 * k_int + r)
            support_idx, query_idx = split_support_query(
                y=y,
                sessions=sessions,
                cfg=cfg,
                k=k_int,
                num_classes=int(num_classes),
                rng=rng,
            )

            # Restore base parameters and adapt on support.
            restore_params(model, base_snapshot)
            support_loader = _loader_from_subset(
                test_ds,
                support_idx,
                batch_size=min(int(cfg.batch_size), int(support_idx.size)),
                shuffle=True,
                seed=int(cfg.seed) + int(fold_seed) + 999 + 1000 * k_int + r,
            )
            finetune_head_on_support(
                model,
                support_loader,
                device=device,
                trainable_prefixes=prefixes,
                tick_loss_cfg=tick_loss_cfg,
                lr=float(cfg.lr),
                weight_decay=float(cfg.weight_decay),
                steps=int(cfg.steps),
                grad_clip=cfg.grad_clip,
            )

            # Evaluate on query.
            query_loader = _loader_from_subset(
                test_ds,
                query_idx,
                batch_size=128,
                shuffle=False,
                seed=int(cfg.seed) + int(fold_seed) + 123 + 1000 * k_int + r,
            )
            m = evaluate(model, query_loader, device=device, eval_cfg=eval_cfg)
            per_run.append(m.to_dict())

        # Aggregate over resamples.
        arr = {key: np.asarray([float(d[key]) for d in per_run], dtype=np.float64) for key in ("accuracy", "kappa", "macro_f1")}
        results[str(k_int)] = {
            "k_shot": k_int,
            "n_resamples": int(cfg.n_resamples),
            "protocol": str(cfg.protocol),
            "support_session": cfg.support_session,
            "query_session": cfg.query_session,
            "mean": {k: float(v.mean()) for k, v in arr.items()},
            "std": {k: float(v.std(ddof=0)) for k, v in arr.items()},
            "runs": per_run,
        }

    # Restore base parameters at the end (important if caller continues using the model).
    restore_params(model, base_snapshot)
    return {"enabled": True, "results": results, "config": cfg.__dict__}
