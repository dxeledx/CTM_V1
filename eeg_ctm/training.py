"""
Training loop utilities for EEG-CTM (classification-only core).

Optional invariance constraints (SupCon / GRL) are implemented in separate modules
and plugged into the same step function (see eeg_ctm/train.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from eeg_ctm.augment.sr import ClasswiseMemoryBank, SegmentationRecombinationAugmenter
from eeg_ctm.models.aggregation import CertaintyWeightedConfig, aggregate_logits, aggregate_rep
from eeg_ctm.models.adversarial import AdvConfig, SubjectHead, grl, linear_warmup
from eeg_ctm.models.losses import TickLossConfig, tick_classification_loss
from eeg_ctm.models.supcon import ProjectionHead, SupConConfig, supervised_contrastive_loss
from eeg_ctm.models.wdro import WDROConfig, wdro_rep_objective


InjectionMode = Literal["none", "concat", "replace"]


@dataclass(frozen=True)
class InjectionConfig:
    mode: InjectionMode = "concat"
    replace_p: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 0
    grad_clip: Optional[float] = 1.0
    amp: bool = False


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    tick_loss_cfg: TickLossConfig,
    augmenter: Optional[SegmentationRecombinationAugmenter],
    bank: Optional[ClasswiseMemoryBank],
    injection: InjectionConfig,
    rng: torch.Generator,
    grad_clip: Optional[float] = 1.0,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    n_steps = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        uid = batch.get("uid")

        if augmenter is not None and bank is not None and injection.mode != "none":
            x_aug = augmenter(x, y, uid, bank=bank, generator=rng)
            x_cat = torch.cat([x, x_aug], dim=0)
            y_cat = torch.cat([y, y], dim=0)
            logits_cat, cert_cat, _ = model(x_cat)

            B = x.shape[0]
            if injection.mode == "concat":
                loss_cls, details = tick_classification_loss(logits_cat, cert_cat, y_cat, cfg=tick_loss_cfg)
            elif injection.mode == "replace":
                p = float(injection.replace_p)
                if not (0.0 <= p <= 1.0):
                    raise ValueError("replace_p must be in [0,1]")
                mask = torch.rand((B,), generator=rng, device=device) < p
                logits1, logits2 = logits_cat[:B], logits_cat[B:]
                cert1, cert2 = cert_cat[:B], cert_cat[B:]
                logits_mix = logits1.clone()
                cert_mix = cert1.clone()
                logits_mix[mask] = logits2[mask]
                cert_mix[mask] = cert2[mask]
                loss_cls, details = tick_classification_loss(logits_mix, cert_mix, y, cfg=tick_loss_cfg)
            else:
                raise ValueError(injection.mode)
        else:
            logits, cert, _ = model(x)
            loss_cls, details = tick_classification_loss(logits, cert, y, cfg=tick_loss_cfg)

        optimizer.zero_grad(set_to_none=True)
        loss_cls.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()

        total_loss += float(loss_cls.detach().cpu())
        n_steps += 1

    avg_loss = total_loss / max(1, n_steps)
    return avg_loss, {"train_loss": avg_loss}


def train_one_epoch_with_constraints(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    tick_loss_cfg: TickLossConfig,
    augmenter: Optional[SegmentationRecombinationAugmenter],
    bank: Optional[ClasswiseMemoryBank],
    injection: InjectionConfig,
    rng: torch.Generator,
    grad_clip: Optional[float],
    # ---- representation aggregation ----
    rep_mode: str,
    rep_cw_alpha: float,
    rep_cw_detach: bool,
    # ---- train accuracy readout ----
    acc_readout: str,
    acc_cw_alpha: float,
    acc_cw_detach: bool,
    # ---- wasserstein DRO (feature-space) ----
    wdro_cfg: WDROConfig,
    epoch: int,
    # ---- supervised contrastive ----
    supcon_cfg: SupConConfig,
    proj_head: Optional[ProjectionHead],
    # ---- subject adversarial ----
    adv_cfg: AdvConfig,
    subject_head: Optional[SubjectHead],
    subject_id_map: Optional[dict[int, int]],
    global_step: int,
    total_steps: int,
) -> tuple[float, dict, int]:
    """
    Full training epoch supporting optional SupCon and GRL constraints.
    Returns (avg_loss, details, updated_global_step).
    """
    model.train()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_supcon = 0.0
    total_loss_adv = 0.0
    total_lambda_adv = 0.0
    cls_details_sum: dict[str, float] = {}
    wdro_details_sum: dict[str, float] = {}
    train_correct = 0
    train_total = 0
    n_steps = 0

    if supcon_cfg.enabled:
        if augmenter is None or bank is None:
            raise ValueError("SupCon requires train-time augmentation to produce two views")
        if proj_head is None:
            raise ValueError("proj_head is required when SupCon is enabled")

    if adv_cfg.enabled:
        if subject_head is None or subject_id_map is None:
            raise ValueError("subject_head and subject_id_map are required when adversarial is enabled")

    rep_cw = CertaintyWeightedConfig(alpha=float(rep_cw_alpha), detach_certainty=bool(rep_cw_detach))
    acc_cw = CertaintyWeightedConfig(alpha=float(acc_cw_alpha), detach_certainty=bool(acc_cw_detach))

    use_wdro = bool(wdro_cfg.enabled) and float(getattr(wdro_cfg, "lambda_wdro", 1.0)) != 0.0

    for batch in loader:
        global_step += 1
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        subj = batch.get("subject")
        uid = batch.get("uid")

        B = x.shape[0]

        need_view2 = (
            augmenter is not None
            and bank is not None
            and (supcon_cfg.enabled or injection.mode in ("concat", "replace"))
        )

        if need_view2:
            x_aug = augmenter(x, y, uid, bank=bank, generator=rng)
            x_cat = torch.cat([x, x_aug], dim=0)
            y_cat = torch.cat([y, y], dim=0)
            if use_wdro:
                logits_cat, cert_cat, z_cat, feat_cat = model(x_cat, return_features=True)
                feat1 = feat_cat[:B]
            else:
                logits_cat, cert_cat, z_cat = model(x_cat)
                feat1 = None
            logits1, logits2 = logits_cat[:B], logits_cat[B:]
            cert1, cert2 = cert_cat[:B], cert_cat[B:]
            z1, z2 = z_cat[:B], z_cat[B:]
        else:
            if use_wdro:
                logits1, cert1, z1, feat1 = model(x, return_features=True)
            else:
                logits1, cert1, z1 = model(x)
                feat1 = None
            logits2 = cert2 = z2 = None
            y_cat = None

        # ---- classification loss ----
        if need_view2 and injection.mode == "concat":
            assert y_cat is not None
            loss_cls, cls_details = tick_classification_loss(logits_cat, cert_cat, y_cat, cfg=tick_loss_cfg)
        elif need_view2 and injection.mode == "replace":
            p = float(injection.replace_p)
            if not (0.0 <= p <= 1.0):
                raise ValueError("replace_p must be in [0,1]")
            mask = torch.rand((B,), generator=rng, device=device) < p
            logits_mix = logits1.clone()
            cert_mix = cert1.clone()
            assert logits2 is not None and cert2 is not None
            logits_mix[mask] = logits2[mask]
            cert_mix[mask] = cert2[mask]
            loss_cls, cls_details = tick_classification_loss(logits_mix, cert_mix, y, cfg=tick_loss_cfg)
        else:
            loss_cls, cls_details = tick_classification_loss(logits1, cert1, y, cfg=tick_loss_cfg)

        loss = loss_cls
        for k, v in cls_details.items():
            cls_details_sum[k] = cls_details_sum.get(k, 0.0) + float(v)

        # ---- WDRO: feature-space robustification (rep-level, v1) ----
        if use_wdro:
            if feat1 is None:
                raise RuntimeError("WDRO enabled but feat1 is missing (return_features not provided by model)")
            if not hasattr(model, "ctm") or not hasattr(getattr(model, "ctm"), "head"):
                raise RuntimeError("WDRO requires model.ctm.head")

            # Aggregate the *pre-head* features over ticks (same shape semantics as z_ticks).
            rep = aggregate_rep(feat1, cert1, mode=rep_mode, cw=rep_cw)  # [B, Dout]
            wdro_loss, wdro_details = wdro_rep_objective(
                rep,
                y,
                head=getattr(model, "ctm").head,
                cfg=wdro_cfg,
                epoch=int(epoch),
            )
            lam = float(getattr(wdro_cfg, "lambda_wdro", 1.0))
            loss = loss + lam * wdro_loss
            for k, v in wdro_details.items():
                wdro_details_sum[k] = wdro_details_sum.get(k, 0.0) + float(v)

        # ---- train accuracy (on original view only) ----
        with torch.no_grad():
            logits_pred = aggregate_logits(logits1, cert1, mode=acc_readout, cw=acc_cw)
            pred = logits_pred.argmax(dim=-1)
            train_correct += int((pred == y).sum().item())
            train_total += int(y.numel())

        # ---- aggregate representations ----
        rep1 = None
        rep2 = None
        if supcon_cfg.enabled or adv_cfg.enabled:
            rep1 = aggregate_rep(z1, cert1, mode=rep_mode, cw=rep_cw)
            if supcon_cfg.enabled:
                assert z2 is not None and cert2 is not None
                rep2 = aggregate_rep(z2, cert2, mode=rep_mode, cw=rep_cw)

        # ---- supervised contrastive ----
        if supcon_cfg.enabled:
            assert proj_head is not None and rep1 is not None and rep2 is not None and subj is not None
            subj_t = subj.to(device)
            e1 = proj_head(rep1)
            e2 = proj_head(rep2)
            emb = torch.cat([e1, e2], dim=0)
            labels2 = torch.cat([y, y], dim=0)
            subjects2 = torch.cat([subj_t, subj_t], dim=0)
            loss_con = supervised_contrastive_loss(
                emb,
                labels=labels2,
                subjects=subjects2,
                tau=float(supcon_cfg.tau),
                include_same_instance=bool(supcon_cfg.include_same_instance),
            )
            loss = loss + float(supcon_cfg.lambda_con) * loss_con
            total_loss_supcon += float(loss_con.detach().cpu())

        # ---- subject adversarial (GRL) ----
        if adv_cfg.enabled:
            assert rep1 is not None and subj is not None and subject_head is not None and subject_id_map is not None
            subj_ids = subj.detach().cpu().tolist()
            subj_mapped = torch.as_tensor([subject_id_map[int(s)] for s in subj_ids], device=device, dtype=torch.long)
            progress = float(global_step) / float(max(1, total_steps))
            lam_adv = float(adv_cfg.lambda_max) * linear_warmup(progress, float(adv_cfg.warmup))
            subj_logits = subject_head(grl(rep1, lam_adv))
            loss_adv = torch.nn.functional.cross_entropy(subj_logits, subj_mapped)
            # GRL already scales the reversed gradient by lam_adv; do NOT scale again here
            # (otherwise the feature extractor receives ~lam_adv^2).
            loss = loss + loss_adv
            total_loss_adv += float(loss_adv.detach().cpu())
            total_lambda_adv += float(lam_adv)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_loss_cls += float(loss_cls.detach().cpu())
        n_steps += 1

    avg_loss = total_loss / max(1, n_steps)
    details = {
        "train_loss": avg_loss,
        "train_loss_cls": total_loss_cls / max(1, n_steps),
        "train_accuracy": float(train_correct) / float(max(1, train_total)),
    }
    for k, s in cls_details_sum.items():
        details[f"train_{k}"] = float(s) / max(1, n_steps)
    for k, s in wdro_details_sum.items():
        details[f"train_{k}"] = float(s) / max(1, n_steps)
    if supcon_cfg.enabled:
        details["train_loss_supcon"] = total_loss_supcon / max(1, n_steps)
    if adv_cfg.enabled:
        details["train_loss_adv"] = total_loss_adv / max(1, n_steps)
        details["train_lambda_adv"] = total_lambda_adv / max(1, n_steps)
    return avg_loss, details, global_step
