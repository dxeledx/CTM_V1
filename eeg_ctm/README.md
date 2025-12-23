# EEG-CTM (BCI-IV-2a)

This folder adds an EEG motor-imagery cross-subject classification training framework on top of the original CTM source tree (`continuous-thought-machines-main/`).

## Run

From repo root:

```bash
python3 -m eeg_ctm.train --config configs/bciiv2a_loso.yaml
```

Within-subject cross-session (train on one session, test on the other):

```bash
python3 -m eeg_ctm.train --config configs/bciiv2a_within_subject_cross_session.yaml
```

Outputs:

- `runs/<exp_name>/fold_<test_subject>/train.log`
- `runs/<exp_name>/fold_<test_subject>/metrics.json`
- `runs/<exp_name>/metrics_summary.json`

Note: to avoid overwriting, if `runs/<exp_name>/` already exists and is non-empty, the runner automatically
creates `runs/<exp_name>__run01/`, `__run02/`, ... (set `overwrite: true` in config to overwrite in-place).

## Leakage prevention

The implementation enforces:

- Test subject data is never used for training or model selection.
- For within-subject cross-session: test **session** data is never used for training or model selection.
- Validation subject (if enabled) is excluded from the S&R donor pool.
- Standardization is **trial-wise** (each trial uses its own statistics), so there is no train/test statistic sharing.

## Few-shot (optional, test-time adaptation)

Baseline LOSO in this repo is **zero-shot**: train on training subjects, then directly evaluate on the held-out test subject.

If you enable `few_shot.enabled: true`, the runner will additionally perform **test-time, subject-specific few-shot adaptation**
after model selection (i.e., after loading `best.pt`). Default is **head-only** adaptation:

- Freeze Tokenizer + CTM core
- Train only the classification head (`trainable_prefixes: ["ctm.head"]`)

Recommended protocol for BCI-IV-2a:

- `few_shot.protocol: cross_session`
- Adapt on session `"0train"` using balanced K-shot per class support
- Report metrics on session `"1test"` query
- Repeat `n_resamples` times and report mean±std

Few-shot results are stored under `fold_<test_subject>/metrics.json -> few_shot` and summarized in
`metrics_summary.json -> few_shot_summary`.

## Config knobs

All key hyperparameters are configurable in YAML/JSON:

- Training protocol:
  - LOSO (cross-subject): `protocol.type=loso`
  - Within-subject cross-session: `protocol.type=within_subject_cross_session` with `protocol.train_session`/`protocol.test_session`
- Sessions: `data.sessions` (default: `null` = use all sessions returned by MOABB; typically 2 sessions, e.g. `"0train"` + `"1test"`).
- FilterBank (multi-band): `data.filterbank.enabled` + `data.filterbank.bands_hz` (MOABB FilterBank; output is flattened to `C = C_raw * n_bands`).
- Epoch window: `data.window.tmin_s/tmax_s` (default `0–4s` aligned to cue-onset, i.e. `2–6s` of each trial).
- Trial-wise standardization: `data.standardize.mode` (`zscore|robust`).
- S&R augmentation: `augment.sr.*` and injection mode `augment.injection.mode`:
  - baseline: `none|concat|replace`
  - token-level fusion/modulation (recommended cross-subject): `adain|film|xattn`
- Tokenizer v1: `tokenizer.*` (targets `N≈20` tokens via `token_pool_kernel/stride`).
- CTM core: `ctm.*` (ticks, pairs, fusion, NLM, head).
- Tick loss: `loss.tick_loss.*` and inference readout: `readout.*`.
- Readout stop-grad: `readout.certainty_weighted.detach_certainty` (helps prevent “certainty shortcut” when certainty weights appear in training objectives).
- Learnable tick pooling: set `readout.mode=learned_attn` and enable `ctm.tick_pool_enabled=true` (train with `loss.pool_ce.enabled=true`).
- Wasserstein-DRO (feature-space PGD): `wdro.*` (disabled by default; robustifies pre-head representations under L2-ball perturbations; set `wdro.mix_robust=1.0` to avoid adding extra clean CE on top of the main tick-loss).
- Validation strategy:
  - `split.val_strategy=within_subject` (recommended): per-training-subject holdout by `split.within_subject_val_fraction` for early stopping.
  - `split.val_strategy=next|fixed|none` are also supported.
- Optional constraints:
  - Supervised contrastive (cross-subject invariance): set `optional.supcon.enabled=true` (often pair with `optional.sampler.type=subject_class_balanced`).
  - Subject adversarial / domain confusion (GRL): set `optional.adversarial.enabled=true` (λ warmup).
  - Representation aggregation stop-grad: `optional.rep_agg.certainty_weighted.detach_certainty`.
- Few-shot test-time adaptation: `few_shot.*` (disabled by default).

Training stability diagnostics:

- `tick_gap_ce`: average per-batch tick-gap of CE (2nd best - best)
- `tick_gap_certainty`: average per-batch tick-gap of certainty (best - 2nd best)
- `wdro_*` (when enabled): robust loss gap, delta norm, effective rho/step size under warmup
- `train_loss_pool_ce` (when enabled): pooled readout CE (trains learnable tick pooling)

## Cross-subject objectives (implemented)

- Subject-adversarial (GRL): `optional.adversarial.enabled=true` (+ `lambda_max`, `warmup`, `head_hidden`).
- Supervised contrastive: `optional.supcon.enabled=true` (+ `tau`, `lambda_con`); for stable batches use `optional.sampler.type=subject_class_balanced`.
- Feature-space Wasserstein-DRO: `wdro.enabled=true` (+ `rho`, `steps`, `step_size`, `warmup_epochs`).

## Self-check / smoke test

Fast dataset-backed sanity check (loads a tiny cached subset and runs 1 epoch):

```bash
python3 -m eeg_ctm.self_check --config configs/bciiv2a_loso.yaml
```

Pure-model smoke test (synthetic tensors):

```bash
python3 -m eeg_ctm.smoke_test
```
