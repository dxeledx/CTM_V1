# EEG-CTM (BCI-IV-2a LOSO)

This folder adds an EEG motor-imagery cross-subject classification training framework on top of the original CTM source tree (`continuous-thought-machines-main/`).

## Run (LOSO)

From repo root:

```bash
python3 -m eeg_ctm.train --config configs/bciiv2a_loso.yaml
```

Outputs:

- `runs/<exp_name>/fold_<test_subject>/train.log`
- `runs/<exp_name>/fold_<test_subject>/metrics.json`
- `runs/<exp_name>/metrics_summary.json`

Note: to avoid overwriting, if `runs/<exp_name>/` already exists and is non-empty, the runner automatically
creates `runs/<exp_name>__run01/`, `__run02/`, ... (set `overwrite: true` in config to overwrite in-place).

## Leakage prevention (LOSO)

The implementation enforces:

- Test subject data is never used for training or model selection.
- Validation subject (if enabled) is excluded from the S&R donor pool.
- Standardization is **trial-wise** (each trial uses its own statistics), so there is no train/test statistic sharing.

## Config knobs

All key hyperparameters are configurable in YAML/JSON:

- Sessions: `data.sessions` (default uses both `session_T` + `session_E`).
- Epoch window: `data.window.tmin_s/tmax_s` (default `0–4s` aligned to cue-onset, i.e. `2–6s` of each trial).
- Trial-wise standardization: `data.standardize.mode` (`zscore|robust`).
- S&R augmentation: `augment.sr.*` and injection mode `augment.injection.mode` (`none|concat|replace`).
- Tokenizer v1: `tokenizer.*` (targets `N≈20` tokens via `token_pool_kernel/stride`).
- CTM core: `ctm.*` (ticks, pairs, fusion, NLM, head).
- Tick loss: `loss.tick_loss.*` and inference readout: `readout.*`.
- Validation strategy:
  - `split.val_strategy=within_subject` (recommended): per-training-subject holdout by `split.within_subject_val_fraction` for early stopping.
  - `split.val_strategy=next|fixed|none` are also supported.
- Optional constraints:
  - Supervised contrastive: `optional.supcon.*` (requires `optional.sampler.type=subject_class_balanced` in practice).
  - Subject adversarial (GRL): `optional.adversarial.*` (λ warmup).

## Self-check / smoke test

Fast dataset-backed sanity check (loads a tiny cached subset and runs 1 epoch):

```bash
python3 -m eeg_ctm.self_check --config configs/bciiv2a_loso.yaml
```

Pure-model smoke test (synthetic tensors):

```bash
python3 -m eeg_ctm.smoke_test
```
