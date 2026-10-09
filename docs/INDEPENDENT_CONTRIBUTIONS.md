# Independent contributions

This project began as my submission pipeline for the **MITSUI & CO. Commodity Prediction
Challenge** on Kaggle (2025). This document separates what came from that competition work
from what was repaired, redesigned or newly built in the independent research extension. No
competition ranking or award is claimed.

## 1. Origin: what came from the competition

| Component | Origin | Status now |
|---|---|---|
| Dataset: `train.csv`, `train_labels.csv`, `target_pairs.csv`, `test.csv` | Kaggle competition (MITSUI & CO.) | Used unchanged as the primary research dataset (`data/raw/mitsui/`, not redistributed) |
| Task definition: 424 targets, lags 1–4, Spearman-based score | Kaggle competition | Re-derived from the data and re-implemented (see §2) |
| `run_pipeline.py`, `preprocess_timeseries.py`, `main_workflow.py`, original README | My competition code | Preserved verbatim in `legacy/original/`; not used by the active code |
| Ideas: multi-asset features, rolling statistics, cross-asset correlations, market-regime indicators, LSTM with attention, multi-model ensemble with uncertainty | My competition code | Kept as research directions, re-implemented from scratch (§3) |

## 2. Repaired: defects found and fixed

The full list is in [LEGACY_AUDIT.md](LEGACY_AUDIT.md). The fixes that change conclusions:

| Legacy defect | Repair |
|---|---|
| Trained on randomly generated synthetic prices; never read the real data | `MitsuiAdapter` loads and validates the real files; synthetic data is confined to a clearly labelled test fixture |
| Targets = raw price-level differences, shifted again by `lag` | Verified target formula $\log(P_{t+1+h}/P_{t+1})$ (pair = difference), reproduced to 1e-6 for all 424 targets; explicit `TargetSpec` with gap and horizon |
| Two-sided interpolation, back-fill, centred Gaussian smoothing, global wavelets, full-sample scaling | Strictly causal features (past-only forward fill, rolling/EWMA/expanding statistics); train-only scaling; bit-identical causality tests |
| `test.csv` treated as test data | Shown to be a verbatim copy of training rows; replaced by a chronological 500-date holdout |
| CV indices applied to a re-engineered frame; 1-row embargo for 4-day labels | Walk-forward folds with horizon-specific purging derived from label availability; purged early-stopping block |
| LSTM: constructor crash, wrong output shape, shallow-copied "best" weights | Redesigned multi-output LSTM with shape checks, masked loss, `deepcopy` early stopping (§3) |
| "Uncertainty" = variance of 3 seed-models, uncalibrated, scale factor never applied | Split-conformal intervals calibrated on development residuals with holdout coverage tests; disagreement analysed separately |
| No competition metric; MSE on price levels; "direction" of price changes | Exact competition-style score plus mean IC, IC IR, HAC t-stats, block-bootstrap CIs, paired tests, per-horizon/regime/category breakdowns |

## 3. Redesigned

- **Problem formulation.** The 424 targets are reformulated as linear combinations of
  106 asset returns × 4 horizons (*asset-factorised forecasting*). Pooled panel models replace
  424 separate per-target networks.
- **Training target.** Vol-scaled, cross-sectionally demeaned returns matched to the rank
  metric, with the raw-label variant kept as an ablation.
- **Sequence model.** A small unidirectional multi-horizon LSTM replaces the
  bidirectional/attention/skip/SWA design. It must compete against simple benchmarks.
- **Ensembles.** Combinations of *different* model classes with four documented weighting
  rules replace seed-ensembles weighted by `1/val_loss`.
- **Software.** A package with adapters, configs, CLI, logging, process isolation, tests and
  machine-generated reports replaces three scripts with hard-coded paths.

## 4. Newly implemented

| Area | New components |
|---|---|
| Data | Dataset-independent `MarketData` schema with validation; MITSUI adapter with hand-curated instrument metadata (market, asset class, FX base/quote); **generic long-format adapter**; deterministic synthetic fixture; FRED energy fetch script and demo config |
| Targets | `TargetSpec` with explicit kinds: log return, return spread, price-spread change, direction. Information interval and availability date per target |
| Features | 38 causal, scale-free features in 4 groups (own, cross-asset, relative-value, regime); six economically defined cross-asset factors; causal regime labels |
| Validation | Label-availability walk-forward, purged internal validation, development/holdout protocol with a documented design freeze |
| Models | Benchmarks (historical mean, momentum, reversal); pooled Ridge; pooled LightGBM; multi-horizon LSTM; direct pair models (pooled and per-target); four ensemble weighting schemes incl. regime-conditional and causal online |
| Evaluation | Competition-score replication with a reference-implementation test; Newey–West HAC; moving-block bootstrap; paired tests; horizon, regime, category and time-block breakdowns; rank stability; directional accuracy |
| Risk | Split-conformal intervals and coverage by regime; disagreement-vs-error analysis; hypothetical dollar-neutral backtest with explicit execution timing and cost sensitivity |
| Diagnostics | Data audit command; information-timing diagnostic (gap-0 vs gap-1 labels); feature attribution at the holdout start |
| Engineering | `pyproject.toml`, configs (FAST/FULL/smoke/demo), CLI, structured logs, per-model subprocess isolation (OpenMP conflict), 62 tests, figures and tables generated from saved results, Streamlit dashboard |

## 5. Experiments added (none existed in the competition code)

| Exp. | Question | Design |
|---|---|---|
| A | Do ML/DL models beat simple benchmarks? | hist_mean, momentum, reversal vs Ridge vs LightGBM vs LSTM, same dates and targets, paired tests |
| B | Does cross-asset information help? | Same model with own-asset features vs all feature groups (Ridge, LightGBM) |
| C | Does accuracy depend on market regime? | IC by causal volatility, trend and correlation regime |
| D | Does regime-aware combination help? | Equal vs IC-weighted vs regime-conditional vs online weights; dev-selected member set |
| E | How does predictability change with horizon? | Per-horizon IC and IC IR; asset-level vs pair-level universes |
| F | Pooled/factorised vs target-specific? | Asset-factorised Ridge vs pooled pair Ridge vs per-target Ridge |
| — | Does the label design matter? | Demeaned vs raw vol-scaled labels |
| — | Why does the gap matter? | Simple signals against gap-0 vs gap-1 labels |
| — | Generality | Same pipeline on public FRED energy data via the generic adapter |

## 6. Research questions investigated

- **RQ1.** How accurately can historical commodity, equity and FX information predict future
  commodity-related returns?
- **RQ2.** Do cross-asset relationships improve forecasts relative to own-history models?
- **RQ3.** Do models perform differently across volatility regimes?
- **RQ4.** Can regime-aware combination and uncertainty estimation improve stability and
  usefulness?

The answers, including the negative ones, are in [RESEARCH_REPORT.md](RESEARCH_REPORT.md)
and [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

## 7. Phase 2: volatility research (independent extension)

Research question:

> Do volatility dynamics provide incremental predictive information for commodity returns,
> and does their usefulness vary across market regimes?

| Origin | What was taken | What was done independently |
|---|---|---|
| Econometrics literature | ARCH/GARCH, GJR asymmetry, Student-t innovations, EWMA (RiskMetrics); Ljung–Box, ARCH-LM, Jarque–Bera, Engle–Ng sign-bias; QLIKE and Patton (2011) robust losses; Diebold–Mariano; Model Confidence Set; Hansen SPA; Mincer–Zarnowitz | A causal walk-forward volatility engine for 106 assets and 424 target spreads (re-estimation every 21 days, exact multi-step recursion), validated against `arch` analytic forecasts. A **panel DM test** on cross-sectionally averaged QLIKE differences. A pre-registered hypothesis set |
| Kaggle competition practice (publicly documented) | Lagged observable target values as inputs (8th place: lag-4 target values); historical rank features; target-level tree models | A target-level pooled LightGBM with an explicit **four-group feature taxonomy** (return/rank, volatility-normalised, volatility-state, cross-asset). The competition's lagged-label files were used to verify, not assume, the availability rule |
| — (new) | — | The separation of the **normaliser** channel (label scaling by EWMA / GARCH / none) from the **predictor** channel (volatility features); the 2×2 factorial of {±volatility} × {±cross-asset}; Holm-adjusted primary tests, MCS and SPA across arms; the pre-registration and freeze record; target-volatility tercile analysis; the post-hoc decomposition into market vs own volatility state; robustness profiles; replication on public FRED data with a never-examined 2026 sample |

New code:

- `volatility/` (models, engine, diagnostics, evaluation);
- `features/target_features.py`;
- `models/target_models.py`;
- `experiments/volatility.py` and `experiments/vol_report.py`;
- `visualization/vol_figures.py`;
- `tests/test_volatility.py`;
- `configs/vol*.yaml`.

## 8. Tooling disclosure

The redesign, implementation and documentation were carried out with an AI coding assistant
(Claude Code) working under my direction in this repository. All empirical numbers come from
code executed in this repository and are regenerated by the documented commands.
