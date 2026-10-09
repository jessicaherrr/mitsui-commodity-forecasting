# Legacy audit: original Kaggle pipeline

**Scope.** This audit covers the original competition code, preserved unchanged in
[`legacy/original/`](../legacy/original/): `run_pipeline.py`, `preprocess_timeseries.py`,
`main_workflow.py` and the original README. SHA-256 hashes were recorded when the files were
moved (see `docs/IMPLEMENTATION_PROGRESS.md`).

**Method.** I read every function, checked each finding by running it where possible
(`legacy_probe` scripts that stub `optuna` and call the legacy classes directly), and checked
data facts programmatically with `python -m commodity_research.cli audit`, which writes
`results/audit/data_audit.json`.

**Verdict.** The legacy pipeline cannot produce a valid result. Its entry point never reads
the competition data. It trains on randomly generated synthetic prices (and the generator
crashes). It defines targets as price-level differences instead of the official log-return
differences. Its preprocessing is non-causal, and its LSTM can be neither constructed nor
trained as written. None of its outputs should be cited as evidence of anything.

Severity: **Critical** = invalidates any result; **High** = causes leakage or wrong
numbers; **Medium** = wrong but secondary; **Low** = quality or robustness issue.

---

## 1. Dataset structure and target definitions

| # | Finding | Severity | Evidence |
|---|---|---|---|
| 1.1 | `main()` never loads `train.csv` or `train_labels.csv`. It calls `create_enhanced_sample_data(2000)`, which generates **random synthetic prices** for 19 hypothetical assets. `run_pipeline.py` preprocesses the real files, then calls `main()`, which ignores them. | Critical | `main_workflow.py:1342`; audit `legacy_static_checks.main_uses_synthetic_generator = true`, `main_reads_train_csv = false` |
| 1.2 | The synthetic generator crashes: `1.2 if bull_market else 0.8` with `bull_market` a NumPy array raises `ValueError: truth value of an array ... is ambiguous`. The pipeline therefore cannot run end to end. | Critical | reproduced by calling `create_enhanced_sample_data(200)` |
| 1.3 | Targets are built as **raw price-level differences**, `all_data[feat1] - all_data[feat2]`. The official labels are **differences of log returns**. A price spread (e.g. LME copper in USD/t minus an ETF price in USD) is unit-dependent, non-stationary and is not a return. | Critical | `main_workflow.py:220`; audit: correlation of the legacy price-level target with the official label ≈ 0 (−0.05 … +0.04) |
| 1.4 | Single-asset targets are taken as the raw price level (`y_target = all_data[pair_expr]`). | Critical | `main_workflow.py:225` |
| 1.5 | Column names do not match: target pairs use `US_Stock_*_adj_close` while the synthetic data uses `*_Close`. Most targets raise `Features not found` and are silently skipped (`except ... continue`). | High | `build_features_for_target` |
| 1.6 | The fallback in `run_pipeline.py` writes **random normal numbers** as "sample predictions" when modelling fails. They are saved alongside real outputs and could be mistaken for forecasts. | High | `create_sample_predictions` |

**Verified target definition** (max abs error ≤ 1e-6 over all 424 targets, 742,640 cells):

```
target(t) = log(P_A[t+1+lag] / P_A[t+1]) − log(P_B[t+1+lag] / P_B[t+1])      (pair "A - B")
target(t) = log(P_A[t+1+lag] / P_A[t+1])                                     (single asset)
```

All 1,421 official label cells that cannot be recomputed belong to the last rows of the
sample. Their prices lie beyond the end of `train.csv`.

## 2. Label alignment and forecasting horizons

| # | Finding | Severity |
|---|---|---|
| 2.1 | `EnhancedFinancialDataset` applies `y.shift(-lag)` to the target. The official labels are already forward-looking (from t+1 to t+1+lag), so applied to them this would shift a second time and predict the wrong window. Applied to price levels, as in the legacy code, it predicts a future *price level*. | Critical |
| 2.2 | The `lag` column is read as both a label shift and a feature "safe lag" (`safe_lag = target_lag + 3`). Lag is the return **horizon**; the 1-date **gap** between decision and return start is a separate quantity the legacy code never models. | High |
| 2.3 | `predict_with_model_optimized` writes the prediction for row t to row `t + lag`, a third, inconsistent alignment. | High |

## 3. Market-data availability at prediction time

| # | Finding | Severity |
|---|---|---|
| 3.1 | No notion of decision time. LME, JPX, US and FX close at different times on the same `date_id`, and labels become observable only at `t+1+lag`. Neither fact is represented. | High |
| 3.2 | Exchange holidays (LME 51 dates, JPX 116, US 67 with all closes missing) are filled with **two-sided interpolation**, so a holiday value depends on the next trading day's price. | High |

## 4. Look-ahead bias and leakage

| # | Finding | Severity |
|---|---|---|
| 4.1 | `fill_missing(strategy='interpolate')` uses `interpolate(limit_direction='both')` followed by `.bfill()`, which **back-fills from the future**. | High |
| 4.2 | `gaussian_smooth` uses `gaussian_filter1d(sigma=2, mode='reflect')`, a **centred** filter. Each smoothed value depends on about 8 future observations. The pipeline applies it to every column. | High |
| 4.3 | `wavelet_denoise` decomposes and thresholds the **entire series** at once, using a global noise estimate. Every value depends on the full sample. | High |
| 4.4 | IQR outlier capping uses quantiles of the **whole** training file, including the periods later used for validation and testing. | Medium |
| 4.5 | Train and test files are preprocessed **independently**, each with its own interpolation, smoothing and statistics, so their transformations differ. | Medium |

## 5. Train / validation / test methodology

| # | Finding | Severity |
|---|---|---|
| 5.1 | `test.csv` is **not an independent test set**. Its 134 rows (date_id 1827–1960) are byte-identical to the last 134 rows of `train.csv`, plus an `is_scored` flag, and it has no labels. It is the competition API mock. | Critical (if used for evaluation) |
| 5.2 | `TimeSeriesCVSplitter(n_splits=5)` yields only **4** folds. The embargo is one row regardless of horizon, which is too short for overlapping 4-day labels. | Medium |
| 5.3 | In `grid_search_cv`, `prepare_data(X_train, ...)` re-splits the training fold 70/15/15 internally. The fold's validation set is then built from `data_info['X_engineered'].iloc[val_idx]`, i.e. **outer-fold indices applied to an inner, re-engineered, shorter frame**. That yields wrong rows or an IndexError. | High |
| 5.4 | The reported "test loss" is computed on the last 15% of each training fold, not on a held-out period. | Medium |

## 6. Feature selection and scaling

| # | Finding | Severity |
|---|---|---|
| 6.1 | `RobustScaler` is fitted on the **full** target-specific feature frame before any split. Validation and test statistics leak into training. | High |
| 6.2 | Feature selection uses `y_dummy = np.random.randn(...)`, a **random target**. The default `variance` method ranks unscaled features by variance, which favours price levels and volumes over informative features. | High |
| 6.3 | Cross-asset correlations are computed on `.iloc[:1000]` only, and are shifted twice. | Medium |
| 6.4 | Features include raw price levels (`*_Close`, moving averages of levels). For vendor-adjusted closes, levels depend on later corporate actions. | Medium |

## 7. LSTM architecture and tensor dimensions

| # | Finding | Severity | Evidence |
|---|---|---|---|
| 7.1 | **The model cannot be constructed.** `_initialize_weights` calls `xavier_uniform_` on every `weight`, including 1-D LayerNorm weights, which raises `ValueError: Fan in and fan out can not be computed for tensor with fewer than 2 dimensions`. | Critical | reproduced |
| 7.2 | **Wrong output shape.** The skip connection adds a `(batch, hidden/2)` projection to a `(batch, 1)` output. Broadcasting yields `(batch, 32)` for hidden=64, and `squeeze(-1)` leaves it 2-D. The loss then fails, or silently broadcasts in other configurations. | Critical | reproduced: output `(8, 32)` for batch 8 |
| 7.3 | `best_model_state = model.state_dict().copy()` is a shallow copy. The stored "best" tensors are mutated by later optimisation steps, so early stopping restores the *last* weights. | High | reproduced |
| 7.4 | When SWA is enabled (the default), the SWA model **replaces** the early-stopped model regardless of validation loss. `update_bn` is called on a loader that yields tuples. | Medium | code |
| 7.5 | Bidirectional LSTM plus attention plus a 3-layer, 256-unit head on roughly 1,000 noisy daily samples per target: hundreds of thousands of parameters per target, far beyond what the signal-to-noise ratio supports. | Medium | design |

## 8. Training and hyper-parameter search

| # | Finding | Severity |
|---|---|---|
| 8.1 | The grid has 1,024 combinations, truncated to the first 5. `ParameterGrid` iterates keys alphabetically with the last key fastest, so the 5 trials vary only `num_layers` and the learning rate. | Medium |
| 8.2 | The grid key `dropout_rate` is read as `model_params.get('dropout')`, so dropout is never tuned. `batch_size` in the grid is ignored because loaders are built with `config.batch_size`. | Medium |
| 8.3 | `optuna` is imported (a hard dependency) and `setup_optuna_study`/`objective` are defined but never used. | Low |
| 8.4 | `predict_with_model_optimized` uses `model_info['model'].device`, but `nn.Module` has no `.device` attribute, so inference raises `AttributeError`. | High |

## 9. Ensemble methodology

| # | Finding | Severity |
|---|---|---|
| 9.1 | The "ensemble" is three runs of the same grid search with different seeds, weighted by `1/val_loss`. Validation losses are near-identical, so the weighting is effectively arbitrary. There is no model diversity. | Medium |
| 9.2 | Ensemble predictions are generated on `X_targets_dict`, the **same data used for training**. | High |

## 10. Uncertainty estimation

| # | Finding | Severity |
|---|---|---|
| 10.1 | "Uncertainty" is the variance across 3 seed-models. Seed disagreement is not a predictive distribution and was never shown to be calibrated. | High |
| 10.2 | `calibrate_uncertainty` uses the last 20% of the *training* data (in-sample). It computes a scale factor that is **never applied**. Its logic (`scale = coverage` if coverage ≥ 0.95) shrinks or expands in the wrong direction. | High |
| 10.3 | The README recommends sizing positions and setting stop-losses from this uncertainty. Nothing in the code supports that. | High |

## 11. Evaluation metrics

| # | Finding | Severity |
|---|---|---|
| 11.1 | No rank-based metric is implemented. The competition score (mean / std of daily cross-sectional Spearman IC) is absent. | High |
| 11.2 | MSE, MAE and R² are computed on **price-level** targets, so they are meaningless for return forecasting. | High |
| 11.3 | "Direction accuracy" compares `np.diff` of predictions with `np.diff` of targets, i.e. day-over-day *changes* of predicted price levels, not the sign of returns. | Medium |

## 12. Runtime, reproducibility and code quality

| # | Finding | Severity |
|---|---|---|
| 12.1 | Seeds are set, but `DataLoader(shuffle=True)` has no generator, there is no deterministic mode, and outputs depend on whether CUDA is present. | Low |
| 12.2 | `warnings.filterwarnings('ignore')` hides real problems, and broad `except Exception: continue` blocks swallow errors per target. | Medium |
| 12.3 | There are no tests, no configuration files, no logging and no dependency specification. | Medium |
| 12.4 | Outputs are written to the working directory (`processed_data/`, `ensembles/`, `corrected_ensemble_predictions.csv`) without provenance. | Low |

---

## What was kept

- The **research intent**: multi-asset features, cross-asset correlations, a sequence model,
  an ensemble and uncertainty. All of these were re-implemented correctly or replaced
  (see `docs/INDEPENDENT_CONTRIBUTIONS.md`).
- The **feature ideas** of multi-window rolling statistics and market-regime indicators
  (`market_volatility_20`, `market_trend_20`). They reappear, computed causally, as the
  `own` and `regime` feature groups.
- The **raw data files** (moved, not modified, to `data/raw/mitsui/`).
- The original code, verbatim, in `legacy/original/`. It expects the CSV files in the working
  directory and depends on `optuna`. It is retained for provenance only and is not part of
  the active codebase.
