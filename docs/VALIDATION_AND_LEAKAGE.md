# Validation protocol and leakage controls

Time-series validation is where most financial-ML results go wrong. This document states the
information timing precisely, shows how the splits implement it, lists the automated tests
that guard it, and discloses every decision made after looking at out-of-sample data.

## 1. Information timing

All positions are rows of the chronologically sorted `date_id` index. One row is one
date on which at least one market traded.

```
decision date t                          label window (horizon h, gap g = 1)
      │                                  ┌────────────────────────────┐
──────●───────────────●──────────────────●────────────────────────────●──────▶ date_id
      t              t+1                t+1                         t+1+h
  forecast formed    first price of the label            last price of the label:
  from data ≤ t      (the t → t+1 return is NOT          label observable at the
                     part of any label)                  close of t+1+h
```

| Quantity | Available at the close of |
|---|---|
| features for decision date $t$ (all markets' closes on `date_id` ≤ $t$) | $t$ |
| horizon-$h$ label for decision date $t$ | $t + g + h = t + 1 + h$ (2–5 rows later) |

Two consequences matter:

- **Labels, not only features, have an availability date.** A model refitted at $\tau$
  may use the label of decision date $t$ only if $t + 1 + h \le \tau$. The last 2–5 decision
  dates before every refit have features but **no usable labels**.
- **The 1-row gap is economically important.** LME, JPX, US and FX close at different clock
  times on the same `date_id`, and daily closes contain microstructure noise. The
  competition's label starts at $t+1$, which removes the next-day return. The timing
  diagnostic (EXPERIMENT_RESULTS §8) shows that the sample's strongest simple daily effect,
  1-day reversal of LME/JPX/FX closes (holdout IC −0.071, $t=-4.6$ on the gap-0 return),
  almost disappears under the gap-1 definition (−0.015, $t=-1.0$).

Verified against the data (`python -m commodity_research.cli audit`):

- **Label formula.** It reproduces all 424 official labels with max abs error ≤ 1e-6.
  Plausible misreadings fail: no gap (correlation −0.13 at lag 1), price-spread changes
  (0.63–0.98), and the legacy price-level target (≈ 0).
- **`test.csv`.** It is a verbatim copy of `train.csv` rows 1827–1960 with no labels, so it is
  never used for evaluation.

## 2. Splits

```
date_id   0 ── 130 ─────────────── 630 ─────────────────────────── 1461 ─────────────── 1960
          warm-up │  first training  │  DEVELOPMENT walk-forward     │  FINAL HOLDOUT
                  │  window          │  (831 OOS dates)              │  (500 OOS dates)
                  │                  │  model selection, ensemble    │  touched once, after
                  │                  │  weights, conformal quantiles │  the design was frozen
inferred calendar ~2018-01           ~2020-06                         ~2023-08            ~2025-06
```

- **Walk-forward, expanding window.** Refits happen at $\tau = 630, 630+R, \dots$ with
  $R = 126$ (FAST) or 42 (FULL) rows. Each fitted model predicts the decision dates
  $[\tau, \tau+R)$ and is then discarded. No observation is ever predicted by a model that
  saw its label.
- **Horizon-specific purge by label availability** (`validation/splits.py: train_dates`): for
  horizon $h$, training decision dates are $[\text{first}, \tau - 1 - h]$. Training labels use
  prices up to $\tau$; test labels use prices from $\tau+1$. Their information intervals never
  overlap, so no extra embargo is needed for a backward-looking window. The purge is derived
  from each target's actual information interval, not from an arbitrary 1-day gap.
- **Early stopping inside the training window** (`internal_validation`). The validation block
  is the last 126 decision dates whose labels are observable at $\tau$. The fit block contains
  only labels observable **before** the validation block starts, so the early-stopping signal
  is itself purged.
- **No shuffling across time.** The LSTM shuffles mini-batches *within* its training window
  only; nothing crosses a fold boundary. There is no K-fold.
- **Holdout labels in training.** Within the holdout walk-forward, a refit at $\tau$ trains on
  every label observable at $\tau$, including earlier holdout dates. That is exactly what a
  live system would do. No hyper-parameter, feature, member set or weight is chosen with
  holdout data.

## 3. Train-only transformations

| Transformation | Fitted on | Applied to |
|---|---|---|
| Feature winsorisation bounds, mean, std (`TrainOnlyScaler`) | training decision dates of the refit (shortest horizon) | all dates predicted by that refit |
| Ridge / LightGBM / LSTM parameters | training (and purged validation) rows | that fold's test block |
| Pair-model scaler and coefficients | same rule | same |
| Ensemble scales $s_{m,h}$, static and regime weights | development OOS forecasts | development + holdout |
| Online ensemble weights | ICs of decision dates $\le s-1-h_{\max}$ | date $s$ |
| Conformal interval quantiles | development OOS residuals | holdout |

Features themselves contain no fitted parameters. They are rolling, EWMA or expanding
statistics of past prices, and cross-sectional ranks within the same date. Forward-filling of
prices uses past values only (at most 5 rows). There is **no** interpolation, back-filling,
centred smoothing, wavelet denoising or full-sample normalisation anywhere in the active
pipeline. All of those were present in the legacy code (LEGACY_AUDIT §4).

Volatility-regime labels use an **expanding** percentile of past VT volatility. A full-sample
tercile would leak the future distribution of volatility.

## 4. Automated anti-leakage tests (`tests/`)

| Test | What it proves |
|---|---|
| `test_causality.py::test_features_unchanged_by_future_prices_{synthetic,mitsui}` | Randomly distorting all prices after row $t_0$ leaves every feature, volatility, regime variable and activity flag at rows $\le t_0$ **bit-identical**, on synthetic and real data. |
| `test_causality.py::test_predictions_unchanged_by_future_data` | For hist_mean, reversal, ridge, LightGBM, both direct ridges and the LSTM: a model fitted at $\tau$ produces identical forecasts for dates $\tau..t_0$ after all prices (hence features and labels) after $t_0$ are distorted. |
| `test_causality.py::test_unobservable_labels_are_not_used` | Replacing every label that is *not yet observable* at $\tau$ ($t > \tau-1-h$) with noise leaves the fitted model unchanged, for every learned model. |
| `test_causality.py::test_observable_labels_are_used` | Control: changing *observable* labels does change the fit, so the previous test is not vacuous. |
| `test_causality.py::test_scaler_ignores_test_period_features` | Setting all test-period feature values to 1e6 leaves the fitted scaler unchanged. |
| `test_splits.py` | Folds partition the test range; for every fold and horizon $t+g+h \le \tau$, train and test dates are disjoint, and nothing observable is wasted; the internal validation block is purged. |
| `test_models.py::test_sequences_only_use_past_rows` | LSTM input windows end at $t$, contain no future row and no other asset. |
| `test_models.py::test_online_ensemble_weights_only_use_observable_ics` | Online weights at $s$ are unchanged when not-yet-observable ICs are altered. |
| `test_models.py::test_same_seed_same_predictions` | Ridge, LightGBM and LSTM are reproducible for a fixed seed (tolerance 1e-7). |
| `test_metrics.py` | The daily IC equals `scipy.stats.spearmanr`; the score matches an independent re-implementation of the official metric; degenerate dates are excluded and counted, never scored favourably; Newey–West and bootstrap behave as expected. |
| `test_data.py::test_official_labels_reproduced`, `test_composed_asset_labels_match_official`, `test_test_csv_is_not_independent` | Real-data checks of the target definition, the composition of asset forecasts into pairs, and the `test.csv` overlap. |

Run them with `python -m pytest` (all tests) or `python -m pytest -m "not slow"`.

## 5. Statistical inference under dependence

- Daily ICs of $h$-day labels overlap. Consecutive labels share $h-1$ daily returns, and the
  competition cross-section mixes $h=1..4$. Mean-IC t-statistics therefore use **Newey–West
  HAC** errors (Bartlett kernel, automatic lag of about 5–6).
- 95% CIs for the mean IC, the IC information ratio (the competition score) and paired IC
  differences use a **circular moving-block bootstrap** (block 20 dates).
- Model comparisons are **paired** (same dates, same targets): $\overline{IC_a - IC_b}$ with
  HAC t and bootstrap CI.
- There is **no multiple-testing adjustment**. About 17 paired comparisons in 2 periods and
  about 20 models are reported, so isolated $p<0.05$ results are exploratory. With 17 holdout
  model scores, the chance that at least one has $|t|>2.5$ under the null is substantial.

## 6. Researcher degrees of freedom: full disclosure

The protocol's guarantee is that **no choice was made using holdout performance of the
evaluated models**. These are the decisions made *after* looking at out-of-sample data, in
chronological order:

1. **Cross-sectional label demeaning** (§1.2 of MODEL_METHODOLOGY), adopted after the first
   FAST run showed negative **development** ICs for Ridge with raw vol-scaled labels. The
   raw-label variant is kept and reported as `ridge_rawlabel`. That first run was discarded
   because its code changed mid-run; its development log is archived in
   `results/archive/fast_v0_rawlabel_dev_log.txt`.
2. **A univariate feature diagnostic** printed holdout-period ICs of single features while I
   investigated (1). No feature was added, removed or transformed afterwards. The feature set
   was fixed before that diagnostic ran. This was a protocol slip and is disclosed here.
3. **A second ensemble group** (`ens_best_*`: members with positive *development* score) and
   the **timing diagnostic** were added after the development results and before any holdout
   evaluation. The member rule uses development scores only.
4. **The FULL profile** (`configs/full.yaml`) was written *after* the FAST holdout
   evaluation. It changes only compute settings (refit every 42 instead of 126 rows, larger
   grids, 3 LSTM seeds, 1000 bootstrap reps), chosen as more expensive versions of FAST rather
   than in response to any holdout number. Its holdout results are reported in full alongside
   FAST, and neither profile's holdout was used to pick between them.
5. **Holdout-only exploratory analyses** in the research report are labelled as such.
   Examples are the per-target direct ridge's holdout result and the regime and category
   breakdowns.

Everything else (target definition, gap, horizons, splits, feature families, model classes,
grids, ensemble schemes, metrics) was fixed before the first holdout evaluation:
`evaluate --profile fast` at 03:09 on 2026-10-09, after the design freeze. Evaluation was
later re-executed with the **same trained models**, and every metric came out identical:

- after making table labels dataset-agnostic for the generic adapter;
- in a clean from-scratch reproduction (`rm -rf results/fast && cli all --profile fast`),
  whose metrics and predictions were bit-identical to the first run (max abs difference 0).

No model, feature or hyper-parameter changed after 03:09.

## 7. Phase 2 (volatility research): additional controls

- **Label availability, now independently verified.** The competition's own lagged-label
  files (from a public mirror of the evaluation gateway, read as data only) give
  `date_id − label_date_id = lag + 1` for all 360 rows, and the values match `train_labels.csv`
  exactly. This is precisely the availability rule $t + 1 + h \le \tau$ used for every purge in
  this project. It also means the competition's "lagged labels" are exactly the
  `lag_label` = $\ell_t - \ell_{t-h}$ feature, computable from prices at $t$.
- **Causal volatility forecasts.**
  - GARCH-family parameters are re-estimated every 21 trading days on returns up to the
    re-estimation date.
  - Between refits, the variance recursion uses realised returns up to the forecast date only.
  - Tests: a future-perturbation test of all seven volatility models; exact agreement with
    reference recursions (HV, EWMA, constant); exact agreement of multi-step forecasts with
    `arch`'s analytic forecasts. That last test caught a real bug (flat forecasts for
    integrated estimates), which was fixed before any Part III result was examined.
- **Causal target features.** A future-perturbation test covers all 47 target-level features
  and both volatility scales. A label-availability test confirms that the target-level
  LightGBM and Ridge fits ignore every label not yet observable at the refit date.
- **Estimation-sample discipline.** Diagnostics that informed modelling choices (e.g.
  including GJR) used rows 0–1460 only.
- **Pre-registration and freeze.** See
  [VOLATILITY_PREREGISTRATION.md](VOLATILITY_PREREGISTRATION.md).
  - Hypotheses, arms, tests and the Holm family were fixed before evaluation.
  - Development-only results were recorded before the holdout was opened.
  - Robustness specifications were fixed in the freeze record.
  - Four deviations are listed there. One is a post-hoc exploratory decomposition, labelled
    as such everywhere it appears.
- **Fresh data.** The FRED 2026 sample (197 dates) is the only data in the project never
  examined before its hypotheses were written.
- **Multiplicity in Phase 2.**
  - Holm across the three primary return hypotheses.
  - Model Confidence Set and Hansen SPA across all target-level arms.
  - BH-FDR for per-asset diagnostic and DM tests.
  - Panel-level (cross-sectionally averaged) DM tests, so a few assets cannot drive a
    "significant" volatility-forecast result.
