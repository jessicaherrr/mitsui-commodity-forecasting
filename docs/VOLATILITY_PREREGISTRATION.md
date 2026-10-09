# Pre-registration: volatility research phase

**Written:** 2026-10-09, after the estimation-sample diagnostics (Part I) and **before** any
volatility-forecast evaluation or any return-forecasting run with volatility features.

**Why this document exists.** The MITSUI final holdout (date_id 1461–1960) was already used in
the first research phase to evaluate other models. Re-using it for new hypotheses is defensible
only if the hypotheses, models, feature sets, metrics and decision rules are fixed *before*
the new models see holdout data. This file fixes them. Any later deviation is listed in §7.

## 1. Research question

> Do volatility dynamics provide incremental predictive information for commodity-linked
> returns, and does their usefulness vary across market regimes?

Volatility can matter in two distinct ways, and they are tested separately:

- **Volatility as an object of prediction.** Is volatility forecastable, and by which model?
  This is evaluated on volatility proxies, separately from returns.
- **Volatility as an input to return forecasting**, through two channels:
  - **normaliser channel:** scaling labels and forecasts by a volatility forecast;
  - **predictor channel:** volatility features added to return/rank features.

## 2. Evidence already seen (estimation sample, rows 0–1460)

- Jarque–Bera rejects normality for all 106 assets.
- Ljung–Box on r² and ARCH-LM reject for all 106.
- Ljung–Box on r rejects for 79%.
- GJR-vs-GARCH LR rejects for 74% after BH-FDR (q = 0.10).
- Engle–Ng joint sign/size bias rejects for 31% after FDR.

**Decision rule (set before the diagnostics ran):** include an asymmetric model if at least 20%
of assets reject symmetry after FDR. Outcome: **GJR-GARCH(1,1)-t is included.** EGARCH is not
added. GJR is nested in GARCH (so a clean LR test applies) and has closed-form multi-step
forecasts; a second asymmetric specification would add multiplicity without a distinct
hypothesis.

## 3. Hypotheses

### Part II: volatility forecasting (106 assets; forecasts re-estimated every 21 trading days, expanding window)

Models: `const`, `hv21`, `hv63`, `ewma` (λ = 0.94), `garch_n`, `garch_t`, `gjr_t`. No tuned
hyper-parameters.

| ID | Hypothesis | Test |
|---|---|---|
| V1 | Volatility dynamics are meaningful out of sample: every dynamic model has lower QLIKE than `const` | panel Diebold–Mariano on cross-sectionally averaged QLIKE differences (HAC); per-asset DM shares |
| V2 | GARCH(1,1)-t beats EWMA at 1-, 5- and 21-day horizons | panel DM; per-asset DM; Model Confidence Set (MCS) inclusion shares |
| V3 | GJR-t beats GARCH-t for US equities (leverage) but not for FX/metals | panel DM by market |
| V4 | Student-t estimation beats Gaussian QML for forecasting | panel DM `garch_t` vs `garch_n` |
| V5 | Relative model performance differs by volatility regime | panel DM within causal volatility terciles |

- **Proxies.** The primary proxy is the sum of squared daily returns over the forecast window,
  which is conditionally unbiased; QLIKE and MSE are Patton-robust under it. The secondary
  proxy is Garman–Klass range variance for assets with OHLC (US, JPX). It is biased (no
  overnight move), so it is a sensitivity check only.
- **Primary loss:** QLIKE $L = \log f + \tilde\sigma^2/f$, which is valid for zero proxies.
  Secondary: MSE. Also reported: Mincer–Zarnowitz regressions.

### Part III: return forecasting with volatility information (target level, 424 targets)

**Model.** Target-level pooled LightGBM, one model per horizon over that horizon's 106
targets. It is fixed *a priori* at 7 leaves, min_child_samples 2000, 200 trees, learning rate
0.03, subsample 0.7, colsample 0.8, λ₂ = 5. A target-level Ridge (α chosen on development from
{1e3, 1e4, 1e5}) serves as the linear benchmark.

**Labels.** Spread return scaled by the spread's EWMA volatility × √h, demeaned within horizon
on each date (identical in every predictor-channel arm). Forecasts are rescaled by the same
scale.

**Feature groups**, all causal, computed from the target's own log price ratio and its legs:

- **R (return/rank):**
  - lagged observable labels $\ell_t-\ell_{t-h}$ and $\ell_{t-h}-\ell_{t-2h}$;
  - raw spread returns over 1/5/21/63 days;
  - deviation of the price ratio from its 20-day mean;
  - within-horizon cross-sectional ranks of the lagged label and of the 5/21/63-day returns.
- **V (volatility):**
  - vol-normalised versions of the R signals and the 63-day spread z-score;
  - the GARCH-t standardised last shock;
  - log GARCH-t forecast of the label-window variance;
  - log EWMA vol;
  - realised-vol ratio (5/63);
  - GARCH volatility term structure (forecast vs long-run variance);
  - GARCH variance innovation;
  - vol-of-vol;
  - own-history volatility percentile;
  - within-horizon rank of spread volatility;
  - GARCH-t vol ratio of the two legs;
  - GARCH persistence;
  - market volatility state (VT volatility percentile, volatility change).
- **C (cross-asset):** leg A minus leg B of the factor exposures, home/equity correlations,
  group-relative and factor-residual momentum; weekly factor returns; trend and
  cross-correlation regime variables.

| ID | Hypothesis | Comparison (same model, dates, targets) |
|---|---|---|
| R1 (primary) | Volatility features add information without cross-asset features | LGBM[R+V] − LGBM[R] |
| R2 (primary) | Volatility features add information on top of cross-asset features | LGBM[R+V+C] − LGBM[R+C] |
| R3 (primary) | A better volatility forecast is a better normaliser | LGBM[R] with GARCH-t scaling − LGBM[R] with EWMA scaling |
| R4 | Normalising matters at all | LGBM[R], EWMA scaling − LGBM[R], no scaling |
| R5 | Volatility *state* vs volatility *normalisation* | LGBM[R+Vstate] − LGBM[R]; LGBM[R+V] − LGBM[R+Vstate] |
| R6 | The volatility-feature gain is larger in high-volatility regimes | ΔIC(R+V − R) by causal market-volatility tercile and by target-volatility tercile |
| R7 | Volatility features improve *stability* | IC standard deviation, share of positive-IC days, worst 126-day rolling mean IC |
| R8 | Results are model-agnostic | the same R1/R2 contrasts with target-level Ridge |

## 4. Metrics and inference

- **Primary metric:** daily Spearman IC across all 424 targets (competition definition), its
  mean and the score mean/std. Also reported: per-horizon IC (106 targets each).
- **Inference:** paired daily ΔIC with Newey–West HAC t and circular block bootstrap 95% CI
  (block 20).
- **Multiple comparisons:**
  - **Holm** adjustment across the three primary tests R1–R3 (family-wise α = 0.05).
  - **Model Confidence Set** (α = 0.10, stationary bootstrap) across all Part III arms, with
    loss = −daily IC.
  - **Hansen SPA test** of "some volatility arm beats LGBM[R]".
- Volatility forecasts use the MCS at α = 0.10 per asset. Shares across assets are
  reported; per-asset DM p-values are FDR-corrected (BH, q = 0.10).

## 5. Samples

- **Development:** walk-forward OOS, date_id 630–1460, refit every 126 dates. All development
  choices are restricted to: the Ridge α; bug fixes; and computational settings.
- **MITSUI holdout:** date_id 1461–1960, evaluated **once** for Part II and once for Part III
  after this pre-registration and the development runs. The prior use of this holdout for
  other models (first phase) is acknowledged; its general behaviour, e.g. the one-month
  reversal, was known when the feature list above was written from the literature and
  competition practice.
- **Independent data:** FRED/EIA daily energy spot prices plus the broad USD index.
  - 2010–2025 history: replication of Parts I–II, and of R1/R3 with configuration-generated
    targets.
  - **2026-01-01 → latest available:** never examined by anyone in this project. It is the
    *fresh* confirmation sample for the volatility-forecast hypotheses V1–V4.

## 6. What would count as support

- **Volatility forecasting.** V1 is supported if the panel DM rejects at 5% in both
  development and holdout for every dynamic model vs `const`. V2–V4 need the panel DM sign to
  agree in development and holdout, plus significance in the holdout.
- **Return forecasting.** A primary hypothesis is supported only if its holdout ΔIC > 0 with
  Holm-adjusted p < 0.05 **and** the development ΔIC has the same sign. A positive development
  result that fails the holdout is reported as "not replicated".

## 7. Deviations (filled in during the study)

1. **Bug fix in multi-step GARCH forecasts (before any Part III result was examined).** The
   first engine version capped persistence at 0.9999 and forecast a *flat* variance path for
   integrated estimates. That affected 2.2% of GARCH-t and 3.0% of GJR-t refits. A unit test
   against `arch`'s analytic forecasts caught it, and the engine now uses the exact recursion
   E[s_{k+1}] = w + phi E[s_k]. The Part II evaluation had already been run once with the
   faulty engine (results inspected). It was re-run with the corrected engine, and the
   corrected numbers are the ones reported. All Part III arms were re-trained after the fix.
2. **C-feature selection made dataset-agnostic** (name patterns instead of a hard-coded MITSUI
   list), so the FRED replication can build the C group. On MITSUI it selects exactly the same
   17 features in the same order (verified), so MITSUI results are unaffected.
3. **Exploratory, post-hoc profile `vol_explore`** (added after the holdout evaluation, prompted
   by the feature importance): it decomposes the volatility-state gain into market-level vs
   target-own state. It is not pre-registered and is reported as exploratory.
4. **Engineering fixes with no effect on reported numbers:**
   - holdout diagnostics accept short samples (≥ 100 observations) so the 197-day FRED 2026
     holdout can be described;
   - figure captions and representative-series selection made dataset-agnostic;
   - joblib worker pools are released after each engine run.

## 8. Freeze record (2026-10-09 09:36, before any Part III holdout evaluation)

**Development results seen** (`results/vol/vol/returns_*_dev.csv`):

- R1 (LGBM[R+V] − LGBM[R]): +0.018 [0.002, 0.034], t = 2.04 (Holm p = 0.125).
- R2 (with cross-asset features): +0.006 (n.s.).
- R3 (GARCH-t vs EWMA normaliser): +0.004 (n.s.).
- R4 (EWMA vs no normalisation): −0.011 (t = −2.03).
- R5: volatility state alone +0.019 (t = 2.17); vol-normalised signals add nothing beyond state.
- R8: Ridge gains nothing from volatility features.
- R6 direction: the R1 gain sits in mid/high market-volatility regimes and is negative in the
  low regime.
- SPA (development): p = 0.033.

No modelling change follows from these numbers.

**Robustness specifications (fixed now, run after the freeze; `configs/vol_robust.yaml`).**
Arms tgbm_R, tgbm_RV, tgbm_RVs under:

1. faster refits (every 42 dates instead of 126);
2. a rolling 750-date training window instead of an expanding one;
3. a higher-capacity LightGBM (15 leaves, min_child_samples 1000).

Also reported from the main run:

- R1/R5 ΔIC by horizon group;
- holdout first half vs second half;
- the FRED replication (development 2015–2025; fresh holdout 2026).
