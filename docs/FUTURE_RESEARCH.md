# Future research roadmap

Everything below is **proposed, not implemented**, unless marked otherwise. Each item builds on
what the experiments showed: predictability in this panel is weak and unstable, simple
slow-moving signals travel best, and complexity has not paid for itself.

## Already implemented (for reference)

Asset-factorised forecasting; causal feature families; label-availability walk-forward;
benchmarks, Ridge, LightGBM and LSTM; direct pair models; four ensemble schemes;
split-conformal intervals; disagreement diagnostics; hypothetical cost-aware backtest; generic
adapter with a FRED energy demonstration.

## 1. Broader commodity datasets

- **Futures term structure.** Front/second-month spreads (roll yield, basis) are the
  best-documented commodity return predictors (basis/carry and inventory theory). They are
  absent from the MITSUI panel. The generic adapter already accepts additional instruments;
  the missing pieces are a roll-adjusted continuous-contract builder and a carry feature
  family.
- **More markets.** Agricultural (CBOT grains, softs), energy futures (WTI/Brent/natural gas)
  and LME inventories/cancelled warrants, via public sources with documented redistribution
  terms (EIA, USDA, exchange end-of-day files).
- **Longer history.** Seven years contains one crisis (2020) and one tightening cycle (2022).
  Regime conclusions need multiple episodes of each kind.

## 2. Concept drift and non-stationarity

The development-to-holdout decay of every signal is the main empirical fact of this project.
Next steps:

- **Rolling vs expanding windows.** The code supports `split.window`; a systematic
  comparison of window lengths is not yet run.
- **Drift detection.** Two candidates: CUSUM or Page-Hinkley tests on the realised IC series;
  and adversarial validation (can a classifier tell training-period features from recent
  ones?) to detect covariate shift before the IC deteriorates.
- **Time-decayed sample weights** in the pooled regressions (exponential weighting of older
  dates).

## 3. Cross-market transfer learning

- **Pre-train and fine-tune.** Pre-train the pooled asset-level model on a large equity/ETF
  universe and fine-tune on commodities. This tests whether generic return dynamics transfer
  across markets.
- **Asset embeddings.** Learned asset embeddings in the LSTM/MLP to share strength between
  similar instruments, regularised toward the hand-built metadata groups.

## 4. Time-varying factor structures

- Replace fixed equal-weight factors with **rolling PCA / IPCA**-style latent factors
  estimated causally. The aim is to forecast factor returns and idiosyncratic returns
  separately.
- **Dynamic conditional correlation (DCC-GARCH)** for the correlation-regime variable, instead
  of a rolling 63-date average |ρ|.

## 5. Probabilistic forecasting

- **Quantile regression** (LightGBM quantile objective, quantile LSTM heads) and **CRPS**-based
  evaluation, to replace point forecasts plus conformal intervals.
- **Conditional conformal prediction.** Locally weighted residual scores or regime-stratified
  calibration. Holdout coverage of the vol-scaled intervals was 84% at a nominal 90%, so the
  calibration set should be refreshed online (adaptive conformal inference).

## 6. Risk-aware commodity allocation

- A proper allocation study on **tradable futures**: roll-adjusted returns, contract
  multipliers, margin, exchange calendars and closing times, FX conversion of JPY- and
  GBP-denominated contracts, and realistic transaction-cost and impact models.
- **Mean-variance or risk-parity overlays** using the forecasts as tilts, with turnover
  penalties. The backtest shows signal turnover, not signal sign, is the binding constraint:
  most gross edges vanish at 2–5 bps.
- **Forecast-horizon matching.** Hold horizon-$h$ forecasts for $h$ days with overlapping
  sub-portfolios (Jegadeesh–Titman style) to cut turnover.

## 7. Economic interpretation of predictive factors

- Formal **factor-attribution** of the historical-mean signal: is the asset-level persistence
  (IC ≈ 0.025–0.04 in both periods) time-series momentum, a carry or risk-premium proxy, or
  an artefact of the mixed-asset universe (bonds vs miners)?
- **Shapley-value** attribution for LightGBM across regimes, to see whether regime features
  act through interactions or merely add noise.
- **Pair economics.** Every one of the 420 pair targets has a commodity leg, so the task is
  "commodity vs X" relative value. Studying which counter-leg types (equity sector, currency
  bloc) carry signal could connect results to commodity-currency and commodity-equity
  literatures.

## 8. Volatility research follow-ups (from Phase 2)

- **Realised variance from intraday data.** This would replace the noisy daily squared-return
  proxy and raise the power of the volatility-forecast comparison by an order of magnitude.
  HAR-RV models (Corsi 2009) would then be the natural benchmark.
- **Implied volatility and variance risk premia.** Commodity options (e.g. OVX, GVZ, CME
  options) carry the best-documented forward-looking volatility information. The variance
  risk premium is a known return predictor that daily price data cannot provide.
- **Shrinking GARCH forecasts toward the unconditional variance.** Mincer–Zarnowitz slopes
  below 1 (0.57–0.69 at 1 day) say current forecasts over-react.
- **Regime adaptation vs volatility state.** The rolling-window result suggests that part of
  what volatility features contributed was adaptation. A direct test would compare
  volatility-conditioned models with explicitly time-weighted ones.
- **Target-own volatility state as a pre-registered hypothesis** on new data. It was the only
  component positive in both MITSUI periods (exploratory, §4.4).

## 9. Methodology and inference

- **Multiple-testing control.** Use the Romano–Wolf stepdown or the Hansen Superior
  Predictive Ability test across the model set, instead of unadjusted paired tests.
- **Combinatorially purged cross-validation (CPCV)** on the development period, to estimate
  the distribution of out-of-sample scores rather than a single path.
- **Deflated Sharpe ratio** for the hypothetical portfolios.
