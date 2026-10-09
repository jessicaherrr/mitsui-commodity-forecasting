# Regime-Aware Multi-Asset Return Prediction and Risk-Aware Signal Analysis

*Commodity Forecasting & Cross-Asset Quantitative Research. An independent research extension of
the MITSUI & CO. Commodity Prediction Challenge.*

---

## Abstract

I study whether historical commodity, equity and FX market information can rank the
forward returns of 424 commodity-linked targets: 4 single-asset log returns and 420
log-return spreads, each pairing an LME base metal or JPX precious-metal future with a US
equity/ETF, an FX rate or another commodity, at horizons of 1–4 days.

The legacy competition code was audited and found unusable: it trained on random synthetic
data, used price-level targets and leaked future information. I rebuilt the pipeline around
three ideas:

- a verified target definition with explicit label-availability timing;
- an *asset-factorised* formulation that forecasts 106 asset returns and composes the 424
  targets linearly;
- a strict development/holdout protocol with walk-forward refits and horizon-specific purging.

Benchmarks, pooled Ridge, LightGBM, a multi-horizon LSTM, direct pair models and four
forecast-combination schemes are compared on the same dates and targets. Dependence-robust
inference uses Newey–West t-statistics and block bootstrap.

On a 500-date final holdout (≈ 2023-08 to 2025-06), **none of the pre-specified
asset-level learned models (Ridge, LightGBM, LSTM) or ensembles produced a competition-style
score statistically distinguishable from zero**. The single significant model, a per-target
pair Ridge (score 0.20, t = 3.1), had failed on development; its holdout result is explained
by pair-level persistence specific to 2023–25. Development-period leaders decayed.
Cross-asset features helped on the development period (Ridge: +0.023 mean IC, p = 0.06) but
not on the holdout. Accuracy differs across volatility regimes: linear and ensemble models
do better in high-volatility states. Regime-aware weighting did not beat equal weighting in
the FAST profile, and gave only a weak, non-significant gain in FULL (+0.007, p = 0.09).

The most stable signal is the simplest: the trailing one-year mean return ranks the 106
*single assets* with IC ≈ 0.025–0.04 in both periods. The competition's mixed-horizon pair
cross-section dilutes it, and only its low turnover lets it survive a few basis points of
trading cost.

Two further findings apply beyond this dataset:

- the competition's one-day gap removes the largest daily effect in the data, a 1-day
  reversal consistent with closing-price noise or short-lived overreaction (gap-0 IC −0.071,
  t = −4.6, vs −0.015 with the gap);
- ensemble disagreement carries no information about forecast error.

The FULL profile (3× faster refits, larger grids, 3-seed LSTM) reproduces these conclusions
(§6.8).

> **Phase 2.** A follow-up study asks whether volatility dynamics carry incremental
> information for these returns. It covers GARCH-family forecasting, econometric diagnostics,
> volatility features in target-level LightGBM, robustness checks and a replication on fresh
> data. See [VOLATILITY_AND_DIAGNOSTICS.md](VOLATILITY_AND_DIAGNOSTICS.md).

---

## 1. Introduction

Commodity prices respond to global growth, the dollar, real rates, inventories and supply
shocks. Commodity-producer equities and commodity currencies (AUD, CAD, NOK, ZAR, NZD) embed
some of the same information. The MITSUI challenge packages this into a forecasting problem:
using about 7.5 years of daily LME, JPX, US-equity and FX data, rank 424 future returns and
return spreads each day.

Kaggle leaderboards reward the best score on one test period. A research question asks
something different: *is there a stable, economically interpretable source of predictability,
and does modelling sophistication earn its keep once the evaluation is honest?* This report
addresses the second question. Its contributions are as much methodological (timing, label
availability, dependence-aware inference, disclosure) as empirical.

## 2. Financial motivation

Three mechanisms could make these targets predictable:

1. **Persistent risk premia and momentum.** Commodities and equities show time-series and
   cross-sectional momentum at 3–12 month horizons (Moskowitz, Ooi & Pedersen, 2012; Asness,
   Moskowitz & Pedersen, 2013), and short-horizon reversal (Lehmann, 1990).
2. **Cross-asset information transmission.** Commodity currencies and producer equities
   respond to commodity news, and markets close at different times (Tokyo → London →
   New York). Information in one market's close can therefore be news for another market's
   next close.
3. **Relative-value mean reversion.** The log price ratio of economically linked assets
   (copper vs miners, gold vs gold miners) may mean-revert, as in classic pairs trading.

Each mechanism maps to a feature family (§4.3). If predictability exists, it should be
regime-dependent: momentum crashes in volatility spikes (Daniel & Moskowitz, 2016), and
cross-asset correlations jump in crises.

## 3. Research questions and hypotheses

| | Question | Hypothesis tested | Experiment |
|---|---|---|---|
| RQ1 | How accurately can historical commodity, equity and FX information predict future commodity-related returns? | $H_1$: out-of-sample mean daily rank IC > 0 for learned models | A, E |
| RQ2 | Do cross-asset relationships improve forecasts vs own-history models? | $H_2$: $IC(\text{all features}) - IC(\text{own features}) > 0$ | B |
| RQ3 | Do models perform differently across volatility regimes? | $H_3$: mean IC differs between low/mid/high-volatility regimes | C |
| RQ4 | Can regime-aware combination and uncertainty estimation improve stability and usefulness? | $H_{4a}$: regime/IC/online weighting beats equal weighting. $H_{4b}$: disagreement predicts error. $H_{4c}$: intervals are calibrated | D, risk layer |

These are hypotheses, not conclusions. Each is confronted with the holdout below.

## 4. Data and methodology

### 4.1 Data

| | |
|---|---|
| Source | Kaggle MITSUI & CO. Commodity Prediction Challenge training files (not redistributed) |
| Dates | 1,961 `date_id` rows (anonymised; inferred ≈ 2018-01 to 2025-06, ~260 rows/year; the largest moves at date_id 566–586 match March 2020, and 1882–1885 match April 2025) |
| Instruments | 557 raw columns. Close series: 4 LME base metals (Al, Cu, Pb, Zn), 6 JPX contracts (gold, platinum, rubber), 95 US stocks/ETFs (adjusted closes), 38 FX rates |
| Missing data | Exchange holidays: all closes missing on 51 (LME), 116 (JPX), 67 (US) dates; FX complete |
| Targets | 106 per lag, lags 1–4. Every pair has a commodity leg: 268 commodity–equity, 132 commodity–FX, 20 commodity–commodity, plus 4 single assets (2 equity ETFs, 2 FX) |
| `test.csv` | A verbatim copy of `train.csv` rows 1827–1960 without labels (the competition API mock), so it is not a test set |

### 4.2 Target definition and timing

Reproduced to $\le 10^{-6}$ for all 742,640 label cells:

$$
y_{j,t} = \underbrace{\log\frac{P^{A}_{t+1+h}}{P^{A}_{t+1}}}_{r^A_{t,h}} - \underbrace{\log\frac{P^{B}_{t+1+h}}{P^{B}_{t+1}}}_{r^B_{t,h}}, \qquad h \in \{1,2,3,4\}.
$$

These are **return spreads**, i.e. changes in the log price ratio $\log(P^A/P^B)$. They are
not price spreads $P^A-P^B$, which are unit-dependent and which the legacy code used. A
forecast for date $t$ is formed at the close of $t$. Its label is observable only at the
close of $t+1+h$, so a model refitted at $\tau$ may train on decision dates
$t \le \tau-1-h$ only.

### 4.3 Feature construction

38 causal, scale-free features per asset and date, in four groups (full definitions in
[MODEL_METHODOLOGY.md](MODEL_METHODOLOGY.md) §2):

- **own** (14): risk-adjusted momentum at 5/21/63 days, 126-21 momentum, moving-average
  deviations, trend efficiency, last returns, EWMA volatility, volatility ratio, downside
  ratio, skewness, kurtosis;
- **cross** (14): rolling-beta-weighted daily moves of six factors (global equity VT, USD,
  LME metals, gold, Treasuries, energy), correlations with the home factor and global equity,
  relative volatility, weekly factor returns;
- **relative** (6): cross-sectional ranks of momentum and volatility, momentum relative to the
  market group, factor-residual momentum;
- **regime** (4): expanding-percentile equity volatility, volatility shock, trend strength,
  cross-factor correlation.

### 4.4 Models

| Level | Model | Notes |
|---|---|---|
| A | `hist_mean`, `momentum_21`, `reversal_5` | no fitting |
| A | `ridge` (pooled per horizon) | α from {1e3, 1e5, 1e6} on development |
| B | `lgbm` (pooled per horizon) | 2 restrained configurations |
| C | `lstm` (multi-horizon, 20-day window, 32 units) | early stopping on a purged block |
| — | `ridge_direct_pooled`, `ridge_direct_target` | pair-level comparators |
| D | `ens_{equal, static_ic, regime, online}` over {ridge, lgbm, lstm}; `ens_best_{equal, regime}` over development-positive models | weights from development data, or observable trailing ICs |

Asset-level models learn $\tilde z_{i,t,h}$, the vol-scaled return demeaned across assets
on date $t$, and target forecasts are $\hat y_j = \hat r^{A_j} - \hat r^{B_j}$.

## 5. Experimental design and validation protocol

- **Splits** (row positions): warm-up 0–129 · development walk-forward OOS 630–1460 (831
  dates, ≈ 2020-06 to 2023-08) · **final holdout 1461–1960** (500 dates, ≈ 2023-08 to 2025-06).
- **Walk-forward**: expanding window, refit every 126 dates (FAST) or 42 (FULL). For each
  horizon, train on labels observable at the refit (purge of $1+h$ rows). Early stopping uses
  a purged internal block.
- **Selection**: hyper-parameters, ensemble members/weights and conformal quantiles come
  from the development period only. The holdout was evaluated once after the design freeze.
  Decisions made after seeing out-of-sample data are disclosed in
  [VALIDATION_AND_LEAKAGE.md](VALIDATION_AND_LEAKAGE.md) §6.
- **Metric**: on each date, the Spearman correlation between the 424 forecasts and realised
  labels; score $= \overline{IC}/\operatorname{sd}(IC)$ (ddof = 0), as in the competition. Also
  reported: per-horizon IC (106 targets), asset-level IC (106 assets), paired IC differences,
  directional accuracy and rank stability.
- **Inference**: Newey–West HAC t-statistics; 95% CIs from a circular block bootstrap
  (block 20); no multiple-testing correction, so isolated significant results are
  exploratory.
- **Leakage tests**: 62 automated tests, including bit-identical features and forecasts under
  perturbation of future prices, and invariance of each fitted model to labels not yet
  observable.

## 6. Empirical results (FAST profile)

All numbers are from `reports/fast/results_tables.md`, generated by the CLI.

### 6.1 RQ1: overall accuracy (Experiment A)

| Model | Dev score | **Holdout score** [95% CI] | Holdout mean IC [95% CI] | t (HAC) |
|---|---|---|---|---|
| hist_mean | 0.092 | 0.024 [−0.118, 0.153] | 0.0059 [−0.029, 0.037] | 0.34 |
| momentum_21 | −0.018 | **−0.178** [−0.334, −0.036] | −0.0479 [−0.085, −0.010] | −2.65 |
| reversal_5 | 0.076 | 0.004 [−0.123, 0.126] | 0.0010 [−0.034, 0.033] | 0.05 |
| ridge | 0.045 | −0.083 [−0.220, 0.049] | −0.0223 [−0.057, 0.014] | −1.18 |
| lgbm | −0.037 | 0.019 [−0.092, 0.143] | 0.0039 [−0.019, 0.030] | 0.29 |
| lstm | 0.044 | −0.031 [−0.165, 0.105] | −0.0073 [−0.038, 0.025] | −0.47 |
| ens_equal | 0.027 | −0.048 [−0.193, 0.096] | −0.0120 [−0.046, 0.024] | −0.66 |
| ens_best_equal | 0.101 | −0.053 [−0.190, 0.082] | −0.0129 [−0.045, 0.020] | −0.79 |
| ridge_direct_target | −0.034 | 0.203 [0.058, 0.366] | 0.0303 [0.009, 0.052] | 3.05 |

*Interpretation.*

- **Typical skill is near zero.** Out-of-sample rank ICs are about one order of magnitude
  below 0.1. Daily IC volatility (≈ 0.25) dwarfs the mean, so even a true mean IC of 0.02
  would need several years to establish at conventional significance.
- **Model class did not help.** No learned model differs significantly from `hist_mean` on
  the holdout. Among learned models, LightGBM − Ridge = +0.026: its bootstrap CI
  [0.0004, 0.054] barely excludes zero, but the HAC test does not reject (t = 1.45,
  p = 0.15), and the sign was reversed on development (−0.020).
- **The sign of fitted signals is unstable.** The development leaders (hist_mean,
  reversal_5, ridge, LSTM) all lost most or all of their development IC. One-month momentum
  was significantly *negative* on the holdout (t = −2.65): short-term reversal dominated
  2023–25.
- **The one significant positive model was not a pre-selected candidate.**
  `ridge_direct_target` ranked poorly on development. Its holdout performance is reproduced
  by its intercept alone, a target-level expanding mean: holdout IC +0.039 (t = 2.79), but
  −0.010 on development (§7). In plain terms, specific pairs kept out- or under-performing
  in 2023–25 (a regime-specific persistence), and with about 17 models evaluated, one
  t ≈ 3 is weak evidence.

**Verdict on $H_1$: not supported** for the competition metric. Learned models did not
deliver reliably positive out-of-sample IC on the 424-target cross-section.

### 6.2 Experiment E: horizon and universe

| Holdout mean IC | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| hist_mean, 106 **single assets** | 0.025 | 0.030 | 0.025 | 0.026 |
| hist_mean, 106 **targets** per horizon (105 pairs + 1 single) | 0.016 | 0.016 | 0.017 | −0.003 |
| lgbm, single assets | −0.002 | 0.016 | 0.017 | 0.029 |
| ridge, targets | −0.007 | −0.015 | −0.026 | −0.032 |
| momentum_21, targets | −0.027 | −0.043 | −0.051 | −0.056 |

- **The one robust regularity.** The trailing one-year mean return ranks the 106 single
  assets consistently: asset-level IC is 0.026–0.040 on development and 0.025–0.030 on the
  holdout, at every horizon.
- **Dilution in the competition metric.** Pair labels add a second leg, and the six commodity
  instruments matter as much as the hundred counter-legs combined (§7). The mixed-horizon
  cross-section adds a further penalty, because ranking 4-day spreads against 1-day spreads
  rewards getting the most volatile targets right. The overall hist_mean score falls to 0.024.
- **Horizon profiles.** Signal quality does not decay monotonically with horizon. Momentum's
  negative IC strengthens with horizon, and LightGBM's asset-level IC rises with horizon.
  Longer horizons average out daily noise, but overlapping labels mean fewer independent
  observations, so HAC errors widen accordingly.

### 6.3 RQ2: cross-asset information (Experiment B)

| Paired difference (all − own features) | Development | Holdout |
|---|---|---|
| Ridge | +0.023 [−0.001, 0.050], t = 1.87 | −0.000 [−0.031, 0.029], t = −0.02 |
| LightGBM | +0.009 [−0.012, 0.033], t = 0.71 | −0.006 [−0.032, 0.021], t = −0.40 |

Cross-asset, relative-value and regime features improved both model classes on the
development period, marginally significantly for Ridge. The improvement did **not** carry
over to the holdout. In the Ridge fitted at the holdout start, cross-asset variables have
some of the largest standardised coefficients:

- correlation with the home factor: +0.021
- beta-weighted Treasury move: +0.017
- beta-weighted USD move: +0.013

LightGBM's highest gain shares go to the regime variables and weekly factor returns. The
models *use* cross-asset information, but its out-of-sample value is not established.

**Verdict on $H_2$: not supported** (development evidence only, which did not replicate).

### 6.4 RQ3: volatility regimes (Experiment C)

Mean daily IC by causal volatility regime, all out-of-sample dates (development and holdout
combined, for power):

| Model | low (398 dates) | mid (569) | high (364) |
|---|---|---|---|
| ridge | −0.027 | −0.005 | **+0.032** |
| ens_equal | −0.013 | −0.004 | +0.020 |
| lstm | −0.006 | +0.006 | +0.012 |
| hist_mean | −0.001 | +0.041 | +0.007 |
| reversal_5 | −0.010 | +0.033 | +0.010 |
| momentum_21 | −0.034 | −0.029 | +0.005 |

- **Most models do best when global equity volatility is high and worst when it is low.**
  This is consistent with cross-asset dispersion being more informative in stressed markets.
  It holds for most models across all out-of-sample dates, but not uniformly: LightGBM is
  the exception, and in the holdout alone (66 high-volatility dates) the pattern is weaker
  (ridge 0.000 high vs −0.035 low).
- **Correlation regime.** Learned models also did better in high cross-factor-correlation
  regimes (ridge +0.023 vs −0.016; LSTM +0.026 vs −0.009, all OOS).

**Verdict on $H_3$: partially supported, descriptively.** Accuracy differs by regime in a
consistent direction for the linear and ensemble models. Regime-level samples are small and
the differences are not tested for significance.

### 6.5 RQ4a: regime-aware combination (Experiment D)

| Paired difference vs `ens_equal` | Development (in-sample weights) | Holdout |
|---|---|---|
| ens_static_ic | +0.003 | −0.004 [−0.009, 0.001] |
| ens_regime | +0.005 | −0.002 [−0.007, 0.003] |
| ens_online (causal) | +0.003 | −0.001 [−0.008, 0.007] |
| ens_best_regime vs ens_best_equal | **+0.016 (t = 2.60)** | +0.003 [−0.013, 0.019] |

- Every weighting scheme looked better than equal weighting on the development period, where
  the weights were estimated. None did on the holdout. This is the "forecast combination
  puzzle": estimated weights lose to equal weights when skill is low and unstable.
- Combining also did not beat the best single benchmark: `ens_best_equal` − `hist_mean` =
  −0.019 on the holdout.

**Verdict on $H_{4a}$: not supported in FAST.** The FULL profile gives weak, non-significant
support for regime weighting (+0.007, p = 0.09; §6.8).

### 6.6 RQ4b/c: uncertainty and risk-aware signals

**Model disagreement** (std of scale-normalised member forecasts) is positively associated
with market volatility (cross-sectional Spearman +0.30 holdout). It is **not** associated with
larger vol-scaled errors: Spearman −0.019 holdout and −0.028 development. Mean |error|/σ
actually *falls* from 0.82 to 0.75 across disagreement quintiles.

**Verdict on $H_{4b}$: rejected.** Disagreement here measures how differently models extrapolate
volatile assets, not forecast reliability.

**Prediction intervals** (split-conformal on vol-scaled development residuals, `ens_equal`,
holdout):

| Nominal 90% | low vol | mid vol | high vol | all |
|---|---|---|---|---|
| vol-scaled interval coverage | 83.1% | 85.6% | 85.2% | 84.4% |
| constant-width interval coverage | 88.5% | 90.0% | 85.6% | 88.7% |
| average width, vol-scaled (bps) | 546 | 597 | 810 | 601 |

Coverage is almost identical for every model, because forecasts are tiny relative to return
volatility: predictive uncertainty is essentially market volatility. Vol-scaled intervals
adapt their width to the regime (546 → 810 bps) and keep coverage more even across regimes.
However, they under-cover by about 6 points: holdout returns were fatter-tailed relative to
EWMA volatility than development returns.

**Verdict on $H_{4c}$: not calibrated at the nominal level.** Adaptive conformal updating is the
natural fix (FUTURE_RESEARCH §5).

**Hypothetical portfolio** (106 single assets, horizon 1, dollar-neutral, enter at close
$t+1$, holdout, gross of costs unless stated):

| Signal | Sharpe 0 bps | Sharpe 2 bps | Sharpe 5 bps | Daily turnover |
|---|---|---|---|---|
| hist_mean | 0.32 | 0.23 | 0.09 | 0.10 |
| lgbm | 0.84 | −0.16 | −1.67 | 0.95 |
| ens_equal | 0.70 | −0.14 | −1.39 | 0.75 |
| ridge | −0.22 | −1.06 | −2.30 | 0.77 |

With 500 days, the standard error of an annualised Sharpe ratio is ≈ 0.7, so none of these
is distinguishable from zero. The economically important pattern is turnover: only the
slow-moving historical-mean signal keeps a positive Sharpe at 5 bps. Vol-scaled weights
outperformed raw-forecast weights (ens_equal gross Sharpe 0.70 vs 0.22) because risk
scaling stops the most volatile names from dominating the book. These are paper diagnostics:
they ignore financing, borrow, FX conversion of JPY- and GBP-quoted futures, contract rolls
and asynchronous closes.

### 6.7 Information timing: why the one-day gap matters

| Signal (39 LME/JPX/FX assets, h = 1) | Gap-0 label (t → t+1) | Gap-1 label (competition) |
|---|---|---|
| own last return `ret_1_z`, development | −0.037 (t = −3.3) | −0.014 (t = −1.3) |
| own last return `ret_1_z`, holdout | **−0.071 (t = −4.6)** | −0.015 (t = −1.0) |
| equity-beta × today's VT move, holdout | −0.001 | −0.007 |

The strongest simple daily effect in the data is next-day reversal of LME/JPX/FX closes. It is
consistent with noise in closing prints (bid-ask bounce, fixing and settlement effects) or
short-lived overreaction. The competition's one-day gap removes almost all of it. The
hypothesised lead-lag from US equities to earlier-closing markets was **not** found: a
beta-weighted VT move has no predictive IC at either gap. A model that accidentally used a
gap-0 label would look skilful for the wrong reason, which is why the audit verified the
gap before anything else.

### 6.8 Robustness: FULL profile

The FULL profile refits every 42 instead of 126 dates and uses larger grids, a 3-seed LSTM
and 1,000 bootstrap reps. It was specified as a compute-only variant and reported regardless
of outcome. Holdout scores: ridge −0.084, lgbm −0.009, lstm −0.061, ens_equal −0.067,
ens_regime −0.040, hist_mean 0.024. All 95% CIs include zero, except ridge_direct_target
(0.202), whose explanation is unchanged.

Conclusions under FULL:

- **RQ1 and RQ2 unchanged.** Ridge's cross-asset gain is again development-only (+0.020,
  t = 1.58; holdout −0.000).
- **RQ3 sharper.** Ridge's all-OOS IC is +0.033 in high-volatility and −0.033 in
  low-volatility regimes; the LSTM's is +0.024 vs −0.017.
- **RQ4a weakly positive, not replicated.** Regime-conditional weighting beat equal weighting
  on the holdout by +0.007 [−0.000, 0.016] (t = 1.69, p = 0.09). That is suggestive, but not
  significant, and the FAST profile showed −0.002. It is a hypothesis for new data, not a
  finding.
- **Paper P&L.** The 3-seed LSTM had the highest gross Sharpe (1.46). That is one of nine
  signals, without a significant IC, and it falls to 0.07 at 5 bps.

Details: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md#full-profile-robustness).

## 7. Robustness and exploratory analysis (post hoc, not used for selection)

- **Effective breadth.** An oracle that knew the realised returns of only the 6 commodity
  instruments would score a daily IC of 0.59 (development) / 0.62 (holdout). One that knew
  only the 100 counter-legs would score 0.57 / 0.55. The 424 targets therefore carry far
  fewer independent bets than their count suggests. Forecasting six commodities well is
  worth as much as forecasting a hundred equities and currencies, which partly explains the
  high day-to-day volatility of the IC.
- **Target-level persistence.** An expanding mean of each target's own (vol-scaled,
  cross-sectionally demeaned) label has IC −0.010 (t = −0.8) on development and +0.039
  (t = 2.8) on the holdout. This is the source of `ridge_direct_target`'s holdout result: in
  2023–25 some commodity-vs-X relationships trended persistently, and earlier they had not.
  Descriptively, pairs with a JPX gold leg moved on average +8.1 bps per day (per unit
  horizon) in gold's favour on the holdout vs +1.7 on development, and JPX platinum +5.5 vs
  −0.5. These contracts are yen-denominated, so this mixes the precious-metal rally with yen
  weakness. It is a regime feature of the holdout, not a stable law.
- **Label design.** Demeaning the vol-scaled labels improved Ridge on development (+0.036,
  t = 2.9) and was adopted for that reason. On the holdout the difference is +0.007 (n.s.):
  the choice mattered for avoiding volatility tilts in development, but neither variant has
  holdout skill.
- **Time stability.** Rolling 126-date mean ICs swing between about −0.10 and +0.10 for every
  model (figure 02). Positive and negative episodes last months, the signature of time-varying
  risk premia and crowded or uncrowded trades rather than estimation noise alone.

## 8. Limitations

1. **Sample length.** 1,961 dates and a 500-date holdout cannot resolve mean ICs of about
   0.01–0.02. Power, not only model quality, limits every conclusion.
2. **One holdout path.** Results depend on the 2023–25 environment: strong gold, rate
   volatility, the 2025 tariff shock. CPCV or multiple holdouts would give a distribution.
3. **Anonymised dates.** Calendar effects and macro-release timing cannot be studied, and the
   calendar mapping is inferred.
4. **Instrument coverage.** There are no futures curves (carry, basis), inventories or
   positioning data, the best-documented commodity predictors.
5. **Mixed asynchronous closes.** Same-`date_id` prices are not simultaneous. The gap-1
   design neutralises most consequences, but cross-market features are still measured at
   different clock times.
6. **Researcher degrees of freedom.** Disclosed in VALIDATION_AND_LEAKAGE §6. One diagnostic
   displayed holdout feature ICs; no decision followed from it.
7. **No multiple-testing adjustment.** About 20 models and 17 pre-specified comparisons are
   reported unadjusted.
8. **Hypothetical P&L.** The backtest is a signal diagnostic and not tradable as specified.

## 9. Conclusion

Rebuilt with verified targets, label-availability-aware validation and honest inference, the
MITSUI problem shows **very little stable predictability** at the 1–4 day horizon once the
competition's one-day gap removes microstructure reversal.

- Complex models (gradient boosting, LSTM) and regime-aware ensembles did not outperform
  simple benchmarks on an untouched holdout.
- Cross-asset information and regime-aware weighting looked useful during development and
  did not replicate.
- The most stable finding is economically familiar: assets' trailing one-year drift
  persistently ranks their near-term returns. It is a slow, low-turnover, momentum-like
  premium, and it is diluted by the pair and mixed-horizon construction of the competition
  metric.

For a practitioner, the lessons are about process:

- verify target timing before modelling;
- purge by label availability;
- evaluate the cross-section the metric actually scores;
- prefer slow signals when costs exist;
- treat development-period gains from added complexity as hypotheses until a holdout
  confirms them.

## References

- Asness, C., Moskowitz, T., Pedersen, L. (2013). Value and Momentum Everywhere. *Journal of Finance* 68(3).
- Bailey, D., López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5).
- Daniel, K., Moskowitz, T. (2016). Momentum Crashes. *Journal of Financial Economics* 122(2).
- Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies* 33(5).
- Lehmann, B. (1990). Fads, Martingales, and Market Efficiency. *Quarterly Journal of Economics* 105(1).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley (purging, embargo, CPCV).
- Moskowitz, T., Ooi, Y. H., Pedersen, L. (2012). Time Series Momentum. *Journal of Financial Economics* 104(2).
- Newey, W., West, K. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica* 55(3).
- Politis, D., Romano, J. (1994). The Stationary Bootstrap. *JASA* 89(428).
- Smith, J., Wallis, K. (2009). A Simple Explanation of the Forecast Combination Puzzle. *Oxford Bulletin of Economics and Statistics* 71(3).
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World* (conformal prediction).
