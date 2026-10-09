# Volatility Dynamics and Commodity Return Predictability

*Financial time-series diagnostics, volatility forecasting and robustness analysis.
Phase 2 of the Commodity Forecasting & Cross-Asset Quantitative Research project.*

> **Research question.** Do volatility dynamics provide incremental predictive information for
> commodity returns, and does their usefulness vary across market regimes?

All numbers in this document come from executed code. Machine-generated tables:

- MITSUI: [`reports/vol/vol_tables.md`](../reports/vol/vol_tables.md)
- robustness: [`reports/vol_robust`](../reports/vol_robust/vol_tables.md), [`reports/vol_rolling`](../reports/vol_rolling/vol_tables.md)
- exploratory: [`reports/vol_explore`](../reports/vol_explore/vol_tables.md)
- FRED replication: [`reports/vol_fred/vol_tables.md`](../reports/vol_fred/vol_tables.md)

The hypotheses, tests and decision rules were fixed in
[`VOLATILITY_PREREGISTRATION.md`](VOLATILITY_PREREGISTRATION.md) before any evaluation. Two
deviations and the freeze record are listed there.

---

## Summary of findings

| Question | Answer | Strength of evidence |
|---|---|---|
| Are daily returns of LME/JPX/US/FX instruments Gaussian and serially independent? | No. Fat tails everywhere: JB rejects for 106/106; median excess kurtosis 12.3 (US), 7.9 (JPX), 3.3 (FX), 1.9 (LME) | Decisive |
| Is volatility clustered and forecastable? | Yes. Ljung–Box on r² rejects for 100% of assets in the estimation sample and 82% in the holdout (after FDR). Every GARCH-family model beats constant variance out of sample at 1, 5 and 21 days in both periods (panel DM t from −2.6 to −8.0) | Decisive, replicated |
| Are *returns* linearly predictable from their own past? | Only in a crisis. Ljung–Box on r rejects for 4% of assets pre-COVID, 67% in the COVID window, 13% post-crash and 12% in the holdout | Decisive: return autocorrelation is a crisis artefact; volatility clustering is not |
| Which volatility forecast is best? | GARCH(1,1)-t beats EWMA at 1 and 5 days on the holdout (t = −2.47, −2.41), with the same sign on development; 21 days is marginal (t = −1.87). The gain comes from FX and US equities, not LME metals | Supported (pre-registered criterion met at 1 and 5 days) |
| Does asymmetry (leverage) help forecasting? | In-sample yes, out of sample no. GJR rejects GARCH for 74% of assets in-sample, yet GJR-t never beats GARCH-t out of sample (all \|t\| < 2). Equities show a leverage effect (VT γ = 0.24); gold shows *inverse* asymmetry (GLD γ = −0.05) | In-sample asymmetry does not translate into forecast gains |
| Do volatility *features* improve return forecasts (target-level LightGBM)? | Directionally yes, statistically not confirmed. ΔIC +0.018 (development, t = 2.04), +0.013 (holdout, t = 1.44, Holm p = 0.45); pooled out of sample +0.016 (t = 2.46, secondary) | Pre-registered primary test not met; consistent sign |
| Through which channel? | Volatility **state**, used non-linearly. State-only features carry the whole gain; vol-normalised signals add nothing; Ridge gains nothing. The trees split most on the market volatility percentile, but post hoc, only the target's *own* volatility state helped in both periods | Channel clear; the market-regime part is fragile (exploratory) |
| Does a better volatility forecast make a better *normaliser*? | No. GARCH-t vs EWMA label scaling: +0.004 (development), +0.0015 (holdout), both n.s. Normalising by EWMA *hurts* rank-IC vs raw returns on MITSUI development (t = −2.03) and on FRED (pooled t = −2.24) | Not supported; normalisation can hurt rank metrics |
| Do the results replicate on independent data (FRED energy, fresh 2026 sample)? | Volatility forecasting: yes. Dynamics beat constant variance, GARCH-t beats EWMA (2026, 1d: t = −2.28), GJR is worse out of sample. Return forecasting: the volatility-feature gain does not replicate (R1 = −0.005 / −0.006), on a 20-target panel with low power | Volatility results replicate; return result does not |
| Is the return-forecast gain robust to specification? | Same sign under faster refits and higher-capacity trees (pooled t ≈ 2.3). It disappears on the holdout with a rolling 750-date window, where the baseline adapts on its own | Fragile |
| Does the usefulness vary across regimes? | Yes, but not monotonically. The gain is in *mid*-volatility regimes (t = 2.65 development, 2.36 holdout); it is negative in calm development markets and in the 66 high-volatility holdout days. It is largest for the *lowest-volatility targets* in both periods | Regime dependence real; "more useful in high volatility" rejected |

---

## 1. Data and sample discipline

- **MITSUI panel:** 1,961 daily rows (inferred ≈ 2018-01 to 2025-06); 106 instruments that
  form the 424 targets (4 LME base metals, 2 JPX precious-metal futures, 67 US stocks/ETFs,
  33 FX rates); 424 target spread series.
- **Estimation sample:** date_id 0–1460. Every statistic that informed a modelling decision
  (e.g. whether to include an asymmetric GARCH) was computed here.
- **Development walk-forward:** 630–1460. **Holdout:** 1461–1960. The holdout was used in
  phase 1 for other models; this phase evaluated its new models on it once, after a
  pre-registration and a freeze.
- **Independent data:** FRED/EIA daily spot prices (WTI, Brent, Henry Hub, NY Harbor heating
  oil and gasoline) and the Fed broad USD index, 2010-01 to 2026-10. The **2026 rows (197
  dates) were never examined before the pre-registration** and serve as a fresh holdout.
- **Returns:** $r_t = \log P_t - \log P_{\text{last trading day}}$ on each instrument's own
  trading days. Holidays are skipped, not zero-filled.

## 2. Part I: stylised facts and diagnostics

### 2.1 Tests

Writing $r_t = \sigma_t z_t$:

- **Jarque–Bera:** $JB = \tfrac n6\big(S^2 + \tfrac{K^2}{4}\big) \sim \chi^2_2$ (S = skewness, K = excess kurtosis).
- **Ljung–Box:** $Q(m) = n(n+2)\sum_{k\le m}\hat\rho_k^2/(n-k)$, applied to $r_t$ (linear
  predictability) and $r_t^2$ (volatility clustering); $\chi^2_{m - \text{model\_df}}$.
- **ARCH-LM:** $nR^2$ from regressing $r_t^2$ on $r_{t-1}^2,\dots,r_{t-5}^2$ (Engle 1982).
- **Engle–Ng sign/size bias**, on GARCH standardised residuals:
  $z_t^2 = c_0 + c_1 S^-_{t-1} + c_2 S^-_{t-1}z_{t-1} + c_3 S^+_{t-1}z_{t-1} + u_t$,
  joint $H_0: c_1=c_2=c_3=0$ (HC1 covariance).
- **GARCH residual diagnostics:** Ljung–Box on $z_t$ and on $z_t^2$ (df − 2 for the two
  dynamic parameters, Li–Mak), ARCH-LM on $z_t$, Jarque–Bera on $z_t$, and an LR test of
  GJR vs GARCH (one restriction).
- **Multiplicity:** per-series p-values are summarised as the share of series rejecting
  after Benjamini–Hochberg FDR control at q = 0.10.

### 2.2 Results (estimation sample unless stated)

| Test | Assets (106) | Target spreads (424) | Assets, holdout |
|---|---|---|---|
| Jarque–Bera | 100% | 100% | 96% |
| Ljung–Box $r$, 10 lags (BH) | 82% | 77% | 12% |
| Ljung–Box $r^2$, 10 lags (BH) | 100% | 100% | 82% |
| ARCH-LM, 5 lags (BH) | 100% | 100% | 83% |

**The return-autocorrelation result is entirely a crisis effect** (Ljung–Box, share rejecting
after BH):

| Window (rows) | returns $r_t$ | squared returns $r_t^2$ |
|---|---|---|
| pre-COVID (130–539) | 3.8% | 60.4% |
| COVID window (540–699) | 67.0% | 88.7% |
| post-crash development (700–1460) | 13.2% | 82.1% |
| holdout (1461–1960) | 12.3% | 82.1% |

Volatility clustering is present in every window. Linear return predictability appears only
when the March 2020 crash and rebound are in the sample.

![ACF of returns vs squared returns](../reports/vol/figures/v01_acf_returns_vs_squared.png)

**GARCH(1,1)-t fits** (estimation sample, medians by market):

| Market | n | Excess kurtosis | Persistence α+β | Half-life (days) | ν (Student-t) | GJR γ | GJR LR rejects |
|---|---|---|---|---|---|---|---|
| US equities/ETFs | 67 | 12.3 | 0.979 | 33 | 6.4 | +0.077 | 84% |
| FX | 33 | 3.3 | 0.939 | 11 | 7.2 | +0.041 | 52% |
| JPX precious | 2 | 7.9 | 0.982 | 38 | 6.0 | −0.012 | 0% |
| LME base metals | 4 | 1.9 | 0.980 | 37 | 11.0 | +0.019 | 50% |

Half-life $=\ln 0.5/\ln(\alpha+\beta)$.

- **Residual diagnostics.** After GARCH-t filtering, squared-residual autocorrelation remains
  for only 18% of assets (BH) and ARCH-LM for 6%, so GARCH(1,1) captures most of the
  clustering. Residual normality is still rejected for 100%, which justifies Student-t
  innovations: the median log-likelihood gain over Gaussian is 34 points for US and FX.
- **Asymmetry is economically interpretable.**
  - Equity volatility rises after falls (leverage, VT γ = 0.24).
  - Gold volatility rises after *rises* (GLD γ = −0.048, JPX gold −0.029). This
    inverse-leverage pattern is consistent with safe-haven buying during stress.

![Q-Q plots](../reports/vol/figures/v02_qq_plots.png)

**Decision (pre-registered rule: include an asymmetric model if at least 20% of assets reject
symmetry after FDR).** 74% reject (LR) and 31% (Engle–Ng), so GJR-GARCH(1,1)-t is included.
EGARCH is not added: GJR is nested in GARCH (clean LR test) and has closed-form multi-step
forecasts, and a second asymmetric model would only add multiplicity.

## 3. Part II: volatility forecasting (out of sample)

### 3.1 Models and causality

| Model | One-step variance forecast $f_{t+1\mid t}$ |
|---|---|
| const | expanding mean of $r^2$ (no dynamics) |
| hv21, hv63 | mean of the last 21/63 $r^2$ |
| ewma | $\lambda f_t + (1-\lambda) r_t^2$, λ = 0.94 (an IGARCH with ω = 0) |
| garch_n / garch_t | $\omega + \alpha r_t^2 + \beta f_t$, Gaussian QML / Student-t ML |
| gjr_t | $\omega + (\alpha + \gamma\,\mathbb 1[r_t<0]) r_t^2 + \beta f_t$, Student-t |

- **Multi-step forecasts** use the exact recursion
  $E_t[\sigma^2_{t+k+1}] = \omega + \phi\,E_t[\sigma^2_{t+k}]$ with $\phi = \alpha+\beta+\gamma/2$.
  This is valid even for integrated estimates; it fixed a bug found by unit testing against
  `arch` (deviation #1).
- **Re-estimation:** parameters are re-estimated every 21 trading days on an expanding window
  (at least 250 observations) and frozen in between.
- **Causality:** the variance recursion is filtered with realised returns only up to the
  forecast date. A unit test perturbs all future returns and checks that every past forecast
  is bit-identical.

### 3.2 Loss functions and proxies

$$
\mathrm{QLIKE}(s, f) = \log f + \frac{s}{f}, \qquad \mathrm{MSE}(s,f) = (s-f)^2 .
$$

- **Why this QLIKE form.** It equals $s/f - \log(s/f) - 1$ up to a term that depends only on
  the proxy, so model rankings are identical, but it remains finite when $s = 0$ (a zero
  return).
- **Patton (2011) robustness.** Both losses rank forecasts correctly when the proxy is
  *conditionally unbiased* for the true variance.
- **Why QLIKE differences can be averaged across assets.** They are scale-free: multiplying
  $f$ and $s$ by $c$ leaves $\log(f_1/f_2)$ and $s/f$ unchanged.

**Proxies, and why daily data limits volatility evaluation.**

1. **Squared returns** (summed over the window): conditionally unbiased but extremely noisy.
   For a 1-day horizon, $r^2_{t+1} = \sigma^2_{t+1} z^2_{t+1}$ has a signal-to-noise ratio
   of about $1/\mathrm{Var}(z^2) \approx 1/(2+\kappa)$. That is why even a perfect forecast
   explains only a few percent of the proxy's variance (Andersen & Bollerslev 1998). The
   Mincer–Zarnowitz R² here is ≈ 0.03 at 1 day.
2. **Garman–Klass range variance** (US and JPX instruments with OHLC): far less noisy, but
   **biased downward** because it ignores the overnight (close-to-open) move. A biased proxy
   can reorder models, so it is a sensitivity check only.
3. **Missing proxy.** Intraday realised variance, the gold standard, is not available.

### 3.3 Inference

- **Per-asset Diebold–Mariano:** $d_t = L_{a,t} - L_{b,t}$ with Newey–West standard errors.
  Lags of 5/10/42 at 1/5/21 days, because multi-day windows overlap.
- **Panel DM:** average $d_t$ across assets on each date first, then the same HAC t-test.
  This uses all 106 assets while absorbing their cross-sectional correlation.
- **Model Confidence Set** (Hansen, Lunde & Nason 2011) per asset: 90%, range statistic,
  stationary bootstrap.
- **Mincer–Zarnowitz:** $s_t = a + b f_t + u_t$ (HAC), testing $a = 0, b = 1$.

### 3.4 Results

Mean QLIKE(model) − QLIKE(EWMA) across 106 assets (negative = better):

| Model | dev 1d | dev 5d | dev 21d | **holdout 1d** | **holdout 5d** | **holdout 21d** |
|---|---|---|---|---|---|---|
| const | +0.106 | +0.087 | +0.064 | +0.142 | +0.097 | +0.048 |
| hv21 | +0.043 | +0.043 | +0.043 | +0.049 | +0.046 | +0.055 |
| hv63 | +0.015 | +0.008 | +0.000 | +0.033 | +0.017 | +0.001 |
| ewma | 0 | 0 | 0 | 0 | 0 | 0 |
| garch_n | +0.001 | −0.004 | −0.007 | −0.032 | −0.038 | −0.048 |
| garch_t | −0.007 | −0.010 | −0.014 | −0.030 | −0.038 | −0.053 |
| gjr_t | −0.002 | −0.001 | −0.014 | −0.029 | −0.037 | −0.049 |

Pre-registered panel DM tests (t-statistics; negative = model A more accurate):

| Hypothesis | A vs B | dev 1d | dev 5d | dev 21d | holdout 1d | holdout 5d | holdout 21d | Verdict |
|---|---|---|---|---|---|---|---|---|
| V1 | garch_t vs const | −7.45 | −7.99 | −6.56 | −6.84 | −5.99 | −2.73 | **supported** (all GARCH variants) |
| V1 | ewma vs const | −5.70 | −5.34 | −3.67 | −4.71 | −2.69 | −0.74 | supported at 1–5d only |
| V1 | hv21 vs const | −2.97 | −2.17 | −1.01 | −2.80 | −1.40 | +0.15 | supported at 1d only |
| V2 | garch_t vs ewma | −1.55 | −1.93 | −2.12 | **−2.47** | **−2.41** | −1.87 | **supported at 1d and 5d** |
| V2 | garch_t vs hv21 | −6.29 | −6.38 | −6.56 | −4.28 | −4.55 | −4.21 | supported |
| V3 | gjr_t vs garch_t | +0.88 | +1.09 | +0.19 | +0.10 | +0.06 | +1.42 | **not supported** |
| V4 | garch_t vs garch_n | −6.32 | −5.26 | −5.15 | +0.75 | +0.07 | −1.31 | **not replicated** (development only) |

![QLIKE vs EWMA](../reports/vol/figures/v04_qlike_vs_ewma.png)

**Interpretation.**

- **Dynamics matter (V1).** Constant variance loses decisively. The 21-day historical window
  is too noisy and slow: it is worse than EWMA and, at 21 days, no better than constant
  variance.
- **Mean reversion is what GARCH adds over EWMA (V2).** EWMA is an integrated model: its
  multi-step forecast is flat. GARCH forecasts revert toward the unconditional variance, so its
  advantage grows with horizon and is largest just after volatility spikes. In the high-vol
  holdout regime, 21-day panel t = −3.61. The gain is concentrated in FX (holdout 1d t = −2.13)
  and US equities (−2.39). LME metals show no difference.
- **In-sample asymmetry did not help forecasting (V3).** The extra parameter is estimated with
  noise, and the asymmetric response is already partly absorbed by the high persistence.
  Under the (biased) Garman–Klass proxy GJR *is* better (t ≈ −2.2 to −3.4) and EWMA beats
  GARCH-t. The direction of those intraday-range results is plausible: leverage effects are
  strongest in intraday ranges. But they also show that conclusions can flip with a biased
  proxy, which is why the squared-return proxy is primary.
- **Fat-tailed estimation helps only when estimation windows contain extreme shocks (V4).**
  The post-COVID development period is such a case; Gaussian QML over-reacts to 2020's
  outliers. In the calmer holdout the advantage disappears.
- **Power is a panel phenomenon.** Per asset, GARCH-t beats EWMA significantly (after BH) for
  at most 4% of assets. The MCS keeps most models for most assets (inclusion 74–98% for
  ewma/garch/gjr) because one-day proxies are so noisy. Only the cross-sectionally pooled
  panel DM has enough power.
- **Calibration.** Mincer–Zarnowitz slopes are below 1 (median b = 0.57–0.69 at 1 day,
  0.17–0.39 at 21 days): forecasts move too much relative to subsequent realised variance.
  Unbiasedness is rejected for 29–36% of assets at 1 day on the holdout.

![MCS inclusion](../reports/vol/figures/v05_mcs_inclusion.png)
![Forecast paths](../reports/vol/figures/v06_vol_forecast_paths.png)

## 4. Part III: does volatility information improve *return* forecasts?

### 4.1 Design

**Model.** A target-level pooled LightGBM, one model per horizon over that horizon's 106
targets (≈ 130k training rows per horizon at the last refit). Its settings were fixed *a
priori* (7 leaves, min_child 2000, 200 trees, learning rate 0.03), so its development-period
forecasts are genuinely out of sample. A target-level Ridge (α selected on development) is
the linear benchmark.

**Feature groups.** For target $j$ with log price ratio $\ell_t$:

- **R (11): return/rank.**
  - lagged observable labels $\ell_t-\ell_{t-h}$ and $\ell_{t-h}-\ell_{t-2h}$, the inputs the
    competition's 8th-place team built on;
  - spread returns over 1/5/21/63 days;
  - deviation of $\ell$ from its 20-day mean;
  - within-horizon cross-sectional ranks of the above ("historical rank features").
- **Vn (7): volatility-normalised versions of the R signals.** The R signals scaled by the
  spread's EWMA vol, the 63-day z-score of $\ell$, and the GARCH-t standardised last shock.
- **Vs (12): volatility state.**
  - log GARCH-t forecast of the *label-window* variance;
  - log EWMA vol;
  - realised-vol ratio (5/63);
  - GARCH term structure $\log\big((\text{cum}_{21}/21)/f_{1}\big)$;
  - GARCH variance innovation;
  - persistence;
  - vol-of-vol;
  - own-history vol percentile;
  - within-horizon vol rank;
  - leg-volatility ratio;
  - market volatility percentile and change.
- **C (17): cross-asset.** Leg-difference of factor exposures and correlations, group and
  factor-residual momentum, weekly factor returns, trend and correlation regimes.

**Two channels, tested separately.**

1. **Normaliser.** Train on $z = y/s$ and forecast $\hat y = \hat z\,s$, with
   $s \in \{\sigma_{\text{EWMA}}\sqrt h,\ \sqrt{\hat\sigma^2_{\text{GARCH-t}}(\text{label window})},\ \text{const}\}$.
   All predictor-channel arms use EWMA scaling.
2. **Predictor.** Add volatility features, with the label scaling held fixed.

**Why volatility state can only matter non-linearly here.** Labels are demeaned across targets
on each date. For a *date-level* feature $m_t$ (e.g. the market volatility percentile), the
pooled least-squares moment is $\sum_{t,j} m_t \tilde z_{j,t} = \sum_t m_t \sum_j \tilde z_{j,t} = 0$.
Its linear coefficient is therefore exactly uninformative, and it can only act by **changing
how other features map into returns**, which needs interactions (trees). Target-level vol
features can act linearly, as in a low-volatility anomaly, but a linear model must then find a
stable sign.

### 4.2 Results: MITSUI, 424 targets, daily rank IC

| Arm | Dev mean IC | Holdout mean IC [95% CI] | Holdout IC std | Worst 126-day rolling IC (holdout) |
|---|---|---|---|---|
| hist_mean (benchmark) | 0.0269 | 0.0059 [−0.024, 0.037] | 0.242 | −0.050 |
| tgbm_R | −0.0051 | −0.0046 [−0.026, 0.015] | 0.183 | −0.070 |
| **tgbm_RV** | 0.0129 | **0.0086** [−0.014, 0.031] | 0.208 | −0.055 |
| tgbm_RVs | 0.0140 | 0.0077 [−0.016, 0.032] | 0.215 | −0.054 |
| tgbm_RC | −0.0010 | 0.0017 [−0.021, 0.025] | 0.201 | −0.058 |
| tgbm_RVC | 0.0049 | 0.0026 [−0.019, 0.026] | 0.211 | −0.043 |
| tgbm_R_garch | −0.0013 | −0.0031 [−0.022, 0.015] | 0.174 | −0.054 |
| tgbm_R_none | 0.0054 | −0.0044 [−0.027, 0.016] | 0.190 | −0.070 |
| tridge_R | 0.0018 | 0.0042 [−0.023, 0.031] | 0.245 | −0.041 |
| tridge_RV | 0.0019 | 0.0014 [−0.026, 0.027] | 0.239 | −0.052 |

Pre-registered comparisons (paired daily ΔIC, HAC t; Holm across R1–R3):

| | Comparison | Development | Holdout | Pooled OOS (secondary) | Verdict |
|---|---|---|---|---|---|
| R1 | RV − R | +0.018 [0.002, 0.034], t = 2.04 | +0.013 [−0.005, 0.032], t = 1.44, Holm p = 0.45 | +0.016, t = 2.46 | **not confirmed**; consistent sign |
| R2 | RVC − RC | +0.006, t = 1.12 | +0.001, t = 0.14 | +0.004, t = 0.96 | not supported |
| R3 | R_garch − R (normaliser) | +0.004, t = 1.55 | +0.0015, t = 0.60 | +0.003, t = 1.62 | not supported |
| R4 | R (EWMA) − R_none | −0.011, t = −2.03 | −0.000, t = −0.05 | | not supported |
| R5 | RVs − R (state only) | +0.019, t = 2.17 | +0.012, t = 1.28 | +0.017, t = 2.48 | carries all of R1 |
| R5 | RV − RVs (normalised signals) | −0.001 | +0.001 | −0.000 | no information |
| R8 | tridge_RV − tridge_R | +0.000 | −0.003 | | none (linear) |

- **Multiplicity:** every target-level arm is in the 90% MCS in both periods. Hansen's SPA
  rejects "no volatility arm beats tgbm_R" on development (p = 0.033) but not on the holdout
  (p = 0.178).
- **Stability (R7):** volatility features raise the IC level and improve the worst 126-day
  window (−0.055 vs −0.070), but they also raise IC dispersion (0.208 vs 0.183). They do not
  make forecasts more *stable* in the IR sense on a significant basis.

![Arms](../reports/vol/figures/v07_return_arms.png)

### 4.3 Where the gain comes from (R6)

ΔIC(RV − R) by causal market-volatility regime:

| Period | low | mid | high |
|---|---|---|---|
| development | −0.021 (t = −1.26) | +0.031 (t = 2.65) | +0.022 (t = 1.41) |
| holdout | +0.010 (t = 0.94) | +0.031 (t = 2.36) | −0.027 (t = −0.82, 66 dates) |

ΔIC(RV − R) within ex-ante *target*-volatility terciles:

| Period | low-vol targets | mid | high-vol targets |
|---|---|---|---|
| development | +0.030 (t = 2.33) | +0.011 | +0.017 |
| holdout | +0.021 | +0.010 | +0.010 |

- **Mid-volatility regimes.** The one regime-level result that replicates is the gain in
  *mid*-volatility regimes (t = 2.65 development, 2.36 holdout). The pre-registered hypothesis
  that the gain is *largest in high-volatility regimes* is **not supported**: the holdout's
  66 high-volatility dates (spring 2025 tariff shock) show a loss.
- **Low-volatility targets.** In both periods the gain is largest for the *lowest-volatility
  targets*, e.g. FX-heavy spreads. For them, the volatility state (rather than the noisy
  return history) carries relatively more information about how they will rank.
- **GARCH normalisation** helps only in high-volatility development dates (t = 2.23), where
  GARCH's mean-reverting forecasts differ most from EWMA. That is consistent with Part II,
  but too weak to matter overall.

![By market vol](../reports/vol/figures/v08_delta_by_market_vol.png)
![By target vol](../reports/vol/figures/v09_delta_by_target_vol.png)

### 4.4 Mechanism: which volatility state, and is it stable?

In tgbm_RVC, refitted at the holdout start:

- **Gain shares by group:** C 51%, Vs 32%, R 11%, Vn 6%.
- **Top single features:** market volatility percentile (11.6% of all gain), weekly metals
  factor return, market volatility change, trend regime, cross-factor correlation.

The model mostly splits on *date-level* state variables. By §4.1, those can only act by
re-weighting the target-level signals.

To check whether that conditioning is useful out of sample, a **post-hoc exploratory
decomposition** (`configs/vol_explore.yaml`, not pre-registered) splits the volatility state
into its market-level part (`mkt_vol_pct`, `mkt_vol_chg`) and each target's *own* volatility
state:

| ΔIC (paired, HAC t) | Development | Holdout | Pooled OOS |
|---|---|---|---|
| E1: R + market vol state − R | +0.003 (t = 0.40) | +0.000 (t = 0.02) | +0.002 (t = 0.33) |
| E2: R + own vol state − R | +0.005 (t = 0.83) | **+0.015 (t = 2.32)** | +0.009 (t = 1.93) |
| E3: market state on top of own state | **+0.014 (t = 2.09)** | −0.003 (t = −0.36) | +0.008 (t = 1.40) |

**Interpretation.**

- **Market state alone does nothing.** This is expected: a date-level variable cannot rank
  targets by itself.
- **Development and holdout gains came from different places.** On development, the gain came
  from *interactions* between market volatility and own volatility state (E3). On the holdout,
  those interactions added nothing, and the target's own volatility state carried the gain
  (E2).
- **What is persistent and what is fragile.** The component that is positive in both periods
  is each target's own volatility state, i.e. its forecast variance, volatility term structure
  and volatility percentile. The market-regime conditioning that the trees rely on most is the
  fragile part.

This explains why:

1. volatility *state* carries the gain while volatility *normalisation* does not;
2. a linear model cannot exploit it;
3. its regime profile differs between periods.

These decomposition results are post hoc and should be read as hypotheses for future data.

![Group importance](../reports/vol/figures/v11_group_importance.png)
![Rolling IC](../reports/vol/figures/v10_rolling_ic.png)

## 5. Robustness

All specifications below were fixed in the freeze record before any holdout result was seen,
except the exploratory decomposition (§4.4). Each cell is ΔIC with its HAC t.

| Specification | R1: RV − R, dev | R1, holdout | R1, pooled | R5: RVs − R, dev | R5, holdout | R5, pooled |
|---|---|---|---|---|---|---|
| Main: expanding window, refit 126, 7 leaves | +0.018 (2.04) | +0.013 (1.44) | +0.016 (2.46) | +0.019 (2.17) | +0.012 (1.28) | +0.017 (2.48) |
| Faster refits: every 42 dates | +0.018 (1.98) | +0.012 (1.30) | +0.015 (2.35) | +0.018 (2.11) | +0.009 (0.96) | +0.015 (2.26) |
| Higher capacity: 15 leaves, min_child 1000, refit 42 | +0.022 (2.04) | +0.014 (1.20) | +0.019 (2.33) | +0.019 (1.83) | +0.015 (1.22) | +0.018 (2.17) |
| Rolling 750-date window, refit 126 | +0.024 (2.51) | **+0.001 (0.07)** | +0.015 (2.17) | +0.026 (2.73) | +0.002 (0.25) | +0.017 (2.45) |
| Linear model (Ridge), main schedule | +0.000 (0.02) | −0.003 (−0.43) | | | | |

**Holdout halves** (main run): R1 is +0.007 (t = 0.67) in the first 250 dates and +0.019
(t = 1.36) in the second.

**By horizon** (main run, R1 t-statistics):

| | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| development | 1.59 | 1.35 | 1.91 | 2.26 |
| holdout | 0.41 | −0.37 | 2.20 | 0.85 |

The gain is larger at 3–4 days, where label windows average out more daily noise.

**Reading the table.**

- **Faster refits and higher capacity** reproduce the main result almost exactly: about
  +0.02 on development, +0.01 on the holdout, about +0.016 pooled with t ≈ 2.3–2.5.
  Development and the holdout are never significant at the same time.
- **A rolling window removes the holdout gain.** The R-only baseline adapts and improves
  (holdout IC +0.011 vs −0.005 with an expanding window), and the volatility features then add
  nothing. One reading: part of what volatility-state features contribute under an expanding
  window is *regime adaptation* that a shorter window delivers more directly.
- **Ridge never benefits.** Whatever information the volatility features carry is
  non-linear.

## 6. Independent replication: FRED energy panel

Six daily FRED series (EIA WTI, Brent, Henry Hub gas, NY Harbor heating oil and gasoline,
and the Fed broad USD index), 2010-01 to 2026-10.

- **Development walk-forward:** 2015-02 → 2025-12.
- **Fresh holdout:** 2026-01-02 → 2026-10-06 (197 dates), never examined before the
  pre-registration.
- **Targets:** 20 configuration-generated targets (6 single assets and 4 spreads vs WTI, at
  h = 1 and 5), so each horizon cross-section has only 10 targets.

**Part I.**

- **Even heavier tails than MITSUI.** Excess kurtosis is 112 (Brent), 118 (WTI) and 119
  (Henry Hub), driven by the 2020 oil collapse and gas freeze-offs.
- **Persistent volatility clustering:** median α+β = 0.988, half-life 59 days.
- **Leverage in oil:** GJR γ = 0.07 (Brent), 0.09 (WTI); volatility rises after price falls.
  The USD index shows γ = −0.03.
- **Too few days to test clustering in 2026.** With 197 days, ARCH effects are detectable in
  only 2 of 6 series. That is a power limit, not evidence of absence.

**Part II (panel DM t-statistics; negative = model A better):**

| Hypothesis | A vs B | dev 1d | dev 5d | dev 21d | **2026 1d** | **2026 5d** | **2026 21d** |
|---|---|---|---|---|---|---|---|
| V1 | garch_t vs const | −3.88 | −3.21 | −1.70 | −2.23 | −2.27 | −2.25 |
| V1 | ewma vs const | −3.64 | −2.93 | −1.48 | −2.15 | −2.12 | −2.19 |
| V2 | garch_t vs ewma | **−3.12** | **−3.22** | **−2.45** | **−2.28** | −1.86 | −1.41 |
| V3 | gjr_t vs garch_t | +0.22 | +0.13 | −0.47 | **+2.07** | **+2.05** | +1.50 |
| V4 | garch_t vs garch_n | +0.96 | −0.47 | −0.59 | −2.05 | −0.32 | +2.70 |

**Volatility-forecasting conclusions replicate on independent data and on a fresh period.**

- Dynamics beat constant variance.
- GARCH-t beats EWMA, significantly at every development horizon and at 1 day on the 2026
  sample.
- The asymmetric GJR model is *worse* out of sample despite rejecting symmetry in-sample for
  5 of 6 series.

**Part III (return forecasting).** Volatility features do not help:

- R1 = −0.005 (development, t = −0.49) and −0.006 (2026, t = −0.17).
- SPA p = 0.60 / 0.65.

With 10 targets per horizon the daily IC is extremely noisy (sd ≈ 0.4), so this is a
low-power check. It does not replicate the MITSUI development gain.

**EWMA normalisation hurts here, as on MITSUI development.** Normalising labels by EWMA
volatility again *hurts* rank accuracy relative to raw returns: −0.013 (t = −1.92),
−0.036 (t = −1.91), pooled −0.015 (t = −2.24). This is the most consistent normaliser
result in the project. When the score is a rank correlation on *raw* returns, the most
volatile targets dominate the realised cross-section. A model trained on vol-scaled labels
down-weights exactly those targets.

## 7. Financial interpretation

1. **Volatility is predictable; returns are not (outside crises).** This is the oldest
   stylised fact in empirical finance, and it holds across LME metals, JPX precious metals,
   US equities, FX and US energy spot markets.
   - Every measure of volatility clustering rejects in every sub-period.
   - Return autocorrelation is significant only when a crash and rebound are in the window.
   - A daily return model that "finds" autocorrelation in 2018–2021 data is mostly fitting
     March–April 2020.
2. **What GARCH adds over EWMA is mean reversion.** RiskMetrics' EWMA is an integrated
   GARCH: it assumes today's volatility persists indefinitely. A GARCH with estimated
   persistence near 0.98 (half-life ≈ 1–2 months) instead predicts decay back toward the
   long-run level.
   - That is why GARCH's advantage grows with horizon and is largest after volatility
     spikes.
   - It matters for risk management: margining, VaR horizons and position sizing over days
     to weeks.
   - The gain is real but small (QLIKE ≈ 0.03–0.05 per asset-day on the holdout), consistent
     with Hansen & Lunde's finding that GARCH(1,1) is hard to beat.
3. **Asymmetry is real but not forecastable enough to pay for its parameter.**
   - Equities show the classic leverage effect: falling prices raise financial leverage and
     risk aversion.
   - Gold shows the *opposite*: volatility rises in rallies, when flight-to-quality demand
     surges.
   - Oil shows leverage-type asymmetry.
   - Out of sample, though, GJR never beats symmetric GARCH-t. The in-sample asymmetry is
     partly a few crash days, and estimating γ adds variance to the forecast.
4. **Fat tails matter for estimation more than for forecasting.** Student-t likelihoods
   prevent a few extreme days from inflating α (copper: α = 0.088 under Gaussian QML vs 0.024
   under t; across all 106 assets the median α falls from 0.105 to 0.091). This helped forecasting only in the turbulent post-2020 development period.
5. **For return forecasting, volatility is a conditioning variable, not a signal.**
   - Volatility features raised the IC of a tree model by about 0.01–0.02 on average. The
     effect has a consistent sign across development, the holdout, refit schedules and model
     capacity. Its evidential strength is that of a pooled t ≈ 2.4 that the pre-registered
     holdout test alone could not confirm.
   - Where it shows up: low-volatility targets and mid-volatility markets. In the
     high-volatility holdout dates it flips.
   - None of it is usable linearly, and normalising returns by better volatility forecasts
     does not help ranking.
   - Economically, volatility tells you *which signals to trust when*. Under strict
     out-of-sample evaluation that information is weak and regime-fragile, the familiar
     pattern for volatility timing (Moreira & Muir 2017 vs. out-of-sample critiques).
6. **For a practitioner.**
   - Use GARCH-t for *risk*: sizing, VaR and stop distances.
   - Do not expect it to create *alpha* in a short-horizon commodity cross-section.
   - If volatility features are added to a return model, judge them on pooled long samples
     and against an adaptive (rolling-window) baseline. Much of their apparent value can be
     regime adaptation.

## 8. Limitations

- **Volatility proxies.** Daily squared returns are noisy, and the range proxy is biased. With
  intraday realised variance, the power of the volatility-forecast comparison would rise by
  an order of magnitude.
- **Holdout reuse.** The MITSUI holdout had been used before. The pre-registration and freeze
  limit, but cannot eliminate, the influence of knowing its general behaviour. The FRED 2026
  sample is the only truly fresh data, and it is short.
- **Power.** One 500-date holdout cannot resolve ΔIC ≈ 0.01. The pooled analysis is secondary
  by design.
- **Small FRED cross-section.** Ten targets per horizon make the Part III replication weak.
- **Re-estimation choices.** Fixed (21-day re-estimation, expanding window for volatility
  models, λ = 0.94) and not optimised. Other choices could change magnitudes.
- **Anonymised MITSUI dates.** Volatility events cannot be linked to macro announcements.
- **Not tested here.** Implied-volatility and variance-risk-premium information
  (options-based) is the best-known channel through which volatility predicts commodity
  returns, and no option data were available.

## References

- Andersen, T., Bollerslev, T. (1998). Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts. *International Economic Review* 39(4).
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics* 31(3).
- Diebold, F., Mariano, R. (1995). Comparing Predictive Accuracy. *JBES* 13(3).
- Engle, R. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of UK Inflation. *Econometrica* 50(4).
- Engle, R., Ng, V. (1993). Measuring and Testing the Impact of News on Volatility. *Journal of Finance* 48(5).
- Garman, M., Klass, M. (1980). On the Estimation of Security Price Volatilities from Historical Data. *Journal of Business* 53(1).
- Glosten, L., Jagannathan, R., Runkle, D. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance* 48(5).
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *JBES* 23(4).
- Hansen, P. R., Lunde, A. (2005). A Forecast Comparison of Volatility Models: Does Anything Beat a GARCH(1,1)? *Journal of Applied Econometrics* 20(7).
- Hansen, P. R., Lunde, A., Nason, J. (2011). The Model Confidence Set. *Econometrica* 79(2).
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics* 6(2).
- Li, W. K., Mak, T. K. (1994). On the Squared Residual Autocorrelations in Non-linear Time Series with Conditional Heteroskedasticity. *Journal of Time Series Analysis* 15(6).
- Mincer, J., Zarnowitz, V. (1969). The Evaluation of Economic Forecasts. In *Economic Forecasts and Expectations*, NBER.
- Moreira, A., Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance* 72(4).
- Nagel, S. (2012). Evaporating Liquidity. *Review of Financial Studies* 25(7).
- Patton, A. (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies. *Journal of Econometrics* 160(1).
- RiskMetrics (1996). *Technical Document*, 4th ed. J.P. Morgan/Reuters.
