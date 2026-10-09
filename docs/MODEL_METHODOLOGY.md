# Model methodology

This document specifies every model's inputs, architecture, training objective and output,
and how forecasts are combined. Code references are relative to `src/commodity_research/`.

## 1. The prediction problem

On each decision date $t$ (a MITSUI `date_id`), forecast 424 targets. Each is either a single
asset's forward log return or the difference of two assets' forward log returns, at horizon
$h \in \{1,2,3,4\}$ dates, starting one date after the decision (gap $g=1$):

$$
r^{i}_{t,h} = \log P^{i}_{t+g+h} - \log P^{i}_{t+g},
\qquad
y_{j,t} = r^{A_j}_{t,h_j} - r^{B_j}_{t,h_j}\ \ (\text{or } r^{A_j}_{t,h_j}).
$$

Evaluation ranks the 424 forecasts against the 424 realised labels on each date (Spearman), so
only the **cross-sectional ordering** of forecasts matters.

### 1.1 Asset-factorised design (primary)

The 424 targets are linear combinations of **106 asset returns × 4 horizons**. The primary
models therefore forecast asset returns $\hat r^{i}_{t,h}$ and compose target forecasts
linearly (`features/panel.py: ModelingPanel.to_targets`):

$$
\hat y_{j,t} = \hat r^{A_j}_{t,h_j} - \hat r^{B_j}_{t,h_j}.
$$

This has three benefits:

1. **Coherence.** Forecasts are internally consistent, e.g. $\hat y(A-B) + \hat y(B-C) = \hat y(A-C)$.
2. **Statistical efficiency.** One pooled panel per horizon (about 106 × 1,300 rows) replaces
   424 separate series of about 1,300 noisy observations each.
3. **Generality.** Any new pair of known assets is forecastable without retraining.

Direct pair-level models are implemented as comparators (§4.4).

### 1.2 Common training target: vol-scaled, cross-sectionally demeaned returns

All learned asset-level models minimise squared error on

$$
z_{i,t,h} = \operatorname{clip}\!\left(\frac{r^{i}_{t,h}}{\sigma_{i,t}\sqrt h}, -5, 5\right),
\qquad
\tilde z_{i,t,h} = z_{i,t,h} - \frac{1}{|\mathcal A_t|}\sum_{k\in\mathcal A_t} z_{k,t,h},
$$

where $\sigma_{i,t}$ is the causal EWMA volatility ($\lambda = 0.94$, known at $t$) and
$\mathcal A_t$ is the set of active assets with an observable label. Forecasts are demeaned
the same way and mapped back to return units with the known-at-$t$ scale:
$\hat r = \hat{\tilde z}\,\sigma_{i,t}\sqrt h$.

**Why vol scaling.** Daily volatility ranges from about 0.4% (FX, short bonds) to about 3%
(energy and mining equities). Without scaling, the pooled loss is dominated by a few volatile
equities.

**Why demeaning.** The metric is a cross-sectional rank correlation. The date-wide average
move (in z units) is noise for that task, and if it is left in the target, a pooled model can
turn date-level variables into **volatility tilts**: a common shift in $\hat z$ times
$\sigma_i$ ranks assets by volatility. The `ridge_rawlabel` ablation measures this choice
(development: demeaned +0.011 vs raw −0.025 mean IC; see EXPERIMENT_RESULTS). With
demeaned labels, purely date-level regime variables cannot enter a linear model as a tilt.
They can only act through interactions (trees, LSTM).

## 2. Inputs: feature families (`features/engineering.py`)

There are 38 features per asset and date, all computed from **past and current prices only**,
and all scale-free. Price levels are never used: US prices are vendor-adjusted closes whose
levels depend on later dividends and splits, whereas log-price differences are invariant to
proportional back-adjustment. Notation: $p_t$ is the forward-filled log price (at most 5 rows),
$r_t$ the daily log return (NaN on non-trading dates), and $\sigma_t$ the EWMA volatility.

| Group | Features | Definition (abridged) | Financial interpretation |
|---|---|---|---|
| **own: momentum / trend / reversal** | `ret_1_z`, `ret_1_lag1_z` | $r_t/\sigma_{t-1}$, its lag | short-term reversal / overreaction |
| | `mom_{5,21,63}_z` | $(p_t-p_{t-k})/(\sigma_t\sqrt k)$ | risk-adjusted time-series momentum |
| | `mom_126_21_z` | $(p_{t-21}-p_{t-126})/(\sigma_t\sqrt{105})$ | medium-term momentum skipping the last month |
| | `madev_{20,60}_z` | $(p_t-\overline{p}_{k})/(\sigma_t\sqrt k)$ | deviation from moving average |
| | `trend_eff_63` | $(p_t-p_{t-63})/\sum_{63}\lvert r\rvert$ | trend efficiency in $[-1,1]$: trend vs range |
| **own: volatility / distribution** | `log_vol_ewma` | $\log\sigma_t$ | risk level (low-vol anomaly) |
| | `log_vol_ratio_21_63` | $\log(\mathrm{vol}_{21}/\mathrm{vol}_{63})$ | volatility expansion |
| | `log_downside_ratio_63` | $\log(\text{downside semi-dev}/\mathrm{vol}_{63})$ | downside asymmetry |
| | `skew_63`, `exkurt_63` | rolling sample moments | crash risk, tail thickness |
| **cross-asset** | `expo_{EQ,USD,METAL,GOLD,RATES,ENERGY}_1d` | $\hat\beta_{i,k}\, f_{k,t}/\sigma_{i,t}$ (126-date rolling beta) | factor-implied move today; lead-lag through asynchronous closes |
| | `beta_home_126`, `corr_home_63`, `corr_eq_63` | rolling beta/correlation to the asset's "home" factor and to global equity | integration with its market / risk-on regime |
| | `home_expo_5d_z` | $\hat\beta\,(F_t-F_{t-5})/(\sigma\sqrt5)$ | home-factor move over a week |
| | `log_vol_rel_home` | $\log(\sigma_i/\sigma_{\text{home}})$ | idiosyncratic vs systematic risk |
| | `fac_{EQ,USD,METAL}_5d_z` | weekly factor returns (date-level) | macro state, usable through interactions |
| **relative value** | `xs_rank_mom_{5,21,63}` | cross-sectional percentile rank − 0.5 | cross-sectional momentum / reversal |
| | `grp_dev_mom_21` | momentum minus its market-group mean | relative strength within commodities / equities / FX |
| | `resid_mom_21_z` | $(\Delta_{21}p - \hat\beta\,\Delta_{21}F_{\text{home}})/(\sigma\sqrt{21})$ | idiosyncratic (factor-residual) momentum |
| | `xs_rank_vol` | cross-sectional rank of $\sigma$ | relative risk |
| **regime (date-level)** | `reg_eq_vol_pct` | expanding percentile of VT 21-date vol | volatility state |
| | `reg_eq_vol_chg` | $\log(\mathrm{vol}_5/\mathrm{vol}_{63})$ of VT | volatility shock |
| | `reg_eq_trend` | $\lvert\Delta_{63}p\rvert/(\mathrm{vol}_{63}\sqrt{63})$ of VT | trending vs ranging market |
| | `reg_xcorr_63` | mean $\lvert\rho\rvert$ among the six factors (63 dates) | cross-asset correlation regime |

**Factors** (`data/mitsui.py: _factor_definitions`) are daily equal-weighted log-return
composites:

- **EQ** = VT (global equity ETF)
- **USD** = sign-oriented mean of all USD crosses (positive = USD strengthens)
- **METAL** = the 4 LME base metals
- **GOLD** = GLD, IAU
- **RATES** = IEF, VGIT
- **ENERGY** = XLE

Each asset has a "home" factor chosen from a hand-curated classification of the 95 US
tickers into broad equity, bonds, precious-metal, base-metal/materials, energy and
industrial/financial groups.

**Pre-processing** (`models/base.py: TrainOnlyScaler`). At each refit, every feature is
winsorised at its training-window 0.5/99.5 percentiles, standardised with training-window
mean and std, and missing values are set to 0 (the training mean). Statistics come from
training decision dates only.

**Feature selection.** No data-driven selection is performed. The two feature sets
(`own` = 14 features, `all` = 38) are fixed in the configuration for the ablation.

## 3. Level A: benchmarks (`models/baselines.py`)

These need no fitting, so they cannot be over-fit:

| Model | Forecast | Rationale |
|---|---|---|
| `hist_mean` | $\hat r_{i,t,h} = h\cdot\overline{r}_{i,[t-251,t]}$ | historical expected return (1-year drift, i.e. cross-sectional 12-month momentum) |
| `momentum_21` | $h\cdot(p_t-p_{t-21})/21$ | 1-month time-series momentum |
| `reversal_5` | $-h\cdot(p_t-p_{t-5})/5$ | 1-week reversal |

All forecasts are in return units for horizon $h$, so they produce valid rankings for single
and pair targets alike.

## 4. Learned models

### 4.1 Level A: pooled Ridge (`models/tabular.py: RidgeModel`)

One `sklearn.linear_model.Ridge` per horizon on the pooled (date × asset) panel of
standardised features, with target $\tilde z$. $\alpha$ is selected from a grid on the
development walk-forward. The features are standardised and $n \approx 10^5$ rows, so
meaningful shrinkage needs $\alpha \sim 10^3$–$10^6$.

### 4.2 Level B: pooled LightGBM (`models/tabular.py: LightGBMModel`)

One gradient-boosted tree ensemble per horizon. Configuration is deliberately restrained:

- 200–300 trees, learning rate 0.02–0.03
- 7–31 leaves, `min_child_samples` 1,000–3,000 (i.e. every leaf averages at least about
  10 dates × 100 assets)
- row subsampling 0.7, column subsampling 0.8, L2 penalty 5
- `deterministic=True`

Trees can use the date-level regime variables through interactions, e.g. "momentum works
only in low-volatility regimes".

### 4.3 Level C: multi-horizon LSTM (`models/lstm.py`)

The legacy architecture (bidirectional LSTM, attention, deep head, skip connection, SWA) was
**replaced**, not patched. It could not be constructed, produced outputs of the wrong shape,
and was grossly over-parameterised for the signal-to-noise ratio
(see [LEGACY_AUDIT.md](LEGACY_AUDIT.md) §7).

```
input   x ∈ R^{B × W × F}   last W = 20 rows of the asset's train-scaled features, ending at t
LSTM    1 layer, 32 hidden units, unidirectional
head    dropout(0.2) → Linear(32 → H=4)        one output per horizon (multi-output)
loss    masked MSE on z̃: a (row, horizon) cell enters the loss only if its label is
        observable at the refit date (per-horizon availability mask)
```

| Training detail | Value |
|---|---|
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Batch / clipping | 512, gradient-norm clip 1.0 |
| Training sample | at most 50k (FAST) / 120k (FULL) sequences sampled per refit with a seeded RNG |
| Early stopping | purged internal validation block (last 126 observable dates; fit labels observable before it starts); patience 3; best weights restored with `copy.deepcopy` |
| Ensembling | 1 seed (FAST), 3 seeds averaged (FULL) |
| Hardware | CPU by default (`device` configurable; MPS/CUDA optional) |

Sequences are gathered with `make_sequences`, which provably reads rows $\le t$ of the same
asset only (unit-tested).

The LSTM sees the same 38 features as the tabular models. Its only extra information is the
ordering of the last 20 rows, so it tests whether temporal dynamics *within* the feature
history add information beyond the hand-built multi-window features. Early stopping typically
selects epoch 1–2, itself a sign of very low signal-to-noise.

### 4.4 Direct pair-level comparators (`models/direct.py`)

| Model | Structure |
|---|---|
| `ridge_direct_pooled` | one ridge per horizon, pooled over that horizon's 106 targets |
| `ridge_direct_target` | one ridge **per target** (424 per refit) with its own intercept |

Inputs are leg-A minus leg-B own features, plus two relative-value features of the log price
ratio $\ell_t = p^A_t - p^B_t$:

- `spread_z_63` $= (\ell_t-\overline{\ell}_{63})/s_{63}(\ell)$, the classic pairs-trading
  mean-reversion indicator
- `spread_mom_21` $= (\ell_t-\ell_{t-21})/(\sigma_\ell\sqrt{21})$

The labels are vol-scaled target labels, demeaned across the targets of the same horizon. The
per-target intercept turns `ridge_direct_target` partly into a *target-level historical-mean*
model.

## 5. Level D: forecast combination (`models/ensemble.py`)

Members' asset forecasts are put on a common scale using
$s_{m,h} = \operatorname{sd}(\hat r^m_{\cdot,\cdot,h})$ over development dates, then combined:

$$
\hat r^{\text{ens}}_{i,t,h} = \bar s_h \sum_m w_{m,t}\,\frac{\hat r^{m}_{i,t,h}}{s_{m,h}},
\qquad \bar s_h = \tfrac1M\sum_m s_{m,h}.
$$

| Scheme | Weights | Information used |
|---|---|---|
| `equal` | $1/M$ | none |
| `static_ic` | $\propto \max(\overline{IC}^{\,\text{dev}}_m, 0)$, shrunk 50% to equal | development OOS ICs |
| `regime` | `static_ic` computed separately for low/mid/high-volatility development dates; date $s$ uses the weights of the regime observed at $s$ | development OOS ICs + causal regime label |
| `online` | $\propto \max(\overline{IC}_{m,[s-g-h_{\max}-125,\ s-g-h_{\max}]}, 0)$, shrunk 50% | trailing realised ICs **observable at $s$** (a live-feasible rule) |

There are two member sets:

- `ens_*`: the three learned models (ridge, lgbm, lstm), **pre-specified**.
- `ens_best_*`: models with a positive **development** score among the benchmarks and
  learned models. This rule was fixed before the holdout was evaluated.

The 50% shrinkage toward equal weights is a standard guard against estimation error in
combination weights (the "forecast combination puzzle").

## 6. Risk-aware layer (`risk/diagnostics.py`)

The layer keeps three quantities separate:

- **Market volatility** $\sigma_{i,t}$: how much prices move.
- **Predictive uncertainty**: how wrong a forecast is likely to be. It is measured by
  split-conformal intervals,
  $\hat r \pm q_{1-\alpha}\big(\lvert r-\hat r\rvert/(\sigma\sqrt h)\big)\,\sigma\sqrt h$,
  with the quantile taken from **development** residuals and coverage checked on the holdout,
  overall and by volatility regime. They are compared against constant-width intervals.
- **Model disagreement**: the cross-model std of scale-normalised member forecasts. It is
  *not* treated as a confidence interval. We only test whether it is associated with larger
  realised (vol-scaled) errors.

A hypothetical dollar-neutral portfolio on the 106 single assets (horizon 1) translates the
signals into a P&L diagnostic:

- weights $\propto$ demeaned $\hat r/\sigma$ (or raw $\hat r$), unit gross exposure
- enter at close $t+1$, exit at close $t+2$, exactly the horizon-1 label window
- cost sensitivity of 0/2/5/10 bps per unit turnover

The 420 pair targets are **not** treated as tradable portfolios. The backtest ignores
financing, borrow, FX conversion of JPY/GBP-denominated futures, contract rolls, market
impact and the different closing times of LME, JPX, US and FX markets. It is a signal
diagnostic, not evidence of tradable profitability.

## 7. Reproducibility

- Seeds: NumPy, `torch.manual_seed`, a seeded `torch.Generator` for shuffling,
  `random_state` for LightGBM with `deterministic=True`, and a seeded RNG for LSTM
  subsampling.
- Each model runs in a fresh spawned process. This isolates the torch and LightGBM OpenMP
  runtimes, which deadlock or segfault in one process on macOS.
- Two independent FAST runs reproduced every development score exactly, and unit tests
  assert identical predictions for identical seeds (Ridge, LightGBM, LSTM).
