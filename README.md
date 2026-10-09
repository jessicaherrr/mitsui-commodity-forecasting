# Commodity Forecasting & Cross-Asset Research

This project asks a simple question: **can we predict short-term commodity returns from
market data, and does volatility help?**

It started as my entry to the Kaggle *MITSUI & CO. Commodity Prediction Challenge* (2025).
I later rebuilt it into a research project with strict out-of-sample testing, then added a
second study on volatility. The results are mixed, and the repository reports them as they are.

## The short version

- **Predicting 1–4 day returns is hard.** None of the main learned models (Ridge, LightGBM, an
  LSTM, or the ensembles) beat a simple one-year average-return benchmark on the holdout period.
  A per-target pair model did score higher, but it looked poor on development, and its result
  traces to a specific 2023–25 trend. The benchmark ranks single assets with an average daily rank
  correlation of about 0.03 in both periods, and it's the most stable signal we found.
- **Volatility is very predictable.** Volatility clusters in nearly every instrument. A GARCH(1,1)
  model with Student-t errors beats a standard EWMA forecast out of sample. The one-day ranking
  also held on a small 2026 energy-price sample.
- **Volatility helps returns only a little, and not reliably.** Adding volatility information to a
  LightGBM model raised the average rank correlation by about 0.01–0.02. The holdout result isn't
  statistically significant, the gain reverses on high-volatility days, and it disappears when the
  training window rolls forward.
- **Asymmetry shows up in the data but doesn't forecast better.** Equity volatility tends to jump
  after prices fall, while gold volatility jumps after prices rise. A GJR-GARCH model captures this
  in-sample, but it doesn't forecast better than plain GARCH out of sample.

## A few pictures

More figures (regimes, risk, robustness, replications) are in the [figure gallery](docs/FIGURE_GALLERY.md).

### Return forecasting

How well each model ranks future returns, out of sample. The rolling average hovers around zero
and flips sign, which is why no model stands out.

![Walk-forward rank correlation over time](reports/fast/figures/02_ic_over_time.png)

Adding volatility information (the RV bars) nudges accuracy up a little, but the confidence
intervals still overlap zero.

![Return models with and without volatility information](reports/vol/figures/v07_return_arms.png)

### Volatility modeling

Each volatility model's forecast error relative to EWMA. Lower is better, and the GARCH models
come out ahead, especially in the holdout.

![QLIKE of volatility models relative to EWMA](reports/vol/figures/v04_qlike_vs_ewma.png)

Month-ahead forecasts against what actually happened. GARCH is smoother and pulls back toward
the long-run level after spikes.

![21-day volatility forecasts against realised volatility](reports/vol/figures/v06_vol_forecast_paths.png)

### Financial econometrics

Returns show almost no autocorrelation, but squared returns do, and it lasts for weeks. That's
volatility clustering.

![ACF of returns and squared returns](reports/vol/figures/v01_acf_returns_vs_squared.png)

Raw returns have fat tails. GARCH removes the clustering, but its residuals are still fat-tailed,
and a Student-t distribution fits them much better.

![Q-Q plots of returns and GARCH residuals](reports/vol/figures/v02_qq_plots.png)

## What's inside

- **Return models:** historical averages and momentum benchmarks, Ridge, LightGBM, an LSTM,
  pair-level models, and four ways of combining them.
- **Volatility models:** rolling and EWMA volatility, GARCH(1,1) with Normal and Student-t errors,
  and GJR-GARCH. They're re-estimated on a rolling schedule, using only past data.
- **Econometric checks:** autocorrelation, Ljung–Box, ARCH-LM, Jarque–Bera, Q-Q plots, and
  sign-bias tests, with corrections for testing many series at once.
- **Evaluation:** walk-forward validation, a locked holdout period, block bootstrap confidence
  intervals, Diebold–Mariano tests, and the Model Confidence Set.
- **Dashboard:** a Streamlit app that shows the saved results, diagnostics and every figure.

## How it avoids fooling itself

- A label only counts once it would actually be known. A model never sees a return before its
  own end date.
- Scaling and feature choices are fit on training data only.
- The holdout period was used once per study, after the hypotheses were fixed in writing.
  Every decision made after seeing holdout data is listed in the docs.
- The test suite includes checks that change future prices and confirm earlier forecasts don't
  move.

## The Kaggle origin, honestly

The original competition code is kept, unchanged, in [`legacy/original/`](legacy/original/).
When I reviewed it, I found serious problems. For example, it trained on synthetic data instead
of the competition files, and it predicted price levels instead of returns. So I rebuilt the
project around a verified target definition and strict validation. I make no claim about the
competition ranking. [`docs/INDEPENDENT_CONTRIBUTIONS.md`](docs/INDEPENDENT_CONTRIBUTIONS.md)
and [`docs/LEGACY_AUDIT.md`](docs/LEGACY_AUDIT.md) give the full breakdown.

## Getting started

You don't need the competition data to explore the results. The dashboard and the test suite
run on the included outputs.

```bash
git clone git@github.com:jessicaherrr/mitsui-commodity-forecasting.git
cd mitsui-commodity-forecasting
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"

streamlit run dashboard/app.py    # explore the saved results
pytest                            # 73 tests; 4 more need the Kaggle data
```

To reproduce the results from scratch, first download the MITSUI data as described in
[`data/raw/mitsui/README.md`](data/raw/mitsui/README.md). Then run:

```bash
python -m commodity_research.cli audit                       # check the data and labels
python -m commodity_research.cli all --profile fast          # return models (about 7 minutes)
python -m commodity_research.cli vol-all --profile vol       # volatility study (about 7 minutes)
```

The public FRED energy data is already in `data/external/`. To refresh it, run
`python scripts/fetch_fred_energy.py`.

## Project layout

```
src/commodity_research/   the package: data, features, models, volatility, evaluation, experiments
configs/                  one YAML file per experiment profile
results/                  saved metrics, manifests and logs from each run
reports/                  figures and generated tables
docs/                     methodology, validation, the volatility study, and the audit
dashboard/                the Streamlit app
legacy/original/          the original Kaggle code, unchanged
tests/                    the test suite
```

## Documentation

- [Research report](docs/RESEARCH_REPORT.md): the return-forecasting study
- [Volatility study](docs/VOLATILITY_AND_DIAGNOSTICS.md): diagnostics, volatility forecasts, and the return effect
- [Pre-registration](docs/VOLATILITY_PREREGISTRATION.md): the hypotheses and decision rules, written before evaluation
- [Model methodology](docs/MODEL_METHODOLOGY.md) and [validation and leakage controls](docs/VALIDATION_AND_LEAKAGE.md)
- [Figure gallery](docs/FIGURE_GALLERY.md): every research figure, with short captions
- [Experiment results](docs/EXPERIMENT_RESULTS.md) and [future research](docs/FUTURE_RESEARCH.md)

## Limitations

- Seven years of daily data and one holdout period make small differences hard to detect.
- The data has no futures curves, inventories or options, which are the best-known commodity
  predictors.
- The competition's dates are anonymised, so the calendar is inferred from market events.

## License

No license has been chosen yet, so all rights are reserved for now. Please ask before reusing
the code.

---

The code and write-up were developed with an AI coding assistant (Claude) under my direction.
Every number comes from code that runs in this repository.
