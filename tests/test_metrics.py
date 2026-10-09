import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from commodity_research.evaluation.metrics import (block_bootstrap, competition_score, daily_rank_ic,
                                                   newey_west_se, paired_ic_test, summarize_ic)


def reference_competition_score(pred: pd.DataFrame, label: pd.DataFrame) -> float:
    """Re-implementation following the structure of the official MITSUI metric."""
    def row_corr(i):
        y = label.iloc[i]
        ok = y.notna()
        p = pred.iloc[i][ok.to_numpy()]
        return np.corrcoef(p.rank(method="average"), y[ok].rank(method="average"))[0, 1]

    daily = pd.Series([row_corr(i) for i in range(len(label))])
    return float(daily.mean() / daily.std(ddof=0))


def test_daily_ic_matches_spearman():
    rng = np.random.default_rng(0)
    p, y = rng.normal(size=(30, 50)), rng.normal(size=(30, 50))
    y[rng.random(y.shape) < 0.1] = np.nan
    ic, n = daily_rank_ic(p, y)
    for d in range(30):
        m = np.isfinite(y[d])
        assert ic[d] == pytest.approx(spearmanr(p[d, m], y[d, m]).statistic)
        assert n[d] == m.sum()


def test_competition_score_matches_reference():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(40, 60))
    p = 0.2 * y + rng.normal(size=y.shape)
    y[rng.random(y.shape) < 0.15] = np.nan
    ref = reference_competition_score(pd.DataFrame(p), pd.DataFrame(y))
    assert competition_score(p, y) == pytest.approx(ref)


def test_degenerate_dates_are_excluded_not_rewarded():
    rng = np.random.default_rng(2)
    p, y = rng.normal(size=(5, 20)), rng.normal(size=(5, 20))
    p[1] = 3.0  # constant forecast -> undefined rank correlation
    y[2] = np.nan  # no labels
    ic, _ = daily_rank_ic(p, y)
    assert np.isnan(ic[1]) and np.isnan(ic[2])
    s = summarize_ic(ic, reps=50)
    assert s["n_degenerate"] == 2 and s["n_dates"] == 3


def test_newey_west_reduces_to_iid_se_without_lags():
    x = np.random.default_rng(3).normal(size=500)
    assert newey_west_se(x, lags=0) == pytest.approx(x.std() / np.sqrt(len(x)))


def test_newey_west_larger_for_positively_autocorrelated_series():
    rng = np.random.default_rng(4)
    e = rng.normal(size=2000)
    x = np.convolve(e, np.ones(5), mode="valid")  # MA(4): overlapping-label structure
    assert newey_west_se(x, lags=8) > 1.5 * newey_west_se(x, lags=0)


def test_bootstrap_reproducible_and_centered():
    x = np.random.default_rng(5).normal(0.1, 1, 400)
    a = block_bootstrap(x, np.mean, block=10, reps=200, seed=7)
    b = block_bootstrap(x, np.mean, block=10, reps=200, seed=7)
    np.testing.assert_array_equal(a, b)
    assert abs(np.mean(a) - x.mean()) < 0.05


def test_paired_test_direction():
    rng = np.random.default_rng(6)
    base = rng.normal(0, 0.1, 300)
    res = paired_ic_test(base + 0.05, base, reps=100)
    assert res["mean_diff"] == pytest.approx(0.05) and res["t_hac"] > 5
