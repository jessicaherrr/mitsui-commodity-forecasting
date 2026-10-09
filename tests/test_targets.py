import numpy as np
import pandas as pd
import pytest

from commodity_research.targets import TargetKind, TargetSpec, parse_mitsui_pair


@pytest.fixture
def prices():
    return pd.DataFrame({"A": [100.0, 101.0, 103.0, 102.0, 104.0, 108.0],
                         "B": [50.0, 50.5, 50.0, 51.0, 52.0, 51.0]})


def test_log_return_definition_with_gap(prices):
    t = TargetSpec("a", TargetKind.LOG_RETURN, ("A",), horizon=2, gap=1)
    y = t.compute(prices)
    # y_0 = log(P_3 / P_1)
    assert y.iloc[0] == pytest.approx(np.log(102.0 / 101.0))
    assert y.iloc[2] == pytest.approx(np.log(108.0 / 102.0))
    assert y.iloc[3:].isna().all()  # needs prices beyond the sample


def test_return_spread_equals_change_in_log_price_ratio(prices):
    t = TargetSpec("ab", TargetKind.RETURN_SPREAD, ("A", "B"), horizon=1, gap=1)
    y = t.compute(prices)
    log_ratio = np.log(prices["A"]) - np.log(prices["B"])
    expected = log_ratio.shift(-2) - log_ratio.shift(-1)
    np.testing.assert_allclose(y.to_numpy()[:4], expected.to_numpy()[:4])


def test_price_spread_change_is_not_return_spread(prices):
    rs = TargetSpec("rs", TargetKind.RETURN_SPREAD, ("A", "B"), 1, 1).compute(prices)
    ps = TargetSpec("ps", TargetKind.PRICE_SPREAD_CHANGE, ("A", "B"), 1, 1).compute(prices)
    spread = prices["A"] - prices["B"]
    assert ps.iloc[0] == pytest.approx(spread.iloc[2] - spread.iloc[1])
    assert not np.allclose(rs.dropna(), ps.dropna())
    # simple-return spread also differs from the log-return spread
    simple = (prices["A"].shift(-2) / prices["A"].shift(-1) - 1) - (prices["B"].shift(-2) / prices["B"].shift(-1) - 1)
    assert not np.allclose(rs.dropna(), simple.dropna())


def test_direction_target(prices):
    d = TargetSpec("d", TargetKind.DIRECTION, ("A",), 1, 1).compute(prices)
    assert set(d.dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert d.iloc[0] == 1.0  # 101 -> 103


def test_missing_price_propagates_to_nan(prices):
    p = prices.copy()
    p.loc[2, "B"] = np.nan
    y = TargetSpec("ab", TargetKind.RETURN_SPREAD, ("A", "B"), 1, 1).compute(p)
    assert np.isnan(y.iloc[0]) and np.isnan(y.iloc[1])  # labels touching row 2


def test_timing_helpers():
    t = TargetSpec("x", TargetKind.LOG_RETURN, ("A",), horizon=4, gap=1)
    assert t.information_interval(10) == (11, 15)
    assert t.available_position(10) == 15


def test_parse_mitsui_pair():
    s = parse_mitsui_pair("target_1", "LME_PB_Close - US_Stock_VT_adj_close", 3)
    assert s.kind == TargetKind.RETURN_SPREAD and s.legs == ("LME_PB_Close", "US_Stock_VT_adj_close")
    assert s.horizon == 3 and s.gap == 1 and s.signs == (1.0, -1.0)
    single = parse_mitsui_pair("target_0", "US_Stock_VT_adj_close", 1)
    assert single.kind == TargetKind.LOG_RETURN and not single.is_pair


@pytest.mark.parametrize("kind,legs", [(TargetKind.LOG_RETURN, ("A", "B")), (TargetKind.RETURN_SPREAD, ("A",))])
def test_invalid_specs(kind, legs):
    with pytest.raises(ValueError):
        TargetSpec("bad", kind, legs, 1)
    with pytest.raises(ValueError):
        TargetSpec("bad", TargetKind.LOG_RETURN, ("A",), 0)
