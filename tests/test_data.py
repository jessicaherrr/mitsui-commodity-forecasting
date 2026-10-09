import numpy as np
import pandas as pd
import pytest

from commodity_research.data import load_long_format, make_synthetic_market, to_long_format, verify_official_labels
from commodity_research.targets import TargetKind

from .conftest import HAS_MITSUI, MITSUI


def test_synthetic_is_flagged():
    md = make_synthetic_market(n_dates=200, n_assets=5)
    assert md.is_synthetic
    assert any("SYNTHETIC" in w for w in md.validate())
    assert "SYNTHETIC" in md.provenance


def test_synthetic_is_deterministic():
    a = make_synthetic_market(seed=3).close
    b = make_synthetic_market(seed=3).close
    pd.testing.assert_frame_equal(a, b)


def test_generic_adapter_round_trip():
    md = make_synthetic_market(n_dates=150, n_assets=4)
    long = to_long_format(md.close.set_index(pd.date_range("2020-01-01", periods=150, freq="B")), md.assets)
    g = load_long_format(long, horizons=[1, 5], gap=1, pairs=[(md.close.columns[0], md.close.columns[1])])
    assert g.close.shape == (150, 4)
    kinds = [t.kind for t in g.targets]
    assert kinds.count(TargetKind.LOG_RETURN) == 8 and kinds.count(TargetKind.RETURN_SPREAD) == 2
    assert g.horizons == [1, 5] and g.gap == 1
    np.testing.assert_allclose(g.close[md.close.columns].to_numpy(), md.close.to_numpy(), equal_nan=True)


def test_generic_adapter_validation_errors():
    with pytest.raises(ValueError, match="missing required columns"):
        load_long_format(pd.DataFrame({"timestamp": ["2020-01-01"], "close": [1.0]}))
    dup = pd.DataFrame({"timestamp": ["2020-01-01"] * 2, "asset_id": ["X"] * 2, "close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        load_long_format(dup)


def test_market_data_validation_rejects_unsorted_index():
    md = make_synthetic_market(n_dates=100, n_assets=3)
    md.close = md.close.iloc[::-1]
    with pytest.raises(ValueError, match="chronolog"):
        md.validate()


@pytest.mark.realdata
@pytest.mark.skipif(not HAS_MITSUI, reason="MITSUI data not available")
def test_official_labels_reproduced(mitsui):
    md, _, _ = mitsui
    assert len(md.targets) == 424 and len(md.universe) == 106 and md.horizons == [1, 2, 3, 4]
    ver = verify_official_labels(md)
    assert ver["max_abs_err"].max() < 1e-6
    # every official label we cannot recompute needs prices beyond the end of train.csv
    assert (ver["n_official_only"] == ver["n_official_only_beyond_sample"]).all()
    assert ver["n_recomputed_only"].sum() == 0


@pytest.mark.realdata
@pytest.mark.skipif(not HAS_MITSUI, reason="MITSUI data not available")
def test_test_csv_is_not_independent():
    train = pd.read_csv(MITSUI / "train.csv").set_index("date_id")
    test = pd.read_csv(MITSUI / "test.csv").set_index("date_id")
    common = [c for c in test.columns if c in train.columns]
    a, b = test[common], train.loc[test.index, common]
    assert set(test.index) <= set(train.index)
    assert ((a == b) | (a.isna() & b.isna())).all().all()


@pytest.mark.realdata
@pytest.mark.skipif(not HAS_MITSUI, reason="MITSUI data not available")
def test_composed_asset_labels_match_official(mitsui):
    from commodity_research.features import target_level_labels

    _, _, panel = mitsui
    tl = target_level_labels(panel)
    ev = panel.eval_labels.to_numpy()
    m = np.isfinite(tl) & np.isfinite(ev)
    assert np.abs(tl[m] - ev[m]).max() < 1e-6
