import numpy as np
import pytest

from commodity_research.validation import internal_validation, train_dates, walk_forward_folds


def test_folds_partition_the_test_range_without_overlap():
    folds = walk_forward_folds(100, 437, 50)
    covered = np.concatenate([f.test for f in folds])
    np.testing.assert_array_equal(covered, np.arange(100, 437))
    assert all(f.test[0] == f.refit for f in folds)


@pytest.mark.parametrize("h", [1, 2, 3, 4])
@pytest.mark.parametrize("gap", [0, 1])
def test_training_labels_are_observable_at_refit(h, gap):
    for fold in walk_forward_folds(300, 600, 37):
        tr = train_dates(fold.refit, h, gap, first=50)
        assert (tr + gap + h <= fold.refit).all()  # label observable at the refit close
        assert tr.max() == fold.refit - gap - h  # and nothing observable is wasted
        assert np.intersect1d(tr, fold.test).size == 0
        # information intervals of train labels end before the first test label starts
        assert tr.max() + gap + h <= fold.test.min() + gap  # may share a boundary price, never a return


def test_rolling_window():
    tr = train_dates(500, 2, 1, first=0, window=100)
    assert len(tr) == 100 and tr.max() == 497


def test_internal_validation_is_purged():
    tau, gap, h_max = 800, 1, 4
    for h in [1, 2, 3, 4]:
        fit, val = internal_validation(tau, h, gap, first=100, val_len=60, h_max=h_max)
        assert (val + gap + h <= tau).all()  # validation labels observable at tau
        assert (fit + gap + h < val.min()).all()  # fit labels observable before validation starts
        assert np.intersect1d(fit, val).size == 0
