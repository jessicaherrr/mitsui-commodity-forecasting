"""End-to-end smoke test of train -> evaluate -> report on the SYNTHETIC smoke profile."""

import json

import pytest

from commodity_research.config import load_config


@pytest.mark.slow
def test_pipeline_end_to_end(tmp_path):
    from commodity_research.experiments.evaluate import run_evaluation
    from commodity_research.experiments.report import run_report
    from commodity_research.experiments.runner import run_training

    cfg = load_config("smoke")
    cfg["output_root"] = str(tmp_path / "results")
    cfg["report_root"] = str(tmp_path / "reports")
    manifest = run_training(cfg)
    assert manifest["is_synthetic"] is True
    ok = {m for m, v in manifest["models"].items() if v.get("status") == "ok"}
    assert {"hist_mean", "ridge", "lgbm", "lstm", "ens_equal", "ens_regime", "ens_online"} <= ok

    result = run_evaluation(cfg)
    out = tmp_path / "results" / "smoke" / "metrics"
    for f in ["model_comparison.csv", "paired_tests.csv", "regime_ic.csv", "by_horizon.csv", "daily_ic.csv", "backtest_holdout.csv"]:
        assert (out / f).exists(), f
    assert result["is_synthetic"] is True

    run_report(cfg)
    md = (tmp_path / "reports" / "smoke" / "results_tables.md").read_text()
    assert "SYNTHETIC" in md
    figs = list((tmp_path / "reports" / "smoke" / "figures").glob("*.png"))
    assert len(figs) >= 4
    summary = json.loads((out / "summary.json").read_text())
    assert "ridge" in summary["summary"]
