"""Command-line entry point.

    python -m commodity_research.cli audit
    python -m commodity_research.cli train    --profile fast
    python -m commodity_research.cli evaluate --profile fast
    python -m commodity_research.cli report   --profile fast
    python -m commodity_research.cli all      --profile fast      # train + evaluate + report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from .config import load_config, results_dir
from .logging_utils import setup_logging

log = logging.getLogger("commodity_research")


def _cmd_audit(args) -> int:
    from .experiments.audit import run_audit

    cfg = load_config(args.profile)
    summary = run_audit(cfg)
    print(json.dumps(summary["headline"], indent=2))
    return 0


def _cmd_train(args) -> int:
    from .experiments.runner import run_training

    cfg = load_config(args.profile)
    setup_logging(results_dir(cfg) / "run.log")
    run_training(cfg, only=[] if getattr(args, "ensembles_only", False) else args.only)
    return 0


def _cmd_evaluate(args) -> int:
    from .experiments.evaluate import run_evaluation

    cfg = load_config(args.profile)
    setup_logging(results_dir(cfg) / "run.log")
    run_evaluation(cfg)
    return 0


def _cmd_report(args) -> int:
    from .experiments.report import run_report

    cfg = load_config(args.profile)
    setup_logging(results_dir(cfg) / "run.log")
    run_report(cfg)
    return 0


def _vol(fn_name: str):
    def run(args) -> int:
        from .experiments import vol_report, volatility

        cfg = load_config(args.profile)
        setup_logging(results_dir(cfg) / "run.log")
        if fn_name == "returns":
            volatility.run_vol_return_analysis(cfg, ("dev",) if args.periods == "dev" else ("dev", "holdout"))
        elif fn_name == "importance":
            vol_report.run_importance(cfg)
        elif fn_name == "report":
            vol_report.run_vol_report(cfg)
        else:
            {"diagnostics": volatility.run_diagnostics, "forecast": volatility.run_vol_forecast}[fn_name](cfg)
        return 0

    return run


def _cmd_vol_all(args) -> int:
    t0 = time.time()
    args.periods = "all"
    for step in ("diagnostics", "forecast"):
        _vol(step)(args)
    _cmd_train(args)
    for step in ("returns", "importance", "report"):
        _vol(step)(args)
    log.info("volatility pipeline finished in %.1f min", (time.time() - t0) / 60)
    return 0


def _cmd_all(args) -> int:
    t0 = time.time()
    for fn in (_cmd_train, _cmd_evaluate, _cmd_report):
        rc = fn(args)
        if rc:
            return rc
    log.info("all stages finished in %.1f min", (time.time() - t0) / 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="commodity_research", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, helptext in [
        ("audit", _cmd_audit, "verify data, target definitions, timing and test.csv overlap"),
        ("train", _cmd_train, "walk-forward training: dev tuning then final holdout"),
        ("evaluate", _cmd_evaluate, "compute out-of-sample metrics and experiment tables"),
        ("report", _cmd_report, "generate figures and markdown result tables"),
        ("all", _cmd_all, "train + evaluate + report"),
        ("vol-diagnostics", _vol("diagnostics"), "Part I: stylised facts, ARCH/GARCH diagnostics"),
        ("vol-forecast", _vol("forecast"), "Part II: out-of-sample volatility-forecast comparison"),
        ("vol-returns", _vol("returns"), "Part III: return-forecast statistics for the volatility arms"),
        ("vol-importance", _vol("importance"), "Part III: feature-group importance of tgbm_RVC"),
        ("vol-report", _vol("report"), "figures and tables for the volatility study"),
        ("vol-all", _cmd_vol_all, "Parts I-III end to end"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--profile", default="fast", help="config profile name in configs/ or a yaml path")
        if name in ("train", "all", "vol-all"):
            p.add_argument("--only", nargs="*", default=None, help="train only these model names")
            p.add_argument("--ensembles-only", action="store_true", help="re-fit ensembles from saved forecasts")
        if name == "vol-returns":
            p.add_argument("--periods", choices=["dev", "all"], default="all",
                           help="'dev' = development dates only (use before the design freeze)")
        p.set_defaults(func=fn)
    args = parser.parse_args(argv)
    if args.command == "audit":
        setup_logging()
    try:
        return args.func(args)
    except FileNotFoundError as err:  # missing competition files or saved results: explain, don't trace
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
