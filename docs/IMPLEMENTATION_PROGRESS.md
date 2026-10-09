# Implementation progress log

Concise record of milestones, commands, verification results, blockers and remaining work.
All times are local (2026-10-09).

## Environment

- macOS (Apple Silicon, 10 cores, 16 GB), Python 3.12 in the project-local `.venv`
  (Homebrew's default 3.14 lacked wheels for some dependencies).
- Installed: numpy 2.5, pandas 3.0, scikit-learn 1.9, LightGBM 4.7, torch 2.14 (CPU),
  matplotlib 3.11, pyyaml, pytest, pyarrow; streamlit 1.65 (dashboard only).
- **Blocker found and resolved.** torch and LightGBM bundle different OpenMP runtimes. In one
  process on macOS this segfaults (torch imported first) or deadlocks (LightGBM first). The
  fix: each model trains in a fresh spawned process, torch is imported lazily, and LightGBM
  drops to one thread whenever torch is already loaded (tests).

## Milestones

| # | Milestone | Status | Verification |
|---|---|---|---|
| 1 | Repository and data audit | done | `cli audit` → `results/audit/data_audit.json`; label formula reproduced (max err 9.9e-7, 742,640 cells); `test.csv` identical to `train.csv` rows 1827–1960 |
| 2 | Legacy preserved | done | `legacy/original/` verbatim; SHA-256 recorded in `legacy/README.md`; defects in `docs/LEGACY_AUDIT.md` (reproduced: synthetic-data crash, LSTM constructor crash, (8,32) output shape, shallow-copy early stopping, 4 folds for `n_splits=5`) |
| 3 | Data layer: schema, MITSUI adapter, generic adapter, synthetic fixture, `TargetSpec` | done | `tests/test_targets.py`, `tests/test_data.py` |
| 4 | Causal features (38, four groups) and modelling panel | done | `tests/test_causality.py` (bit-identical under future perturbation, synthetic and real data) |
| 5 | Label-availability walk-forward, purged internal validation | done | `tests/test_splits.py` |
| 6 | Benchmarks, Ridge, LightGBM, LSTM, direct pair models, four ensemble schemes | done | `tests/test_models.py`; reproducibility with identical seeds |
| 7 | Metrics (competition score, HAC, block bootstrap, paired tests) | done | `tests/test_metrics.py` (matches a reference implementation of the official metric) |
| 8 | FAST real-data run | done | 5.4 min training; a second full run reproduced every development score exactly; a clean from-scratch `all --profile fast` (7.2 min) reproduced all metrics and predictions bit-for-bit |
| 9 | Design freeze, then one-shot holdout evaluation (FAST) | done | `evaluate --profile fast` 03:09 |
| 10 | Risk layer: conformal intervals, disagreement, hypothetical backtest, timing diagnostic | done | `results/fast/metrics/*` |
| 11 | Figures and machine-generated tables | done | `reports/fast/` |
| 12 | FULL profile run | done | 39.4 min (`all --profile full`); conclusions consistent with FAST; `reports/full/` |
| 13 | Generic-adapter demo on public FRED energy data | done | 2.4 min training; exposed and fixed a row-position vs timestamp bug in the report layer; `reports/demo_fred_energy/` |
| 14 | Documentation set and Streamlit dashboard | done | dashboard executed headlessly with `streamlit.testing.AppTest`: 6 tabs, no exceptions |

## Phase 2: volatility research (2026-10-09)

| # | Milestone | Status | Verification |
|---|---|---|---|
| 15 | Literature and competition review | done | Competition write-ups of top teams not publicly reachable. The meetup deck documents 8th place (lag-4 target values as inputs) and 15th place (recency-weighted rolling ensemble). The competition's lagged-label files independently confirm the label-availability rule (`date_id − label_date_id = lag + 1`, 360/360 rows, exact value match) |
| 16 | `arch` 8.0 and `statsmodels` 0.15 added | done | MCS / SPA from `arch.bootstrap` |
| 17 | Part I diagnostics (`vol-diagnostics`) | done | 106 assets + 424 spreads; estimation-sample decisions only |
| 18 | Pre-registration (`docs/VOLATILITY_PREREGISTRATION.md`) | done | written before any volatility-forecast or Part III evaluation |
| 19 | Causal volatility engine (const/HV/EWMA/GARCH-N/GARCH-t/GJR-t, re-estimated every 21 days) | done | causality test (perturb future returns), equality with reference recursions, multi-step equality with `arch` analytic forecasts |
| 20 | Bug found by test: flat multi-step forecasts for integrated GARCH estimates | fixed | exact recursion; 2–3% of refits affected; Part II re-run, Part III re-trained (deviation #1) |
| 21 | Part II volatility-forecast evaluation (`vol-forecast`) | done | panel DM, per-asset DM + BH-FDR, MCS, Mincer–Zarnowitz, regimes, Garman–Klass sensitivity |
| 22 | Target-level features (R / Vn / Vs / C, 47 features) and pooled target model | done | causality and label-availability tests (`tests/test_volatility.py`) |
| 23 | Part III: development run, freeze record, one-shot holdout (`vol-returns`) | done | Holm, MCS, SPA; pooled-OOS and holdout-half analyses |
| 24 | Robustness profiles `vol_robust` (refit 42, 15-leaf LGBM) and `vol_rolling` (750-date window) | done | specified in the freeze record before running |
| 25 | FRED replication with a fresh 2026 holdout (`vol_fred`) | done | 2026 data never examined before pre-registration |
| 26 | Exploratory decomposition (`vol_explore`: market vs own volatility state) | done | labelled post hoc everywhere it appears (deviation #3) |
| 27 | Determinism check | done | clean from-scratch `vol-all --profile vol` (6.6 min): all 11 result tables and every prediction bit-identical to the first run | labelled post hoc |

## Key commands run

```bash
/opt/homebrew/bin/python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # + torch lightgbm
python -m commodity_research.cli audit
python -m commodity_research.cli train    --profile fast      # 5.4 min
python -m commodity_research.cli evaluate --profile fast      # 52 s
python -m commodity_research.cli report   --profile fast
python -m commodity_research.cli all      --profile full      # 39.4 min
python scripts/fetch_fred_energy.py && python -m commodity_research.cli all --profile demo_fred_energy
python scripts/exploratory_analysis.py --profile fast
python -m commodity_research.cli vol-diagnostics --profile vol
python -m commodity_research.cli vol-forecast    --profile vol
python -m commodity_research.cli train           --profile vol
python -m commodity_research.cli vol-returns     --profile vol --periods dev   # before the freeze
python -m commodity_research.cli vol-returns     --profile vol --periods all   # one-shot holdout
python -m commodity_research.cli vol-importance  --profile vol
python -m commodity_research.cli vol-report      --profile vol
python -m commodity_research.cli vol-all         --profile vol_fred            # replication
python -m pytest
```

## Decisions made after seeing out-of-sample data

See [VALIDATION_AND_LEAKAGE.md §6](VALIDATION_AND_LEAKAGE.md#6-researcher-degrees-of-freedom-full-disclosure).
In short:

- Cross-sectional label demeaning was adopted after development-period evidence; the
  raw-label ablation is kept.
- A univariate diagnostic displayed holdout feature ICs (no action taken, disclosed).
- `ens_best` and the timing diagnostic were added before the holdout evaluation.
- FULL was configured after the FAST holdout, changing only compute settings.

## Remaining work and continuation checklist

- [ ] Multiple-testing-robust comparison (Hansen SPA / Romano–Wolf) across the model set.
- [ ] Rolling-window vs expanding-window study (`split.window` already supported).
- [ ] Futures term-structure (carry) features. They need a data source with contract months.
- [ ] Optional GPU/MPS LSTM run (`params.device: mps`). Not needed for CPU feasibility.
- To resume from a clean state: `rm -rf results/fast && python -m commodity_research.cli all --profile fast`.
  Results are deterministic for a given seed and data fingerprint.
