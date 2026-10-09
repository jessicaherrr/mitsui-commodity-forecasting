# Legacy code (original Kaggle submission pipeline)

`original/` holds the competition code **verbatim** (only moved, never edited):

| File | SHA-256 (first 16 hex) |
|---|---|
| `main_workflow.py` | `a16e55505bb67e47` |
| `preprocess_timeseries.py` | `00c2a60d5c844444` |
| `run_pipeline.py` | `c9f9bcd1ad3cbc80` |
| `README_original.md` | `30fd4e8f7c06e2db` |

It is kept so the project's provenance can be checked. **It is not part of the active
codebase, and its outputs are not valid evidence.** See
[`docs/LEGACY_AUDIT.md`](../docs/LEGACY_AUDIT.md) for the defects found. Among them: it trains
on randomly generated synthetic data, targets price levels instead of returns, uses centred
filters and back-filling that leak future data, and its LSTM can be neither constructed nor
trained as written.

To run it anyway, copy the four CSV files from `data/raw/mitsui/` into the working directory
and install `optuna`. It will fail as documented in the audit.
