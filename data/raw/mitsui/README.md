# MITSUI raw data (not included)

The four competition files are **not** in this repository. They are provided under the Kaggle
competition rules and cannot be redistributed.

1. Download them from the competition page (a free Kaggle account is required):
   <https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data>
2. Place these files in this folder:
   - `train.csv` (features, 1,961 dates)
   - `train_labels.csv` (424 target labels)
   - `target_pairs.csv` (target definitions)
   - `test.csv` (optional; used only by the leakage audit)

Then run, for example:

```bash
python -m commodity_research.cli audit
python -m commodity_research.cli train --profile fast
```

Without these files, everything that reads the saved results still works: the tables, the
figures, the Streamlit dashboard and the test suite (data-dependent tests are skipped).
