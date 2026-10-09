# External data

The two files here are public-domain US government data, redistributed with attribution:

| File | Content | Source |
|---|---|---|
| `fred_energy_long.csv` | Daily spot prices, 2010-01 to 2025-12: WTI, Brent, Henry Hub natural gas, NY Harbor heating oil and gasoline, and the nominal broad USD index | EIA series DCOILWTICO, DCOILBRENTEU, DHHNGSP, DHOILNYH, DGASNYH; Federal Reserve H.10 series DTWEXBGS, via FRED (Federal Reserve Bank of St. Louis) |
| `fred_energy_long_2026.csv` | The same series extended through 2026-10-06 (used as the fresh 2026 holdout in the volatility study) | as above |

Both files use the long format the generic adapter expects:
`timestamp, close, asset_id, asset_class, market, source_series`.

To regenerate them, or to fetch a newer extract:

```bash
python scripts/fetch_fred_energy.py                                  # 2010-2025
python scripts/fetch_fred_energy.py --end 2026-12-31 --out data/external/fred_energy_long_2026.csv
```

Any other file placed in this folder is ignored by git.
