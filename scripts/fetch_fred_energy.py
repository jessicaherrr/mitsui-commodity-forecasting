"""Download a small public daily energy-commodity panel from FRED (for the generic adapter demo).

Series (all daily, public domain US-government data redistributed by FRED):

    DCOILWTICO     WTI crude oil spot, Cushing OK (USD/bbl)            source: EIA
    DCOILBRENTEU   Brent crude oil spot, Europe (USD/bbl)              source: EIA
    DHHNGSP        Henry Hub natural gas spot (USD/MMBtu)               source: EIA
    DHOILNYH       No.2 heating oil, New York Harbor (USD/gal)          source: EIA
    DGASNYH        Conventional gasoline, New York Harbor (USD/gal)     source: EIA
    DTWEXBGS       Nominal broad U.S. dollar index (Jan 2006 = 100)     source: Federal Reserve H.10

Citation: Federal Reserve Bank of St. Louis, FRED, https://fred.stlouisfed.org/ (series as above).
Non-positive prints (e.g. WTI on 2020-04-20) are kept in the CSV and treated as missing by the
adapter, because a log return is undefined for them.

Usage:  python scripts/fetch_fred_energy.py [--start 2010-01-01] [--out data/external/fred_energy_long.csv]
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

SERIES = {
    "DCOILWTICO": ("WTI", "energy"),
    "DCOILBRENTEU": ("BRENT", "energy"),
    "DHHNGSP": ("HENRY_HUB_GAS", "energy"),
    "DHOILNYH": ("NYH_HEATING_OIL", "energy"),
    "DGASNYH": ("NYH_GASOLINE", "energy"),
    "DTWEXBGS": ("USD_BROAD", "fx"),
}
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch(sid: str, timeout: int = 30) -> pd.DataFrame:
    with urllib.request.urlopen(URL.format(sid=sid), timeout=timeout) as r:  # noqa: S310 (fixed https host)
        raw = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "timestamp", sid: "close"})
    df["close"] = pd.to_numeric(df["close"], errors="coerce")  # FRED uses "." for missing
    return df[["timestamp", "close"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="data/external/fred_energy_long.csv")
    a = ap.parse_args(argv)
    frames = []
    for sid, (asset, cls) in SERIES.items():
        df = fetch(sid)
        df = df[(df.timestamp >= a.start) & (df.timestamp <= a.end)].dropna(subset=["close"])
        df["asset_id"], df["asset_class"], df["market"] = asset, cls, "FRED"
        df["source_series"] = sid
        frames.append(df)
        print(f"{sid:14s} -> {asset:16s} {len(df):5d} rows  {df.timestamp.min()} .. {df.timestamp.max()}")
    out = pd.concat(frames, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
