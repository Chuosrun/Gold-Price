"""
Step 1 - Download the raw data (run once).

Source: Yahoo Finance public chart endpoint (daily closing prices, 2010-01-01 -> today).
Only pandas + the Python standard library are used. The raw file is saved so that the
whole project is reproducible even if the website changes later.

Output: gold_raw.csv  (one row per calendar date on which AT LEAST ONE market traded;
        NO cleaning is done here - missing values are kept on purpose for Section II).
"""
import json
import urllib.parse
import urllib.request

import pandas as pd

SYMBOLS = {
    "Gold": "GC=F",         # Gold futures (USD / troy ounce)   <- TARGET
    "Silver": "SI=F",       # Silver futures (USD / troy ounce)
    "Oil": "CL=F",          # WTI crude oil futures (USD / barrel)
    "USD_Index": "DX-Y.NYB",  # US Dollar Index
    "Yield_10Y": "^TNX",    # 10-year US Treasury yield (%)
    "VIX": "^VIX",          # CBOE volatility ("fear") index
    "SP500": "^GSPC",       # S&P 500 stock index
}


def fetch_close(symbol: str) -> pd.Series:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?period1=1262304000&period2=1893456000"
        "&interval=1d&events=history"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    dates = pd.to_datetime(res["timestamp"], unit="s").normalize()
    close = res["indicators"]["quote"][0]["close"]
    s = pd.Series(close, index=dates, dtype="float64")
    # a few symbols return the same date twice (intraday snapshot) -> keep the last one
    return s[~s.index.duplicated(keep="last")]


if __name__ == "__main__":
    cols = {}
    for name, sym in SYMBOLS.items():
        cols[name] = fetch_close(sym)
        print(f"{name:10s} {sym:10s} rows={len(cols[name])}")
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "Date"
    df.to_csv("gold_raw.csv")
    print("\nSaved gold_raw.csv ->", df.shape, df.index.min().date(), "to", df.index.max().date())
