import pandas as pd
from rep_lib import *


def s2(c):
    R = c.R
    mp = R["missing_partial"]
    c.heading("II")
    c.para("This section lists every cleaning step we did. Nothing was edited by hand in the file. Everything is done in code "
           "(`02_gold_analysis.py`), so anyone can run it again and get the same result.")
    c.bullet("Initial Data Inspection:",
             f"The raw file has {R['raw_shape'][0]:,} rows and {R['raw_shape'][1]} columns. Each row is a calendar date where at least one market was open. "
             "All 7 price columns are `float64`, which is right. The `Date` column was read as text, so we changed it to `datetime` and used it as the index.")
    c.sub(f"**Missing values:** every column had around 1,040 missing values (Figure {c.fno('fig01_missing_values')}, left). "
          f"It looked bad, but **{R['empty_rows']:,} rows were completely empty**: {R['empty_rows_sunday']} Sundays and {R['empty_rows_weekday']} weekday holidays when every market was closed. "
          f"Those rows have no information, so we dropped them and kept **{R['shape_after_drop_empty'][0]:,} rows**.")
    c.sub(f"**Small gaps left:** only {R['rows_with_any_missing']} rows still had a missing value ({mp['Gold']} in Gold, {mp['Silver']} Silver, {mp['Oil']} Oil, "
          f"{mp['USD_Index']} USD Index, {mp['Yield_10Y']} Yield, {mp['VIX']} VIX, {mp['SP500']} S&P 500). The longest gap was **1 day** in every column "
          f"(one market was closed while the others were open), so we filled each gap with the previous close (forward fill). After that there were {R['missing_after_ffill']} missing values.")
    c.sub(f"**Duplicates:** no repeated dates ({R['duplicated_dates']}). The raw file showed {R['duplicated_rows_raw']:,} duplicated rows, but those were just the empty rows above. "
          f"After dropping them, `data.duplicated().sum()` gives {R['duplicated_rows_after_drop']}.")
    c.figure("fig01_missing_values", "Missing values in the raw file (left) and after removing the empty rows (right).", max_w=6.0)
    c.sub(f"**One invalid value:** WTI crude oil closed at **{sgn(R['oil_invalid_value'])} USD** on {R['oil_invalid_date']}. This really happened (a negative oil price), "
          f"but it is not a normal price and it breaks log transformations. We replaced it with the previous close, {n2(R['oil_replaced_by'])} USD.")
    d = pd.read_csv(PROJ + r"\table_describe.csv", index_col=0)
    rows = [[i, n0(r["count"]), n2(r["mean"]), n2(r["std"]), n2(r["min"]), n2(r["50%"]), n2(r["max"])] for i, r in d.iterrows()]
    c.table("Summary statistics of the clean data.", ["Variable", "Count", "Mean", "Std", "Min", "Median", "Max"], rows,
            widths=[1.2, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9])
