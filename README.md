# What Drives the Gold Price? A Data Analysis of Market Factors, 2010-2026

Final project for **Data Analysis I** (Lecturer: Dr. HAS Sothea).
Author: Lim Chuosrun (2024588)

The goal of this project is to find out which market factors move the gold price. I built the dataset from Yahoo Finance daily closing prices (2010-01-04 to 2026-11-25, 4,258 trading days), cleaned it, explored it, tested some ideas with hypothesis tests, and then fit regression models.

The full report is in **`Final_Project_Report_Gold_Price_Simple.docx`** (17 pages, written in the course template).

## Data

| Variable | Yahoo Finance symbol | Meaning |
|---|---|---|
| Gold (target) | GC=F | Gold futures, USD per ounce |
| Silver | SI=F | Silver futures, USD per ounce |
| Oil | CL=F | WTI crude oil futures, USD per barrel |
| USD_Index | DX-Y.NYB | US Dollar Index |
| Yield_10Y | ^TNX | 10-year US Treasury yield (%) |
| VIX | ^VIX | Market "fear" index |
| SP500 | ^GSPC | S&P 500 stock index |

## What was done (same order as the report)

1. **Data preprocessing**: dropped 1,031 completely empty rows (Sundays and holidays), forward filled the 1-day gaps, replaced the negative oil price of 2020-04-20, checked duplicates and data types, looked at outliers with boxplots and the IQR rule (kept them, because they are real events), and made the daily return variables.
2. **EDA**: histograms, boxplots, scatter plots, moving averages (50 and 200 days), gold/silver ratio, Pearson and Spearman correlation.
3. **Hypothesis tests**: z-test (mean daily return = 0), two t-tests (stress vs calm days, dollar up vs down), chi-square test of independence (gold direction vs dollar direction).
4. **Models**: simple linear regression, multiple linear regression (with VIF) and logistic regression (gold up or down), using a time-ordered 80/20 split (no shuffling).
5. **Results**: RMSE, MAE, R2, accuracy, precision, recall, F1, AUC and a confusion matrix, compared with a simple baseline.

## Main findings

- **The dollar is the clearest driver.** On days when the USD index rises, gold falls about 0.29% on average, and when it falls, gold gains about 0.34% (t-test, p < 0.001). Gold goes up on 67% of the days when the dollar falls but only 38% when it rises (chi-square test, p < 0.001). The 10-year yield is second.
- **Market fear (VIX) does not change the average gold return** (t-test p = 0.30, chi-square p = 0.14).
- **The average daily return is not clearly different from 0** (z = 1.63, p = 0.10).
- **Correlation on price levels can lie.** Gold vs S&P 500 is r = 0.86 on price levels, but only r = 0.07 on daily changes, because both just trended up.
- **The models learn a little, not a lot.** Multiple regression has a test R2 of about 0.09, and the simple regression on the dollar alone is slightly better (0.11) when both are fitted on the same training set. Logistic regression gets 61.9% accuracy against a 54.3% baseline (AUC 0.68).
- The models use the other markets' changes on the **same day**, so they explain a day but do not forecast tomorrow.

## Folder structure

```
Final_Project_Report_Gold_Price_Simple.docx   the report (editable Word file)
Final_project_template.docx                   the course template
gold_project/
  01_download_data.py      downloads the raw data from Yahoo Finance -> gold_raw.csv
  02_gold_analysis.py      cleaning and EDA -> gold_clean.csv, results.json, figures fig01-fig21
  03_gold_simple.py        tests and regression models -> results_simple.json, figures s01-s06
  gold_raw.csv             raw data
  gold_clean.csv           clean data with the new variables
  results.json             numbers from 02_gold_analysis.py
  results_simple.json      numbers from 03_gold_simple.py
  table_describe.csv       summary statistics table
  table_annual_summary.csv gold by year
  figures/                 the 17 figures used in the report
  report_build/            code that writes the Word report from the template
evidence/                  PDF export of the report, run logs and file checks
```

## How to run

Python 3.13 with these libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn` (the same ones used in the labs). Building the Word report also needs `python-docx` and `Pillow`.

```
cd gold_project
python 01_download_data.py     # optional, needs internet. The raw data is already included
python 02_gold_analysis.py
python 03_gold_simple.py
cd report_build
python build_simple.py         # writes Final_Project_Report_Gold_Price_Simple.docx in the main folder
```

Note: `01_download_data.py` downloads the newest prices, so a new download will not give exactly the same numbers as the report. Keep `gold_raw.csv` to get the same numbers.

## Limitations

- Only 5 drivers and daily closing prices. Inflation, real interest rates, central bank buying and ETF flows are not included.
- Gold and silver are futures prices, not spot prices.
- Linear models only, and one test period (2023-2026) that was more volatile than the training years.
- Daily returns are not perfectly normal (heavy tails), so p-values should be read with some care.
