# %% [markdown]
# # **Final Project - What Drives the Gold Price?**
# **Course**: **INF-604: Data Analysis I** <br>
# **Lecturer**: **Sothea HAS, PhD**
#
# * Lim Chuosrun
# * 2024588
#
# This notebook contains the full pipeline used in the report: data inspection, preprocessing,
# exploratory data analysis (EDA), model development and evaluation.
# The raw data (`gold_raw.csv`) is downloaded by `01_download_data.py` (Yahoo Finance, daily closing prices, 2010 -> 2026).
# Only libraries seen in the labs are used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `sklearn`.

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures without opening windows (remove this line to see them in Jupyter)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 11, "axes.labelsize": 10})

FIG = "figures"
os.makedirs(FIG, exist_ok=True)
RES = {}          # every number used in the report is stored here -> results.json


def save(name):
    """Save the current figure to figures/<name>.png and show it."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close("all")


# %% [markdown]
# # **II. Data Preprocessing**
# ## **1. Initial data inspection**

# %%
raw = pd.read_csv("gold_raw.csv")
print("Shape:", raw.shape)
print(raw.dtypes)
raw.head()

# %%
# 'Date' was read as text -> convert it to datetime and use it as the index
raw["Date"] = pd.to_datetime(raw["Date"])
raw = raw.set_index("Date")

RES["raw_shape"] = list(raw.shape)
RES["date_min"] = str(raw.index.min().date())
RES["date_max"] = str(raw.index.max().date())

# missing values, duplicates
miss_raw = raw.isna().sum()
RES["missing_raw"] = miss_raw.to_dict()
RES["duplicated_dates"] = int(raw.index.duplicated().sum())
RES["duplicated_rows_raw"] = int(raw.duplicated().sum())

# rows with NO data at all (all 7 columns are NaN)
empty = raw[raw.isna().all(axis=1)]
RES["empty_rows"] = int(len(empty))
RES["empty_rows_sunday"] = int((empty.index.dayofweek == 6).sum())
RES["empty_rows_weekday"] = int((empty.index.dayofweek < 5).sum())
print(miss_raw)
print("Duplicated dates:", RES["duplicated_dates"], "| duplicated rows:", RES["duplicated_rows_raw"])
print("Rows with no data at all:", len(empty), "(Sundays:", RES["empty_rows_sunday"], ", weekday holidays:", RES["empty_rows_weekday"], ")")


# %%
# drop the completely empty rows (non-trading days) and look at what is still missing
data = raw.dropna(how="all").copy()
RES["shape_after_drop_empty"] = list(data.shape)
RES["duplicated_rows_after_drop"] = int(data.duplicated().sum())
miss_partial = data.isna().sum()
RES["missing_partial"] = miss_partial.to_dict()
RES["rows_with_any_missing"] = int(data.isna().any(axis=1).sum())


def longest_gap(s):
    """Longest run of consecutive NaN in a series."""
    return int(s.isna().astype(int).groupby(s.notna().cumsum()).sum().max())


RES["longest_gap"] = {c: longest_gap(data[c]) for c in data.columns}
print(miss_partial)
print("Longest gaps:", RES["longest_gap"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=miss_raw.index, y=miss_raw.values, color="steelblue", ax=axes[0])
axes[0].set_title(f"Missing values in the raw file ({len(raw):,} rows)")
axes[0].set_ylabel("Number of missing values")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=45)
sns.barplot(x=miss_partial.index, y=miss_partial.values, color="darkorange", ax=axes[1])
axes[1].set_title(f"After removing the {len(empty):,} empty rows ({len(data):,} rows)")
axes[1].set_ylabel("Number of missing values")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=45)
save("fig01_missing_values")

# %%
# the remaining gaps are 1-day gaps (one market closed on a day the others were open) -> forward fill
data = data.ffill()
RES["missing_after_ffill"] = int(data.isna().sum().sum())
print("Missing after forward fill:", RES["missing_after_ffill"])

# invalid values: any price <= 0 ?
RES["nonpositive_counts"] = {c: int((data[c] <= 0).sum()) for c in data.columns}
print(RES["nonpositive_counts"])
bad_oil = data[data["Oil"] <= 0]
print(bad_oil)

# %%
# WTI crude oil closed at a NEGATIVE price on 2020-04-20 (a real event, but it breaks log transformations)
# -> we treat it as invalid and replace it by the previous close
RES["oil_invalid_date"] = str(bad_oil.index[0].date())
RES["oil_invalid_value"] = float(bad_oil["Oil"].iloc[0])
data.loc[data["Oil"] <= 0, "Oil"] = np.nan
data["Oil"] = data["Oil"].ffill()
RES["oil_replaced_by"] = float(data.loc[RES["oil_invalid_date"], "Oil"])
RES["clean_shape"] = list(data.shape)
RES["dtypes"] = {c: str(t) for c, t in data.dtypes.items()}

desc = data.describe().T
RES["describe"] = desc.round(3).to_dict(orient="index")
desc.round(3).to_csv("table_describe.csv")
print(desc.round(3))

# %% [markdown]
# ## **2. Identify and handle outliers**

# %%
price_cols = list(data.columns)          # the 7 original columns (levels)


def iqr_table(d):
    rows = []
    for c in d.columns:
        q1, q3 = d[c].quantile(0.25), d[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_low, n_high = int((d[c] < lo).sum()), int((d[c] > hi).sum())
        rows.append({"column": c, "Q1": q1, "Q3": q3, "lower": lo, "upper": hi,
                     "n_low": n_low, "n_high": n_high, "n_out": n_low + n_high,
                     "pct_out": 100 * (n_low + n_high) / len(d)})
    return pd.DataFrame(rows).set_index("column")


out_levels = iqr_table(data)
RES["outliers_levels"] = out_levels.round(3).to_dict(orient="index")
print(out_levels.round(2))

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, c in enumerate(price_cols):
    sns.boxplot(data, y=c, ax=axes[i // 4, i % 4], color="lightsteelblue")
    axes[i // 4, i % 4].set_title(c)
    axes[i // 4, i % 4].set_ylabel("")
axes[1, 3].axis("off")
save("fig02_boxplots_levels")


# %%
# daily changes: log return for prices/indices, simple difference for the yield (already a % rate)
ret = pd.DataFrame(index=data.index)
for c in ["Gold", "Silver", "Oil", "USD_Index", "SP500", "VIX"]:
    ret[c] = np.log(data[c]).diff()
ret["Yield_10Y"] = data["Yield_10Y"].diff()
ret = ret.dropna()
RES["returns_shape"] = list(ret.shape)

out_ret = iqr_table(ret)
RES["outliers_returns"] = out_ret.round(4).to_dict(orient="index")
print(out_ret[["n_low", "n_high", "n_out", "pct_out"]].round(2))

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, c in enumerate(ret.columns):
    sns.boxplot(ret, y=c, ax=axes[i // 4, i % 4], color="lightsalmon")
    axes[i // 4, i % 4].set_title(c + " (daily change)")
    axes[i // 4, i % 4].set_ylabel("")
axes[1, 3].axis("off")
save("fig03_boxplots_returns")

# %%
# Are the outliers errors? Look at the most extreme gold days
extreme = ret["Gold"].sort_values()
RES["gold_worst_days"] = {str(d.date()): float(v) for d, v in extreme.head(3).items()}
RES["gold_best_days"] = {str(d.date()): float(v) for d, v in extreme.tail(3).items()}
print(extreme.head(3), extreme.tail(3))

# skewness before / after a log transformation (like Lab 10) to decide which columns to transform
skew_orig = data.skew()
skew_log = np.log(data).skew()
skew_tab = pd.DataFrame({"skew_original": skew_orig, "skew_log": skew_log})
RES["skew_table"] = skew_tab.round(3).to_dict(orient="index")
print(skew_tab.round(2))

# %% [markdown]
# ## **3. Feature engineering**

# %%
df = data.copy()
df["Year"] = df.index.year
df["Period"] = pd.cut(df["Year"], bins=[2009, 2019, 2022, 2026], labels=["2010-2019", "2020-2022", "2023-2026"])
df["Gold_Silver_Ratio"] = df["Gold"] / df["Silver"]                 # EDA only (uses the target)
df["Gold_Return"] = np.log(df["Gold"]).diff()                        # daily log return of gold
df["High_VIX"] = np.where(df["VIX"] > 20, "VIX > 20 (stress)", "VIX <= 20 (calm)")

# price-level model: a driver is log transformed if it is skewed (|skew| > 0.8) and the log reduces the skew
cand = ["USD_Index", "Yield_10Y", "VIX", "SP500", "Oil"]
log_cols = [c for c in cand if abs(skew_orig[c]) > 0.8 and abs(skew_log[c]) < abs(skew_orig[c])]
RES["log_cols"] = log_cols
for c in log_cols:
    df["Log_" + c] = np.log(df[c])
level_features = [("Log_" + c if c in log_cols else c) for c in cand]
RES["level_features"] = level_features
print("Log transformed:", log_cols)
print("Level-model features:", level_features)

# return model: same-day change of each driver -> same-day gold return
ret_features = ["USD_Index", "Yield_10Y", "VIX", "SP500", "Oil"]
RES["ret_features"] = ret_features
df.head()

# %% [markdown]
# # **III. Exploratory Data Analysis**
# ## **1. Univariate analysis**

# %%
# gold & silver over time with key events
gold_peak_2011 = df.loc["2011", "Gold"].idxmax()
vix_peak = df["VIX"].idxmax()
gold_ath = df["Gold"].idxmax()
events = {
    f"2011 peak\n{gold_peak_2011.date()}": gold_peak_2011,
    f"VIX peak (COVID)\n{vix_peak.date()}": vix_peak,
    "Ukraine invasion\n2022-02-24": pd.Timestamp("2022-02-24"),
    f"Highest close\n{gold_ath.date()}": gold_ath,
}
RES["events"] = {k.replace("\n", " "): str(v.date()) for k, v in events.items()}
RES["gold_max"] = [str(gold_ath.date()), float(df["Gold"].max())]
RES["gold_min"] = [str(df["Gold"].idxmin().date()), float(df["Gold"].min())]
RES["gold_first"] = float(df["Gold"].iloc[0])
RES["gold_last"] = [str(df.index[-1].date()), float(df["Gold"].iloc[-1])]
RES["vix_max"] = [str(vix_peak.date()), float(df["VIX"].max())]
RES["gold_peak_2011"] = [str(gold_peak_2011.date()), float(df.loc[gold_peak_2011, "Gold"])]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df.index, df["Gold"], color="goldenrod", lw=1.3, label="Gold (USD/oz)")
ax.set_ylabel("Gold price (USD per ounce)")
ax2 = ax.twinx()
ax2.plot(df.index, df["Silver"], color="gray", lw=1, alpha=0.8, label="Silver (USD/oz)")
ax2.set_ylabel("Silver price (USD per ounce)")
ax2.grid(False)
for i, (lab, d) in enumerate(events.items()):
    ax.axvline(d, color="red", ls="--", lw=0.7)
    ax.text(d, df["Gold"].max() * (0.99 - 0.1 * (i % 3)), lab, fontsize=7,
            ha="right" if d.year >= 2025 else "center", color="red",
            bbox=dict(fc="white", ec="none", alpha=0.7))
ax.set_title("Gold and silver daily closing prices, 2010-2026")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center left")
save("fig04_gold_silver_time")

# %%
# distributions of the 7 original columns
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for i, c in enumerate(price_cols):
    sns.histplot(data[c], kde=True, ax=axes[i // 4, i % 4], color="steelblue")
    axes[i // 4, i % 4].set_title(f"{c} (skew = {skew_orig[c]:.2f})")
    axes[i // 4, i % 4].set_xlabel("")
axes[1, 3].axis("off")
save("fig05_histograms_levels")


# %%
# distribution of the daily gold return vs a normal distribution
g = ret["Gold"]
RES["gold_ret_stats"] = {"mean": float(g.mean()), "std": float(g.std()), "skew": float(g.skew()),
                         "excess_kurtosis": float(g.kurt()), "min": float(g.min()), "max": float(g.max())}
sh = stats.shapiro(g)                      # all ~4,200 daily returns (shapiro is reliable up to 5,000 values)
RES["gold_ret_shapiro_p"] = float(sh.pvalue)
print(RES["gold_ret_stats"], sh)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(g, bins=80, stat="density", color="goldenrod", ax=axes[0])
xs = np.linspace(g.min(), g.max(), 300)
axes[0].plot(xs, norm.pdf(xs, g.mean(), g.std()), color="red", label="Normal with same mean/std")
axes[0].set_title("Daily gold log return")
axes[0].legend()
stats.probplot(g, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q plot of daily gold return")
save("fig06_gold_return_distribution")

# %%
# the drivers over time
drivers = ["USD_Index", "Yield_10Y", "VIX", "SP500", "Oil"]
fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
for ax, c in zip(axes, drivers):
    ax.plot(data.index, data[c], lw=0.9)
    ax.set_ylabel(c)
axes[0].set_title("The five candidate drivers over time")
save("fig07_drivers_time")

# %%
# annual summary table
yr = df["Gold"].groupby(df.index.year)
annual = pd.DataFrame({"Avg_Gold": yr.mean(), "Min_Gold": yr.min(), "Max_Gold": yr.max(), "Year_End": yr.last()})
annual["Change_pct"] = annual["Year_End"].pct_change() * 100
annual.iloc[0, annual.columns.get_loc("Change_pct")] = (annual["Year_End"].iloc[0] / df["Gold"].iloc[0] - 1) * 100
annual["Volatility_pct"] = ret["Gold"].groupby(ret.index.year).std() * np.sqrt(252) * 100
annual = annual.round(1)
annual.to_csv("table_annual_summary.csv")
RES["annual"] = {str(k): v for k, v in annual.to_dict(orient="index").items()}
print(annual)


# %% [markdown]
# ## **2. Bivariate analysis**

# %%
# gold vs each driver (colored by period)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), ["SP500", "USD_Index", "Yield_10Y", "VIX", "Oil"]):
    sns.scatterplot(df, x=c, y="Gold", hue="Period", s=8, alpha=0.6, ax=ax, palette="viridis", legend=(c == "SP500"))
    sns.regplot(df, x=c, y="Gold", scatter=False, color="red", ax=ax, line_kws={"lw": 1.5})
    ax.set_title(f"Gold vs {c}  (r = {df['Gold'].corr(df[c]):.2f})")
axes[1, 2].axis("off")
save("fig08_scatter_gold_vs_drivers")

# %%
# gold vs silver and the gold/silver ratio
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.scatterplot(df, x="Silver", y="Gold", hue="Period", s=8, alpha=0.6, ax=axes[0], palette="viridis")
sns.regplot(df, x="Silver", y="Gold", scatter=False, color="red", ax=axes[0])
axes[0].set_title(f"Gold vs Silver (r = {df['Gold'].corr(df['Silver']):.2f})")
axes[1].plot(df.index, df["Gold_Silver_Ratio"], color="gray", lw=1)
axes[1].axhline(df["Gold_Silver_Ratio"].mean(), color="red", ls="--", lw=1, label=f"Mean = {df['Gold_Silver_Ratio'].mean():.1f}")
axes[1].set_title("Gold / Silver ratio")
axes[1].legend()
save("fig09_gold_silver_ratio")
RES["gs_ratio"] = {"mean": float(df["Gold_Silver_Ratio"].mean()), "min": float(df["Gold_Silver_Ratio"].min()),
                   "max": float(df["Gold_Silver_Ratio"].max()), "max_date": str(df["Gold_Silver_Ratio"].idxmax().date()),
                   "min_date": str(df["Gold_Silver_Ratio"].idxmin().date()),
                   "last": float(df["Gold_Silver_Ratio"].iloc[-1])}
RES["corr_gold_silver_level"] = float(df["Gold"].corr(df["Silver"]))

# %%
# grouped boxplots: gold price by year, gold return by VIX regime
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
sns.boxplot(df, x="Year", y="Gold", color="goldenrod", ax=axes[0])
axes[0].tick_params(axis="x", rotation=60)
axes[0].set_title("Gold price by year")
rd = df.dropna(subset=["Gold_Return"])
sns.boxplot(rd, x="High_VIX", y="Gold_Return", palette=["lightgreen", "salmon"], hue="High_VIX", legend=False, ax=axes[1])
axes[1].set_title("Daily gold return by market-stress regime")
axes[1].set_xlabel("")
save("fig10_grouped_boxplots")

# %%
# hypothesis test (like Lab 7): is the mean daily gold return different on stress days vs calm days?
hi = rd.loc[rd["VIX"] > 20, "Gold_Return"]
lo = rd.loc[rd["VIX"] <= 20, "Gold_Return"]
tt = stats.ttest_ind(hi, lo, equal_var=False)
RES["ttest_vix"] = {"n_high": int(len(hi)), "n_low": int(len(lo)), "mean_high": float(hi.mean()), "mean_low": float(lo.mean()),
                    "std_high": float(hi.std()), "std_low": float(lo.std()), "t": float(tt.statistic), "p": float(tt.pvalue)}
lv = stats.levene(hi, lo)
RES["levene_vix"] = {"stat": float(lv.statistic), "p": float(lv.pvalue)}
print(RES["ttest_vix"], RES["levene_vix"])


# %% [markdown]
# ## **3. Correlation analysis**

# %%
pear = data.corr(method="pearson")
spear = data.corr(method="spearman")
RES["pearson_levels"] = pear.round(3).to_dict(orient="index")
RES["spearman_levels"] = spear.round(3).to_dict(orient="index")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title("Pearson correlation (price levels)")
sns.heatmap(spear, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title("Spearman correlation (price levels)")
save("fig11_corr_levels")

# %%
cr = ret.corr()
RES["pearson_returns"] = cr.round(3).to_dict(orient="index")
plt.figure(figsize=(7.5, 6))
sns.heatmap(cr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson correlation of daily changes")
save("fig12_corr_returns")

# significance of the gold-return correlations
pv = {}
for c in ["Silver", "USD_Index", "Yield_10Y", "VIX", "SP500", "Oil"]:
    r_, p_ = stats.pearsonr(ret["Gold"], ret[c])
    pv[c] = {"r": float(r_), "p": float(p_)}
RES["gold_ret_corr_tests"] = pv
print(pv)

# %%
# is the relationship stable? 1-year (252 trading days) rolling correlation of daily changes with gold
roll = pd.DataFrame({c: ret["Gold"].rolling(252).corr(ret[c]) for c in ["USD_Index", "SP500", "Yield_10Y"]})
RES["rolling_corr"] = {c: {"min": float(roll[c].min()), "max": float(roll[c].max()), "mean": float(roll[c].mean())} for c in roll.columns}
roll.plot(figsize=(12, 4))
plt.axhline(0, color="black", lw=0.8)
plt.title("Rolling 1-year correlation between daily gold return and the drivers")
plt.ylabel("Correlation")
save("fig13_rolling_corr")


# %% [markdown]
# # **IV. Model Development**
# ## **1. Data splitting**

# %%
# --- A. PRICE-LEVEL MODEL: predict the gold price from the 5 drivers ---
X_lv = df[level_features]
y_lv = df["Gold"]
X_train, X_test, y_train, y_test = train_test_split(X_lv, y_lv, test_size=0.2, shuffle=False)   # chronological split
RES["split_level"] = {"train": int(len(X_train)), "test": int(len(X_test)),
                      "train_start": str(X_train.index.min().date()), "train_end": str(X_train.index.max().date()),
                      "test_start": str(X_test.index.min().date()), "test_end": str(X_test.index.max().date()),
                      "train_gold_min": float(y_train.min()), "train_gold_max": float(y_train.max()),
                      "test_gold_min": float(y_test.min()), "test_gold_max": float(y_test.max()),
                      "test_share_above_train_max": float((y_test > y_train.max()).mean())}
print(RES["split_level"])

plt.figure(figsize=(12, 4))
plt.plot(y_train.index, y_train, color="steelblue", label=f"Train ({len(y_train)} days)")
plt.plot(y_test.index, y_test, color="darkorange", label=f"Test ({len(y_test)} days)")
plt.axhline(y_train.max(), color="red", ls="--", lw=1, label=f"Highest training price = {y_train.max():.0f}")
plt.ylabel("Gold price (USD/oz)")
plt.title("Chronological 80% / 20% split of the gold price")
plt.legend()
save("fig14_train_test_split")

# %%
# --- B. RETURN MODEL: explain the daily gold return from the daily change of the drivers ---
X_rt = ret[ret_features]
y_rt = ret["Gold"]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_rt, y_rt, test_size=0.2, shuffle=False)
RES["split_ret"] = {"train": int(len(Xr_train)), "test": int(len(Xr_test)),
                    "train_start": str(Xr_train.index.min().date()), "train_end": str(Xr_train.index.max().date()),
                    "test_start": str(Xr_test.index.min().date()), "test_end": str(Xr_test.index.max().date())}
print(RES["split_ret"])

# %% [markdown]
# ## **2. Model selection and training**

# %%
def make_models():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def scores(y_true, y_pred, mape=True):
    s = {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
         "MAE": float(mean_absolute_error(y_true, y_pred)),
         "R2": float(r2_score(y_true, y_pred))}
    if mape:
        s["MAPE"] = float(mean_absolute_percentage_error(y_true, y_pred))
    return s


# ---- Price-level models
models_lv = make_models()
pred_lv, perf_lv = {}, {}
for name, m in models_lv.items():
    m.fit(X_train, y_train)
    pred_lv[name] = pd.Series(m.predict(X_test), index=X_test.index)
    perf_lv[name] = {"train": scores(y_train, m.predict(X_train)), "test": scores(y_test, pred_lv[name])}
RES["perf_level"] = perf_lv
perf_lv_tab = pd.DataFrame({k: v["test"] for k, v in perf_lv.items()}).T
print(perf_lv_tab.round(3))
RES["max_pred_level"] = {k: float(v.max()) for k, v in pred_lv.items()}

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, (name, p) in zip(axes, pred_lv.items()):
    ax.plot(y_test.index, y_test, color="black", lw=1.2, label="Actual")
    ax.plot(p.index, p, color="tab:red", lw=1.1, label=f"Predicted ({name})")
    ax.axhline(y_train.max(), color="gray", ls="--", lw=0.8, label="Highest training price")
    ax.set_ylabel("Gold (USD/oz)")
    ax.set_title(f"{name}: R2 = {perf_lv[name]['test']['R2']:.2f}, RMSE = {perf_lv[name]['test']['RMSE']:.0f}")
    ax.legend(loc="upper left")
save("fig15_actual_vs_pred_level")

# %%
# ---- Return models
models_rt = make_models()
pred_rt, perf_rt = {}, {}
for name, m in models_rt.items():
    m.fit(Xr_train, yr_train)
    pred_rt[name] = pd.Series(m.predict(Xr_test), index=Xr_test.index)
    perf_rt[name] = {"train": scores(yr_train, m.predict(Xr_train), mape=False), "test": scores(yr_test, pred_rt[name], mape=False)}
    perf_rt[name]["test"]["Direction_acc"] = float(np.mean(np.sign(pred_rt[name]) == np.sign(yr_test)))
    cv = cross_val_score(make_models()[name], Xr_train, yr_train, cv=TimeSeriesSplit(5), scoring="r2")
    perf_rt[name]["cv_r2_mean"] = float(cv.mean())
    perf_rt[name]["cv_r2_std"] = float(cv.std())
RES["perf_return"] = perf_rt
perf_rt_tab = pd.DataFrame({k: v["test"] for k, v in perf_rt.items()}).T
print(perf_rt_tab.round(4))
print({k: (round(v["cv_r2_mean"], 3), round(v["cv_r2_std"], 3)) for k, v in perf_rt.items()})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
lim = [float(yr_test.min()), float(yr_test.max())]
for ax, (name, p) in zip(axes, pred_rt.items()):
    ax.scatter(yr_test, p, s=8, alpha=0.5)
    ax.plot(lim, lim, color="red", ls="--", lw=1, label="Perfect prediction")
    ax.set_xlabel("Actual daily gold return")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name}: R2 = {perf_rt[name]['test']['R2']:.3f}")
    ax.legend()
save("fig16_actual_vs_pred_return")


# %% [markdown]
# # **V. Results and Evaluation**
# ## **1. Performance metrics**

# %%
def metric_bars(perf, metrics, title, fname):
    names = list(perf.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    for ax, mt in zip(np.atleast_1d(axes), metrics):
        vals = [perf[n]["test"][mt] for n in names]
        sns.barplot(x=names, y=vals, hue=names, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"{mt} (test)")
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        if mt == "R2":
            ax.axhline(0, color="black", lw=0.8)
    plt.suptitle(title, y=1.03)
    save(fname)


metric_bars(perf_lv, ["RMSE", "MAE", "R2"], "Price-level models (test set)", "fig17_metrics_level")
metric_bars(perf_rt, ["RMSE", "MAE", "R2"], "Daily-return models (test set)", "fig18_metrics_return")

# %%
# why a chronological split? the same models with a RANDOM 80/20 split (neighbouring days leak into the training set)
Xa, Xb, ya, yb = train_test_split(X_lv, y_lv, test_size=0.2, shuffle=True, random_state=42)
rand = {}
for name, m in make_models().items():
    m.fit(Xa, ya)
    rand[name] = {"R2": float(r2_score(yb, m.predict(Xb))), "RMSE": float(np.sqrt(mean_squared_error(yb, m.predict(Xb))))}
RES["random_split_level"] = rand
print(rand)

cmp = pd.DataFrame({"Random split": {k: v["R2"] for k, v in rand.items()},
                    "Chronological split": {k: perf_lv[k]["test"]["R2"] for k in rand}})
cmp.plot(kind="bar", figsize=(8, 4.2), color=["tab:green", "tab:red"])
plt.axhline(0, color="black", lw=0.8)
plt.ylabel("Test R2")
plt.title("Same models, two ways of splitting the data")
plt.xticks(rotation=0)
save("fig19_random_vs_chrono")

# %% [markdown]
# ## **2. Feature importance**

# %%
# --- price-level models: standardized coefficients (linear) + impurity importance (trees)
coef_lv = pd.Series(models_lv["Linear Regression"][-1].coef_, index=level_features)
imp_rf_lv = pd.Series(models_lv["Random Forest"].feature_importances_, index=level_features)
imp_gb_lv = pd.Series(models_lv["Gradient Boosting"].feature_importances_, index=level_features)
RES["importance_level"] = {"coef_std": coef_lv.round(3).to_dict(), "rf": imp_rf_lv.round(3).to_dict(), "gb": imp_gb_lv.round(3).to_dict()}
print(coef_lv.round(1), imp_rf_lv.round(3), imp_gb_lv.round(3), sep="\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
coef_lv.sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Linear Regression: standardized coefficients")
imp_rf_lv.sort_values().plot(kind="barh", ax=axes[1], color="seagreen")
axes[1].set_title("Random Forest: importance")
imp_gb_lv.sort_values().plot(kind="barh", ax=axes[2], color="darkorange")
axes[2].set_title("Gradient Boosting: importance")
save("fig20_importance_level")

# %%
# --- return models: permutation importance on the TEST set (drop in R2 when a column is shuffled)
perm = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, m) in zip(axes, models_rt.items()):
    pi = permutation_importance(m, Xr_test, yr_test, n_repeats=30, random_state=42, scoring="r2")
    s = pd.Series(pi.importances_mean, index=ret_features)
    e = pd.Series(pi.importances_std, index=ret_features)
    perm[name] = {"mean": s.round(4).to_dict(), "std": e.round(4).to_dict()}
    order = s.sort_values().index
    ax.barh(order, s[order], xerr=e[order], color="slateblue")
    ax.set_title(name)
    ax.set_xlabel("Drop in test R2 when shuffled")
RES["permutation_return"] = perm
print(perm)
save("fig21_importance_return")

# standardized coefficients of the linear return model (direction of effect)
coef_rt = pd.Series(models_rt["Linear Regression"][-1].coef_, index=ret_features)
RES["coef_return_std"] = coef_rt.round(5).to_dict()
print(coef_rt)

# %%
# --- statsmodels view of the price-level linear model (like Lab 8): VIF, p-values, autocorrelation
Xs = sm.add_constant(X_train)
ols = sm.OLS(y_train, Xs).fit()
vif = pd.Series([variance_inflation_factor(Xs.values, i) for i in range(Xs.shape[1])], index=Xs.columns).drop("const")
RES["vif_level"] = vif.round(2).to_dict()
RES["ols_level"] = {"r2_train": float(ols.rsquared), "adj_r2_train": float(ols.rsquared_adj),
                    "coef": ols.params.round(3).to_dict(), "pvalues": {k: float(v) for k, v in ols.pvalues.items()},
                    "durbin_watson": float(durbin_watson(ols.resid))}
print(vif.round(2))
print(ols.summary().tables[1])
print("Durbin-Watson:", RES["ols_level"]["durbin_watson"])

# %%
# save the cleaned data + engineered columns, the tables, and every number used in the report
df.drop(columns=["Period"]).assign(Period=df["Period"].astype(str)).to_csv("gold_clean.csv")
perf_lv_tab.round(4).to_csv("table_perf_level.csv")
perf_rt_tab.round(5).to_csv("table_perf_return.csv")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


with open("results.json", "w") as f:
    json.dump(RES, f, indent=1, default=_default)
print("Saved results.json, gold_clean.csv and", len(os.listdir(FIG)), "figures")

