# %% [markdown]
# # **Final Project - What Drives the Gold Price? (course-level version)**
# Only methods from the course: descriptive statistics, z-test / t-test, chi-square test, Pearson / Spearman,
# simple, multiple and logistic regression, moving averages.
# Input: gold_clean.csv (made by 02_gold_analysis.py). Output: results_simple.json + figures s01...s06.

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

sns.set_theme(style="whitegrid")
FIG = "figures"
RES = {}


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close("all")


df = pd.read_csv("gold_clean.csv", parse_dates=["Date"], index_col="Date")
cols = ["Gold", "Silver", "Oil", "USD_Index", "Yield_10Y", "VIX", "SP500"]
data = df[cols]

# daily change: log return for prices, simple difference for the yield
ret = pd.DataFrame(index=data.index)
for c in ["Gold", "Silver", "Oil", "USD_Index", "SP500", "VIX"]:
    ret[c] = np.log(data[c]).diff()
ret["Yield_10Y"] = data["Yield_10Y"].diff()
ret = ret.dropna()
g = ret["Gold"]
n = len(g)

# %% [markdown]
# ## Descriptive statistics, confidence interval, z-test and normal probability

# %%
mean, sd = g.mean(), g.std()
se = sd / np.sqrt(n)
ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
z_stat = mean / se
p_z = 2 * (1 - norm.cdf(abs(z_stat)))
zs = (g - mean) / sd
worst_z = float(zs.min())
RES["desc"] = {"n": n, "mean": mean, "median": float(g.median()), "std": sd, "min": float(g.min()), "max": float(g.max()),
               "skew": float(g.skew()), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
               "ann_mean_pct": float(mean * 252 * 100), "ann_vol_pct": float(sd * np.sqrt(252) * 100)}
RES["ztest_mean0"] = {"z": float(z_stat), "p": float(p_z)}
RES["normal_check"] = {"worst_day": str(g.idxmin().date()), "worst_ret": float(g.min()), "worst_z": worst_z,
                       "p_normal_worst": float(norm.cdf(worst_z)),
                       "n_beyond3": int((zs.abs() > 3).sum()), "expected_beyond3": float(0.0027 * n),
                       "share_within1": float((zs.abs() <= 1).mean()), "share_within2": float((zs.abs() <= 2).mean())}
print(RES["desc"], RES["ztest_mean0"], RES["normal_check"])

# %% [markdown]
# ## Time series basics: trend and moving averages

# %%
gold = data["Gold"]
ma50, ma200 = gold.rolling(50).mean(), gold.rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(gold.index, gold, color="goldenrod", lw=1, label="Gold (daily close)")
ax.plot(ma50.index, ma50, color="steelblue", lw=1.3, label="50-day moving average")
ax.plot(ma200.index, ma200, color="red", lw=1.5, label="200-day moving average")
ax.set_ylabel("Gold price (USD per ounce)")
ax.set_title("Gold price with 50-day and 200-day moving averages")
ax.legend(loc="upper left")
save("s01_moving_average")
ok = ma200.notna()
RES["moving_avg"] = {"share_above_ma200": float((gold[ok] > ma200[ok]).mean()),
                     "last_gold": float(gold.iloc[-1]), "last_ma50": float(ma50.iloc[-1]), "last_ma200": float(ma200.iloc[-1])}
print(RES["moving_avg"])

# %% [markdown]
# ## Hypothesis tests (t-test, chi-square) and correlation (Pearson / Spearman)

# %%
# t-test 1: stress days (VIX > 20) vs calm days
stress = (data["VIX"].loc[ret.index] > 20)
a, b = g[stress], g[~stress]
t1 = stats.ttest_ind(a, b, equal_var=False)
RES["ttest_vix"] = {"n_a": int(len(a)), "n_b": int(len(b)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "std_a": float(a.std()), "std_b": float(b.std()), "t": float(t1.statistic), "p": float(t1.pvalue)}

# t-test 2: days when the dollar went up vs days when it went down
usd_up = ret["USD_Index"] > 0
a, b = g[usd_up], g[~usd_up]
t2 = stats.ttest_ind(a, b, equal_var=False)
RES["ttest_usd"] = {"n_a": int(len(a)), "n_b": int(len(b)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "t": float(t2.statistic), "p": float(t2.pvalue)}

# chi-square test of independence: gold up/down vs dollar up/down, and vs stress/calm
gold_up = g > 0
chi = {}
for name, grp in [("usd", usd_up), ("vix", stress)]:
    tab = pd.crosstab(grp, gold_up)
    c2, p, dof, exp = stats.chi2_contingency(tab)
    chi[name] = {"table": tab.values.tolist(), "chi2": float(c2), "p": float(p), "dof": int(dof),
                 "min_expected": float(exp.min()),
                 "prop_up_if_group_true": float(gold_up[grp].mean()), "prop_up_if_group_false": float(gold_up[~grp].mean())}
RES["chi2"] = chi
print(RES["ttest_vix"], RES["ttest_usd"], chi)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (lab, grp, names) in zip(axes, [("USD index", usd_up, ["Dollar down", "Dollar up"]),
                                         ("VIX", stress, ["Calm (VIX <= 20)", "Stress (VIX > 20)"])]):
    vals = [gold_up[~grp].mean(), gold_up[grp].mean()]
    sns.barplot(x=names, y=vals, hue=names, palette=["lightsteelblue", "salmon"], legend=False, ax=ax)
    ax.axhline(gold_up.mean(), color="black", ls="--", lw=1, label=f"All days ({gold_up.mean():.2f})")
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Share of days gold went up")
    ax.set_title(f"Gold up-days by {lab}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.legend()
save("s02_prop_up")

# %%
# Pearson and Spearman between the daily gold return and each driver
rows = {}
for c in ["Silver", "USD_Index", "Yield_10Y", "VIX", "SP500", "Oil"]:
    rp, pp = stats.pearsonr(g, ret[c])
    rs, ps = stats.spearmanr(g, ret[c])
    rows[c] = {"pearson": float(rp), "p_pearson": float(pp), "spearman": float(rs), "p_spearman": float(ps)}
RES["corr_returns"] = rows
print(rows)

# simple linear regression on all days: gold return ~ dollar return
lr = stats.linregress(ret["USD_Index"], g)
RES["simple_reg_all"] = {"slope": float(lr.slope), "intercept": float(lr.intercept), "r": float(lr.rvalue),
                         "r2": float(lr.rvalue ** 2), "p": float(lr.pvalue), "se": float(lr.stderr)}
plt.figure(figsize=(7, 4.8))
sns.scatterplot(x=ret["USD_Index"], y=g, s=8, alpha=0.4, color="steelblue")
xs = np.linspace(ret["USD_Index"].min(), ret["USD_Index"].max(), 100)
plt.plot(xs, lr.intercept + lr.slope * xs, color="red", lw=2, label=f"y = {lr.intercept:.4f} + {lr.slope:.3f} x   (R2 = {lr.rvalue**2:.3f})")
plt.xlabel("Daily change of the USD index (log return)")
plt.ylabel("Daily gold return (log return)")
plt.title("Simple linear regression: gold return vs dollar return")
plt.legend()
save("s03_simple_regression")
print(RES["simple_reg_all"])

# %% [markdown]
# ## Multiple linear regression (daily gold return) and logistic regression (gold up or down)

# %%
feats = ["USD_Index", "Yield_10Y", "VIX", "SP500", "Oil"]
X, y = ret[feats], g
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)    # time order is kept
RES["split"] = {"train": len(X_tr), "test": len(X_te), "train_start": str(X_tr.index.min().date()), "train_end": str(X_tr.index.max().date()),
                "test_start": str(X_te.index.min().date()), "test_end": str(X_te.index.max().date())}

ols = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
pred = ols.predict(sm.add_constant(X_te))
vif = pd.Series([variance_inflation_factor(sm.add_constant(X_tr).values, i) for i in range(1, len(feats) + 1)], index=feats)
RES["mlr"] = {"coef": ols.params.to_dict(), "pvalues": ols.pvalues.to_dict(), "r2_train": float(ols.rsquared),
              "adj_r2_train": float(ols.rsquared_adj), "f_p": float(ols.f_pvalue),
              "rmse_test": float(np.sqrt(mean_squared_error(y_te, pred))), "mae_test": float(mean_absolute_error(y_te, pred)),
              "r2_test": float(r2_score(y_te, pred)), "vif": vif.round(2).to_dict(),
              "rmse_baseline": float(np.sqrt(mean_squared_error(y_te, np.full(len(y_te), y_tr.mean()))))}
# standardized coefficients (to compare the size of the effects)
Xs = (X_tr - X_tr.mean()) / X_tr.std()
RES["mlr"]["std_coef"] = sm.OLS(y_tr, sm.add_constant(Xs)).fit().params.drop("const").to_dict()
print(RES["mlr"])

plt.figure(figsize=(6.5, 5))
plt.scatter(y_te, pred, s=8, alpha=0.5)
lim = [float(y_te.min()), float(y_te.max())]
plt.plot(lim, lim, color="red", ls="--", label="Perfect prediction")
plt.xlabel("Actual daily gold return")
plt.ylabel("Predicted")
plt.title(f"Multiple regression on the test set (R2 = {RES['mlr']['r2_test']:.3f})")
plt.legend()
save("s04_mlr_actual_pred")

# %%
# logistic regression: will gold go up (1) or down (0) on a day, given what the other markets did that day?
yc = (g > 0).astype(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, yc, test_size=0.2, shuffle=False)
sc = StandardScaler().fit(Xc_tr)
Ztr = pd.DataFrame(sc.transform(Xc_tr), index=Xc_tr.index, columns=feats)
Zte = pd.DataFrame(sc.transform(Xc_te), index=Xc_te.index, columns=feats)
logit = sm.Logit(yc_tr, sm.add_constant(Ztr)).fit(disp=0)
prob = logit.predict(sm.add_constant(Zte))
cls = (prob >= 0.5).astype(int)
cm = confusion_matrix(yc_te, cls)
major = max(yc_te.mean(), 1 - yc_te.mean())
RES["logit"] = {"coef": logit.params.to_dict(), "pvalues": logit.pvalues.to_dict(),
                "odds_ratio": {k: float(np.exp(v)) for k, v in logit.params.items()},
                "accuracy": float(accuracy_score(yc_te, cls)), "precision": float(precision_score(yc_te, cls)),
                "recall": float(recall_score(yc_te, cls)), "f1": float(f1_score(yc_te, cls)),
                "auc": float(roc_auc_score(yc_te, prob)), "cm": cm.tolist(), "baseline_acc": float(major),
                "share_up_test": float(yc_te.mean()), "share_up_train": float(yc_tr.mean()),
                "acc_train": float(accuracy_score(yc_tr, (logit.predict(sm.add_constant(Ztr)) >= 0.5).astype(int)))}
print(RES["logit"])

plt.figure(figsize=(4.6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic regression: confusion matrix (test)")
save("s05_confusion")

# %%
# a warning example: regression on the PRICE LEVELS (both series trend up)
lv = stats.linregress(data["SP500"], data["Gold"])
RES["level_reg"] = {"r": float(lv.rvalue), "r2": float(lv.rvalue ** 2), "slope": float(lv.slope)}
plt.figure(figsize=(7, 4.6))
sns.scatterplot(x=data["SP500"], y=data["Gold"], hue=data.index.year, palette="viridis", s=8, legend=False)
xs = np.linspace(data["SP500"].min(), data["SP500"].max(), 100)
plt.plot(xs, lv.intercept + lv.slope * xs, color="red", lw=2, label=f"R2 = {lv.rvalue**2:.2f}")
plt.xlabel("S&P 500 (level)")
plt.ylabel("Gold (level)")
plt.title("Regression on price levels: looks strong, but both just trend up")
plt.legend()
save("s06_level_regression")

with open("results_simple.json", "w") as f:
    json.dump(RES, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("saved results_simple.json")
