from q1 import *


def q6(c):
    SR, M, LG, T2, ST = S["simple_reg_all"], S["mlr"], S["logit"], S["ttest_usd"], S["simple_reg_train"]
    c.heading("VI")
    c.para("In short, a few market numbers explain only a small part of what gold does on a normal day, but the results are clear about which ones matter.")
    c.bullet("Key Findings:", "")
    c.sub(f"**1. The dollar is the main driver we found.** On the daily changes, the USD index has the strongest negative link with gold (r = -0.35). "
          f"When the dollar goes up gold loses about {abs(T2['mean_a'])*100:.2f}% on average, and when it goes down gold gains about {T2['mean_b']*100:.2f}% (t-test, p < 0.001). "
          f"Gold goes up on {pct(S['chi2']['usd']['prop_up_if_group_false'], 0)} of the days when the dollar falls, against {pct(S['chi2']['usd']['prop_up_if_group_true'], 0)} when the dollar rises (chi-square test). "
          "The dollar and the 10-year yield are the only significant drivers in both regression models, apart from a small effect of oil.")
    c.sub("**2. Stock market fear does not matter on average.** Stress days (VIX above 20) do not have a different average gold return from calm days (t-test p = 0.30, chi-square p = 0.14). "
          "Gold is not a sure hedge on a single day.")
    c.sub("**3. Correlation on price levels can lie.** Gold and the S&P 500 have r = 0.86 on price levels, but only r = 0.07 on daily changes. Both just trended up for 16 years.")
    c.sub(f"**4. The models learn a little, not a lot.** Multiple regression has a test R2 of {M['r2_test']:.2f}, and the simple regression on the dollar alone does slightly better ({ST['r2_test']:.2f}). Logistic regression has {pct(LG['accuracy'], 0)} accuracy against a {pct(LG['baseline_acc'], 0)} baseline (AUC {LG['auc']:.2f}). "
          "So the drivers explain less than 10% of the size of a gold move and give a small edge on the direction.")
    c.bullet("Implications:", "")
    c.sub("**For investors:** watching the dollar and the US yield gives a better view of gold's day-to-day moves than watching the stock market or the VIX. But these numbers are not enough to time gold, "
          "and big daily drops do happen (the worst was -11.4% in log return, about -10.8% in price, in one day).")
    c.sub("**For students and analysts:** use the daily changes and not the price levels when two series trend, and always compare a model with a simple baseline before saying it is good.")
    c.bullet("Limitations & Future Work:", "")
    c.sub("**Data:** only 5 drivers and daily closing prices. We did not include inflation, real interest rates, central bank gold buying or ETF flows, which many people think matter, because we could not download them easily. "
          "Gold and silver are futures prices, not spot prices.")
    c.sub("**Method:** the models use the other markets' changes on the same day, so they explain a day but do not forecast the next one. We used only linear models and one test period (2023-2026), which was more volatile than the training years. "
          "Daily returns are not perfectly normal (they have heavy tails), so p-values should be read with some care.")
    c.sub("**Future work:** add inflation and real yield, use yesterday's values to try a real forecast, and look at the time series with more advanced methods.")


def q7(c):
    c.heading("VII")
    refs = [
        "Yahoo Finance. (2026). Historical daily closing prices for GC=F (gold futures), SI=F (silver futures), CL=F (WTI crude oil futures), DX-Y.NYB (US Dollar Index), ^TNX (10-year Treasury yield), ^VIX (CBOE Volatility Index) and ^GSPC (S&P 500). https://finance.yahoo.com",
        "McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, 56-61. (pandas)",
        "Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585, 357-362.",
        "Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261-272.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. https://scikit-learn.org",
        "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. Proceedings of the 9th Python in Science Conference.",
        "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.",
        "Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.",
        "HAS, S. (n.d.). Data Analysis I: lecture notes and Labs 1 to 10. Course material.",
    ]
    for i, r in enumerate(refs, 1):
        c.para(f"[{i}] {r}", align="left")


def q8(c):
    c.heading("VIII")
    c.bullet("Source Code Link:", "https://github.com/Chuosrun/Gold-Price")
    c.sub("Files in the project folder: `01_download_data.py` (downloads the raw data), `02_gold_analysis.py` (cleaning and EDA), `03_gold_simple.py` (tests and regression models used in this report), "
          "`gold_raw.csv` (raw data), `gold_clean.csv` (clean data with the new variables), `results.json` and `results_simple.json` (every number used in this report). "
          "The libraries are the ones used in the labs: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels` and `sklearn`.")
    c.bullet("Supplementary figures:", "")
    c.figure("fig07_drivers_time", "The five drivers over time (USD index, 10-year yield, VIX, S&P 500, Oil).", appendix=True, max_w=5.4, max_h=6.0)
