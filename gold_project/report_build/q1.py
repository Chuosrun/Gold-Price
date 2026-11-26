import json
from rep_lib import *

S = json.load(open(PROJ + r"\results_simple.json"))


def pf(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def q1(c):
    R = c.R
    c.heading("I")
    c.para(
        "Gold is one of the most watched assets in the world. People buy it when they are scared about the economy, "
        "central banks keep it in their reserves, and jewelers and factories use it too. The price goes up and down a lot, "
        "so the question for this project is: **which market factors move the gold price?** We clean the data, explore it with "
        "charts and statistics, test some ideas with hypothesis tests, and then build regression models to see how much of gold "
        "we can explain."
    )
    c.para(
        f"The dataset was built for this project from Yahoo Finance daily closing prices. It runs from **{R['date_min']} to {R['date_max']}** "
        f"and has **{R['clean_shape'][0]:,} trading days and 7 variables**: Gold (the target, futures price in USD per ounce), Silver, "
        "Oil (WTI crude, USD per barrel), the US Dollar Index (`USD_Index`), the 10-year US Treasury yield (`Yield_10Y`, in %), "
        "the VIX index (the market \"fear\" index) and the S&P 500 stock index. "
        f"Gold went from {n0(R['gold_first'])} USD to a record of {n0(R['gold_max'][1])} USD on {R['gold_max'][0]}, "
        f"and it was at {n0(R['gold_last'][1])} USD on the last day."
    )
    c.para(
        "We use two kinds of targets. First the gold price itself, which we only use for exploring, and then the **daily change** "
        "(return) of gold, which we use for the models. We explain the reason in Section III. As the template says, most of the work "
        "is in preprocessing and EDA, and the models are the smaller part."
    )


def q2b(c):
    R = c.R
    ol, orr, st = R["outliers_levels"], R["outliers_returns"], R["skew_table"]
    c.bullet("Identify & Handle Outliers:", "We used boxplots and the IQR rule (a value is an outlier if it is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR).")
    c.sub(f"**Price levels (Figure {c.fno('fig02_boxplots_levels')}):** Gold has {ol['Gold']['n_out']:.0f} outliers ({ol['Gold']['pct_out']:.1f}%), "
          f"Silver {ol['Silver']['n_out']:.0f} ({ol['Silver']['pct_out']:.1f}%) and VIX {ol['VIX']['n_out']:.0f} ({ol['VIX']['pct_out']:.1f}%), all on the high side. "
          "The other four variables have none. The gold outliers (above 2,827 USD) are all from **2025 and 2026**, so they come from the recent rally and are not mistakes.")
    c.figure("fig02_boxplots_levels", "Boxplots of the 7 price series.", max_w=6.2)
    c.sub(f"**Daily changes (Figure {c.fno('fig03_boxplots_returns')}):** every series has between {orr['Yield_10Y']['pct_out']:.1f}% and {orr['Silver']['pct_out']:.1f}% outliers. "
          "The worst gold days are 2026-01-30 (-11.4%, gold fell from 5,318 to 4,745 USD while silver fell from 114 to 78 USD the same day), "
          "2013-04-15 (-9.8%) and 2013-06-20 (-6.6%). The best days are 2026-02-03 (+5.9%) and 2020-03-23 and 24. These are real market events, not typing mistakes.")
    c.figure("fig03_boxplots_returns", "Boxplots of the daily changes.", max_w=6.2)
    c.sub("**Decision:** we **kept** the outliers. If we removed them we would delete exactly the crises and the rally that we want to study, "
          "and a price that was cut is not a real price anymore.")
    c.bullet("Feature Engineering:", "We made these new variables from the clean data.")
    c.sub(f"**Daily change (return):** for Gold, Silver, Oil, USD Index, VIX and S&P 500 we used `log(price today) - log(price yesterday)`, which is about the percentage change. "
          f"For the yield we used the simple difference because it is already a percentage. This gives {R['returns_shape'][0]:,} rows. "
          "We need this because gold and many drivers have a strong trend over the years, and two things that both trend up look related even when they are not (we show this in Section III).")
    c.sub(f"**Other variables:** `Year`, `Gold_Silver_Ratio` (Gold divided by Silver, EDA only), moving averages of 50 and 200 days, and two groups made from the data: "
          "stress days (VIX above 20) vs calm days, and days when the dollar went up vs down. "
          f"We also checked the skewness: Gold {st['Gold']['skew_original']:.2f}, Silver {st['Silver']['skew_original']:.2f} and VIX {st['VIX']['skew_original']:.2f} are right skewed. "
          "We did not transform them, because the models use the daily changes, which are not strongly skewed.")
