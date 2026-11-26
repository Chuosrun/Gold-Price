from q1 import *


def q3b(c):
    R = c.R
    f = c.fno
    pl = R["pearson_levels"]
    gs = R["gs_ratio"]
    T1, T2, CH = S["ttest_vix"], S["ttest_usd"], S["chi2"]
    c.bullet("Bivariate Analysis (Relationships):", "")
    c.sub(f"**Gold against each driver (Figure {f('fig08_scatter_gold_vs_drivers')}):** on the price levels the S&P 500 has the strongest link with gold (r = {pl['Gold']['SP500']:.2f}), then the yield (r = {pl['Gold']['Yield_10Y']:.2f}) "
          f"and the USD index (r = {pl['Gold']['USD_Index']:.2f}). VIX (r = {pl['Gold']['VIX']:.2f}) and Oil (r = {pl['Gold']['Oil']:.2f}) show almost nothing. "
          "But the colors show that each period makes its own cloud, so the relationship is not one straight line. These numbers can fool us, and we check why in the correlation part below.")
    c.figure("fig08_scatter_gold_vs_drivers", "Gold against each driver, colored by period, with a regression line.", max_w=6.3)
    c.sub(f"**Gold and silver (Figure {f('fig09_gold_silver_ratio')}):** the two prices have r = {R['corr_gold_silver_level']:.2f}, but the gold/silver ratio is not constant. It averages {gs['mean']:.1f}, "
          f"with a low of {gs['min']:.1f} ({gs['min_date']}) and a high of {gs['max']:.1f} ({gs['max_date']}, the COVID crash, when silver fell harder than gold).")
    c.figure("fig09_gold_silver_ratio", "Gold vs silver (left) and the gold/silver ratio over time (right).", max_w=6.3)
    c.sub(f"**Grouped boxplots (Figure {f('fig10_grouped_boxplots')}):** the gold price by year shows the climb, and the box gets wider after 2020. "
          "The daily return in stress days and calm days has a similar center, but stress days have a wider box.")
    c.figure("fig10_grouped_boxplots", "Gold price by year (left) and daily gold return in calm vs stress days (right).", max_w=6.3)
    c.sub("**Hypothesis test 1 (t-test, stress vs calm):** H0: the mean daily gold return is the same on stress days (VIX > 20) and calm days. H1: it is different. "
          f"Stress days: mean = {T1['mean_a']*100:.3f}% (n = {T1['n_a']:,}). Calm days: mean = {T1['mean_b']*100:.3f}% (n = {T1['n_b']:,}). Welch two-sample t-test: t = {T1['t']:.2f}, p = {pf(T1['p'])}. "
          "p is bigger than 0.05, so we **do not reject H0**. Market fear does not change the average gold return.")
    c.sub("**Hypothesis test 2 (t-test, dollar up vs down):** H0: the mean daily gold return is the same on days when the USD index goes up and days when it goes down. H1: it is different. "
          f"Dollar up: mean = {T2['mean_a']*100:.3f}% (n = {T2['n_a']:,}). Dollar down: mean = {T2['mean_b']*100:.3f}% (n = {T2['n_b']:,}). t = {T2['t']:.2f}, p {pf(T2['p'])}. "
          "p is far below 0.05, so we **reject H0**. On days when the dollar rises gold loses about 0.29% on average, and when it falls gold gains about 0.34%.")
    c.sub(f"**Hypothesis test 3 (chi-square test of independence):** here we only look at direction. H0: gold going up is independent of the dollar going up. "
          f"Gold went up on {pct(CH['usd']['prop_up_if_group_false'])} of the days when the dollar fell, but only on {pct(CH['usd']['prop_up_if_group_true'])} of the days when the dollar rose "
          f"(Figure {f('s02_prop_up')}, left). Chi-square = {CH['usd']['chi2']:.1f}, df = {CH['usd']['dof']}, p {pf(CH['usd']['p'])}, so we **reject H0**: the two are not independent. "
          f"For stress vs calm days gold went up on {pct(CH['vix']['prop_up_if_group_true'])} vs {pct(CH['vix']['prop_up_if_group_false'])} of the days, chi-square = {CH['vix']['chi2']:.2f}, p = {pf(CH['vix']['p'])}, "
          f"so we **do not reject H0** here. All expected counts are above 5 (the smallest is {min(CH['usd']['min_expected'], CH['vix']['min_expected']):.0f}), so the test is valid. Both results agree with the t-tests.")
    c.figure("s02_prop_up", "Share of days gold went up, by dollar direction (left) and by VIX regime (right).", max_w=5.8)
