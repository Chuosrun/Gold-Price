from q1 import *


def q3c(c):
    R = c.R
    f = c.fno
    pl, pr = R["pearson_levels"], R["pearson_returns"]
    CR, L = S["corr_returns"], S["level_reg"]
    c.bullet("Correlation Analysis:", "")
    c.sub(f"**Price levels (Figure {f('fig11_corr_levels')}):** Pearson and Spearman correlations of the price levels are shown as heatmaps. Gold is strongly linked with Silver ({pl['Gold']['Silver']:.2f}) and the S&P 500 ({pl['Gold']['SP500']:.2f}). "
          "The two methods agree on the main ideas, but not on every pair. For gold and the yield, Pearson is 0.58 and Spearman is only 0.33, so that link is not a clean straight line.")
    c.figure("fig11_corr_levels", "Pearson and Spearman correlation of the price levels.", max_w=6.3)
    c.sub(f"**A warning about trends (Figure {f('s06_level_regression')}):** a regression of gold on the S&P 500 levels gives R2 = {L['r2']:.2f}, which looks strong. "
          "But both series went up over the 16 years, so any two trending series will look related. This is correlation, not cause. To remove the trend we look at the **daily changes**.")
    c.figure("s06_level_regression", "Regression on the price levels. The colors are the years (dark = 2010, yellow = 2026).", max_w=4.8)
    c.sub(f"**Daily changes (Figure {f('fig12_corr_returns')}):** the picture changes. Gold and the S&P 500 have only r = {pr['Gold']['SP500']:.2f}. "
          f"The strongest link is with **Silver ({pr['Gold']['Silver']:.2f})**, then the **USD index ({pr['Gold']['USD_Index']:.2f})** and the **10-year yield ({pr['Gold']['Yield_10Y']:.2f})**. "
          f"Oil is small ({pr['Gold']['Oil']:.2f}) and VIX is almost 0 ({pr['Gold']['VIX']:.2f}). The S&P 500 and the VIX have r = {pr['SP500']['VIX']:.2f}: when stocks fall, the VIX jumps.")
    c.figure("fig12_corr_returns", "Pearson correlation of the daily changes.", max_w=5.0)
    rows = [[k, f"{v['pearson']:.3f}", pf(v["p_pearson"]), f"{v['spearman']:.3f}", pf(v["p_spearman"])] for k, v in CR.items()]
    c.table("Correlation of the daily gold return with the daily change of each variable (n = 4,257).",
            ["Variable", "Pearson r", "p-value", "Spearman rho", "p-value"], rows, widths=[1.5, 1.1, 1.1, 1.3, 1.1])
    c.sub("Pearson and Spearman are close for every variable, so the results are not caused by a few extreme days. With n = 4,257 even small correlations can be significant, "
          "so we look at the **size** of r and not only the p-value. Only the dollar, the yield and silver have a correlation big enough to matter. "
          "Silver is not used as a predictor later, because it is another precious metal and mostly copies gold.")
