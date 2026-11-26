from q1 import *


def q3a(c):
    R = c.R
    f = c.fno
    an, st = R["annual"], R["skew_table"]
    D, Z, N, MA = S["desc"], S["ztest_mean0"], S["normal_check"], S["moving_avg"]
    c.heading("III")
    c.para("This is the longest section. We use the clean data and look at one variable at a time, then at pairs of variables, then at correlations and tests.")
    c.bullet("Univariate Analysis (Distributions):", "")
    c.sub(f"**Gold over time (Figure {f('fig04_gold_silver_time')}):** gold had a first peak of {n0(R['gold_peak_2011'][1])} USD in {R['gold_peak_2011'][0][:7]}, then fell to a low of "
          f"{n0(R['gold_min'][1])} USD on {R['gold_min'][0]}. It stayed mostly between 1,050 and 1,900 USD until 2019, and then climbed almost straight up in 2024 ({an['2024']['Change_pct']:+.1f}%) "
          f"and 2025 ({an['2025']['Change_pct']:+.1f}%) to the record of {n0(R['gold_max'][1])} USD. After that it fell about 33%. Silver has the same shape with bigger swings.")
    c.figure("fig04_gold_silver_time", "Gold and silver daily closing prices with some key dates.", max_w=6.3)
    c.sub(f"**Trend (Figure {f('s01_moving_average')}):** the 200-day moving average smooths the daily noise and shows the long trend. Gold was above its 200-day average on {pct(MA['share_above_ma200'], 0)} of the days. "
          f"On the last day gold ({n0(MA['last_gold'])}) is below both the 50-day average ({n0(MA['last_ma50'])}) and the 200-day average ({n0(MA['last_ma200'])}), which fits the recent drop.")
    c.figure("s01_moving_average", "Gold price with 50-day and 200-day moving averages.", max_w=6.2)
    c.sub(f"**Distributions (Figure {f('fig05_histograms_levels')}):** Gold (skew {st['Gold']['skew_original']:.2f}), Silver ({st['Silver']['skew_original']:.2f}) and VIX ({st['VIX']['skew_original']:.2f}) are strongly right skewed: "
          "most days are low and a few days are very high. Oil is close to a bell shape. USD Index has **two humps** (around 80 and 98), one for before 2014 and one for after. "
          "Yield and S&P 500 are spread out because both changed a lot over 16 years.")
    c.figure("fig05_histograms_levels", "Histograms and density curves of the 7 variables.", max_w=6.3)
    c.sub(f"**Daily gold return:** the mean is {D['mean']*100:.3f}% per day, the median is {D['median']*100:.3f}% and the standard deviation is {D['std']*100:.2f}% (n = {D['n']:,}). "
          f"That is about {D['ann_mean_pct']:.1f}% per year on average with a yearly volatility of {D['ann_vol_pct']:.1f}%. The 95% confidence interval of the mean is from {D['ci_low']*100:.4f}% to {D['ci_high']*100:.4f}%.")
    c.sub(f"**z-test:** we tested H0: the true mean daily return is 0, against H1: it is not 0. With n = {D['n']:,} we use the z-test. z = {Z['z']:.2f}, p-value = {Z['p']:.3f}. "
          "Since p is bigger than 0.05 we **do not reject H0**. Even though gold went up a lot in the end, the average daily return is not clearly different from 0, because the daily noise is big compared with the average.")
    c.sub(f"**Is it normal?** We turned each daily return into a z-score. {pct(N['share_within1'])} of the days are within 1 standard deviation (a normal curve gives 68%) and {pct(N['share_within2'])} are within 2 (normal gives 95%), "
          f"so the middle looks normal. But the tails are heavy: {N['n_beyond3']} days are beyond 3 standard deviations, and a normal curve would give only about {N['expected_beyond3']:.0f}. "
          f"The worst day ({N['worst_day']}) has z = {N['worst_z']:.1f}, which a normal curve says should almost never happen. So we should not trust the normal curve for the extreme days.")
    c.table("Gold by year (USD per ounce). Change is from the previous year end (2026 is up to 2026-11-25). Volatility is the annualized volatility: the standard deviation of that year's daily log returns times the square root of 252 trading days.",
            ["Year", "Average", "Min", "Max", "Change", "Volatility"],
            [[y, n0(v["Avg_Gold"]), n0(v["Min_Gold"]), n0(v["Max_Gold"]), f"{v['Change_pct']:+.1f}%", f"{v['Volatility_pct']:.1f}%"] for y, v in an.items()],
            widths=[0.8, 1.1, 1.1, 1.1, 1.1, 1.1], size=9)
    c.sub(f"The worst year was 2013 ({an['2013']['Change_pct']:+.1f}%) and the best was 2025 ({an['2025']['Change_pct']:+.1f}%). Volatility was lowest in 2017 and 2018 (about 10%) and highest in 2026 ({an['2026']['Volatility_pct']:.1f}%).")
