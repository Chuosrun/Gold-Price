import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rep_lib import *
from s2 import s2
from q1 import q1, q2b
from q3a import q3a
from q3b import q3b
from q3c import q3c
from q4 import q4
from q5 import q5
from q6 import q6, q7, q8

OUT = os.path.join(BASE, "Final_Project_Report_Gold_Price_Simple.docx")
TITLE = "What Drives the Gold Price? A Data Analysis of Market Factors, 2010-2026"

c = Ctx()
c.plan_figures(["fig01_missing_values", "fig02_boxplots_levels", "fig03_boxplots_returns", "fig04_gold_silver_time",
                "s01_moving_average", "fig05_histograms_levels", "fig08_scatter_gold_vs_drivers", "fig09_gold_silver_ratio",
                "fig10_grouped_boxplots", "s02_prop_up", "fig11_corr_levels", "s06_level_regression", "fig12_corr_returns",
                "s03_simple_regression", "s04_mlr_actual_pred", "s05_confusion"])


def replace_in_runs(par, old, new):
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    return False


P = c.doc.paragraphs
assert replace_in_runs(P[4], "[Project Title]", TITLE)
assert replace_in_runs(P[10], "[Names and Student IDs]", "Lim Chuosrun (2024588)")
for par in P[:14]:
    for r in par.runs:
        if "\xa0" in r.text:
            r.text = r.text.replace("\xa0", " ")

q1(c)
s2(c)
q2b(c)
q3a(c)
q3b(c)
q3c(c)
q4(c)
q5(c)
q6(c)
q7(c)
q8(c)
assert len(c.fig_added) == len(c.fig_order), (len(c.fig_added), len(c.fig_order))
c.doc.save(OUT)
print("saved", OUT, "figures", len(c.fig_added), "appendix", c.app_added, "tables", c.tab_n)
