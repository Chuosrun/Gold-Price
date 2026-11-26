from q4 import *


def q5(c):
    f = c.fno
    SR, M, LG, ST = S["simple_reg_all"], S["mlr"], S["logit"], S["simple_reg_train"]
    cm = LG["cm"]
    names = {"USD_Index": "USD Index", "Yield_10Y": "Yield 10Y", "VIX": "VIX", "SP500": "S&P 500", "Oil": "Oil"}
    c.heading("V")
    c.para("We compare the models on the test set, which they did not see during training.")
    c.bullet("Performance Metrics:", "For the regression models we use RMSE, MAE and R2. R2 near 0 means the model explains almost nothing. For logistic regression we use accuracy, precision, recall, F1 and AUC.")
    c.table("Regression results on the daily gold return. All models are fitted on the same training set and tested on the same 852 test days.",
            ["Model", "Train R2", "Test RMSE", "Test MAE", "Test R2"],
            [["Always predict the training mean (baseline)", "0", f"{M['rmse_baseline']:.5f}", "-", "about 0"],
             ["Simple linear regression (USD index only)", f"{ST['r2_train']:.3f}", f"{ST['rmse_test']:.5f}", f"{ST['mae_test']:.5f}", f"{ST['r2_test']:.3f}"],
             ["Multiple linear regression (5 drivers)", f"{M['r2_train']:.3f}", f"{M['rmse_test']:.5f}", f"{M['mae_test']:.5f}", f"{M['r2_test']:.3f}"]],
            widths=[2.9, 0.9, 1.0, 1.0, 0.9], size=9)
    c.table("Logistic regression results (test set, 852 days).", ["Accuracy", "Precision", "Recall", "F1", "AUC", "Baseline accuracy"],
            [[f"{LG['accuracy']:.3f}", f"{LG['precision']:.3f}", f"{LG['recall']:.3f}", f"{LG['f1']:.3f}", f"{LG['auc']:.3f}", f"{LG['baseline_acc']:.3f}"]],
            widths=[1.0, 1.0, 1.0, 0.8, 0.8, 1.5], aligns=["c"] * 6)
    c.bullet("Comparison of models' performance:", "")
    c.sub(f"**Linear regression:** both regressions beat the baseline, but only by a small margin (RMSE {M['rmse_baseline']:.5f} for the baseline). "
          f"The multiple regression has a test R2 of {M['r2_test']:.3f}, so the 5 drivers explain about {M['r2_test']*100:.0f}% of the movement of gold on a day in the test period. "
          f"On the training set the 5 drivers fit better than the dollar alone (R2 = {M['r2_train']:.3f} vs {ST['r2_train']:.3f}), but on the test set the **simple regression is slightly better** "
          f"(test R2 = {ST['r2_test']:.3f} vs {M['r2_test']:.3f}, RMSE {ST['rmse_test']:.5f} vs {M['rmse_test']:.5f}). "
          "So the extra four variables do not help on new data. This fits the EDA and the coefficient table below, where the dollar is the one variable that really matters. "
          "Both test R2 values are lower than the multiple regression's training R2, partly because the test years (2023-2026) were more volatile than the training years, "
          "so the relationships were not exactly the same.")
    c.sub(f"**Logistic regression:** the accuracy is {pct(LG['accuracy'])} on the test set. The confusion matrix shows {cm[0][0]} days correct for down and {cm[1][1]} correct for up, with {cm[0][1]} false up and {cm[1][0]} false down. "
          f"Precision is {pct(LG['precision'])} and recall is {pct(LG['recall'])}. The AUC is {LG['auc']:.3f}, which is above 0.5 (random guessing) but not high. "
          f"To judge the accuracy we need a baseline: gold went up on {pct(LG['share_up_test'])} of the test days, so always saying \"up\" gives {pct(LG['baseline_acc'])}. "
          f"The model beats it by about {100*(LG['accuracy']-LG['baseline_acc']):.0f} points. The accuracy on the training set is {pct(LG['acc_train'])}, a bit higher than on the test set.")
    c.sub("**Important:** both models use the other markets' changes on the **same day**. So they explain what happened on a day, but they do not forecast tomorrow. "
          "A real forecast would need yesterday's information, and we expect it would be much weaker.")
    c.bullet("Feature Importance:", "")
    rows = [[names[k], f"{M['coef'][k]:.4f}", pf(M["pvalues"][k]), f"{M['std_coef'][k]:.5f}", f"{LG['odds_ratio'][k]:.2f}", pf(LG["pvalues"][k])] for k in names]
    c.table("Coefficients of the multiple regression (MLR) and the logistic regression (LOGIT). Std coef = coefficient when each input is standardized, so sizes can be compared.",
            ["Variable", "MLR coef", "MLR p", "MLR std coef", "LOGIT odds ratio", "LOGIT p"], rows, widths=[1.2, 0.9, 0.8, 1.1, 1.4, 0.9], size=9)
    c.sub("**USD Index** is the most important variable in both models. Its coefficient is negative and it has the biggest standardized effect. In the logistic regression the odds ratio is "
          f"{LG['odds_ratio']['USD_Index']:.2f}: when the dollar change goes up by one standard deviation, the odds that gold goes up are cut roughly in half. "
          f"**Yield 10Y** is second (odds ratio {LG['odds_ratio']['Yield_10Y']:.2f}), also negative: a rising yield goes with a weaker gold. "
          f"**Oil** is third with a small positive effect (odds ratio {LG['odds_ratio']['Oil']:.2f}). **VIX** and **S&P 500** are not significant (p = {pf(M['pvalues']['VIX'])} and {pf(M['pvalues']['SP500'])} in the regression), "
          "so they add nothing once the dollar and the yield are in the model. This matches the EDA, where the dollar was the clearest driver.")
