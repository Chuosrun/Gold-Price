from q1 import *


def q4(c):
    R = c.R
    f = c.fno
    SP, SR, M, LG = S["split"], S["simple_reg_all"], S["mlr"], S["logit"]
    ST = S["simple_reg_train"]
    c.heading("IV")
    c.para("We use regression models from the course: simple linear regression, multiple linear regression and logistic regression. "
           "The target is the **daily gold return**, not the price, because the EDA showed that the price levels have a trend that makes regressions look better than they are.")
    c.bullet("Data Splitting:", f"The data is a time series, so we did not shuffle it. The first 80% of the days went to training ({SP['train']:,} days, {SP['train_start']} to {SP['train_end']}) "
             f"and the last 20% to testing ({SP['test']:,} days, {SP['test_start']} to {SP['test_end']}), with `train_test_split(..., test_size=0.2, shuffle=False)`. "
             "A random split would put days from the future in the training set. The inputs are the daily changes of the USD index, the 10-year yield, VIX, S&P 500 and Oil on the same day.")
    c.bullet("Model Selection & Training:", "")
    c.sub(f"**Simple linear regression (Figure {f('s03_simple_regression')}):** gold return = a + b * (USD index return), using all days. The fitted line is y = {SR['intercept']:.4f} + ({SR['slope']:.3f}) x, "
          f"with r = {SR['r']:.3f} and R2 = {SR['r2']:.3f}. The slope is negative: when the dollar rises 1%, gold falls about {abs(SR['slope']):.2f}% on average. The p-value of the slope is {pf(SR['p'])}. "
          f"The dollar alone explains about {SR['r2']*100:.1f}% of the daily movement of gold. "
          f"This line and R2 use all days, only to draw the picture. To compare it fairly with the other models we also fit the same simple regression on the training set only (R2 = {ST['r2_train']:.3f}) and test it on the test set in Section V.")
    c.figure("s03_simple_regression", "Simple linear regression of the daily gold return on the daily USD index return.", max_w=4.8)
    c.sub(f"**Multiple linear regression (Figure {f('s04_mlr_actual_pred')}):** we fit all 5 drivers on the training set with `statsmodels` OLS, like in Lab 8. The model is "
          f"R2 = {M['r2_train']:.3f} (adjusted R2 = {M['adj_r2_train']:.3f}) on training, and the F-test p-value is {pf(M['f_p'])}, so the model as a whole is significant. "
          f"The VIF values are all below 3, well under the usual limit of 5 (USD Index {M['vif']['USD_Index']}, Yield {M['vif']['Yield_10Y']}, VIX {M['vif']['VIX']}, S&P 500 {M['vif']['SP500']}, Oil {M['vif']['Oil']}), "
          "so there is no multicollinearity problem. The plot shows actual and predicted returns on the test set: the points follow the red line a little, with a lot of noise.")
    c.figure("s04_mlr_actual_pred", "Multiple regression: actual vs predicted daily gold return on the test set.", max_w=4.6)
    c.sub(f"**Logistic regression (Figure {f('s05_confusion')}):** like in Lab 9, we predict a yes/no target: did gold go **up** (1) or **down** (0) on the day. We standardized the 5 inputs and fit `statsmodels` Logit on the training set, "
          "using 0.5 as the cut-off for the predicted probability. The confusion matrix on the test set is shown below.")
    c.figure("s05_confusion", "Logistic regression: confusion matrix on the test set.", max_w=3.6)
