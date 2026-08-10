# 363 Features, 399 Rows: What Ridge Regularisation Actually Buys You

Predicting house prices from degree-3 polynomial features, and measuring how
much regularisation improves not just accuracy but the *stability* of what the
model learns.

## The question

Expanding 11 features to degree 3 produces 363 polynomial terms from only 399
observations. That should overfit badly. I wanted to quantify three things:

1. How much does Ridge actually improve predictive accuracy here?
2. How stable are the learned coefficients when the training data changes?
3. Does regularisation make the model easier to interpret, not just more accurate?

## Data

399 observations, 11 neighbourhood and property features, predicting median
house price in thousands of USD. This is a renamed subset of the Boston Housing
dataset, which was removed from scikit-learn in version 1.2 over the ethical
problems with its original construction. I've used it here purely as a
regression benchmark, and the coefficients should not be read as claims about
housing markets or the communities in the data.

## Method

- `StandardScaler` → `PolynomialFeatures(degree=3)` → `Ridge`, all inside a
  `Pipeline` so scaling is fit within each CV fold rather than on the full set
- Scaling before expansion matters: unscaled, the cubic term of `Tax` (values
  around 300 to 700) would dwarf the cubic term of `NitricOxides` (0.4 to 0.9)
- Alpha tuned across 0.01 to 1000 by 5-fold CV on the training set only
- Ridge over Lasso deliberately: with heavily correlated polynomial terms, Lasso
  arbitrarily picks one from each correlated group. Ridge shrinks them together
- Coefficient stability measured by 50 bootstrap resamples, comparing Ridge
  against the unregularised model

## Results

| Model | CV RMSE | CV std | Test RMSE | Test R² |
|---|---|---|---|---|
| Degree-3 polynomial, no regularisation | 93.03k | 33.56k | — | — |
| Ridge (α = 50) on degree-3 polynomial | 4.05k | 0.82k | 2.90k | 0.897 |

The unregularised model's CV RMSE of 93k against a target that only ranges from
5k to 50k is the headline: it isn't a slightly worse model, it's unusable.

The bootstrap comparison is the more interesting result. Mean coefficient
standard deviation across 50 resamples was 0.084 for Ridge and 7.29 without
regularisation, so the unregularised coefficients were roughly **87 times more
variable**. Retrain on a slightly different sample and you get a completely
different model.

`Rooms` and `DisadvantagedPosition` and their interaction terms come out as the
dominant, and most stable, signals. Features with high coefficient of variation
are mostly minor interaction terms, which Ridge shrinks toward zero so they
don't destabilise predictions.

## Limitations

- Train R² 0.935 vs test R² 0.897 shows mild residual overfitting still present
- Prices are capped at 50k in the source data, which distorts upper-tail
  predictions and shows as a funnel in the residual plot
- Individual coefficients in a degree-3 expansion of correlated, standardised
  features should be read as indicative of where the signal sits, not as clean
  marginal effects
- 50 bootstrap iterations is enough to show the effect but a larger number would
  tighten the estimates

## Running it

Open `house-price-ridge-regression.ipynb` in Jupyter or Google Colab.

Requires: numpy, pandas, matplotlib, seaborn, scikit-learn
