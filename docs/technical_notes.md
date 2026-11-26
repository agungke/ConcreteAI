# ConcreteAI — Technical Notes

## Research data and provenance
The study uses the UCI Concrete Compressive Strength dataset donated by I-Cheng Yeh. It contains 1,030 observations, eight numerical predictors and one target in MPa. DOI: [10.24432/C5PK67](https://doi.org/10.24432/C5PK67). License: CC BY 4.0. The acquisition script downloads from the official UCI CSV endpoint, checks the original header and expected row count, verifies numeric finite values and basic physical domains, then changes column names only.

## Data quality protocol
- Confirm exactly 1,030 rows and nine columns.
- Check missing/non-finite values, numeric conversion and basic physical domains.
- Report exact full-row duplicates and repeated feature-profile observations; do not silently delete them.
- Use the seven ingredient quantities (excluding curing age) as the group key. This keeps every age/replicate of one mix on the same side of the holdout split.
- Fixed random seed: 42; group-aware holdout: 80/20 target fraction; five-fold GroupKFold on training mixtures only.

The holdout split is by groups, so the resulting row ratio can differ slightly from exactly 80/20. This is intentional and avoids mixture leakage.

## Mathematical foundation

### Linear Regression
Ordinary Least Squares estimates coefficients by minimizing the residual sum of squares:

\[
\min_{\beta_0,\beta} \sum_{i=1}^{n}\left(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\right)^2
\]

The fitted prediction is \(\hat y=\hat\beta_0+\sum_j\hat\beta_jx_j\). Coefficients are conditional associations and should not be interpreted causally, especially because concrete ingredients can be correlated.

### Random Forest Regressor
Random Forest averages predictions from an ensemble of decision trees:

\[
\hat f_{RF}(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
\]

Trees are trained with bootstrap samples and randomized feature subsets. This allows nonlinear relationships and interactions without requiring a manually specified equation.

## Evaluation
- MAE: mean absolute error in MPa.
- RMSE: square root of mean squared error in MPa; penalizes large errors more strongly.
- R²: proportion of target variance explained relative to a constant mean predictor.

Report train metrics as a diagnostic only. Generalization claims must use held-out test metrics and group cross-validation together.

## Interpretation cautions
Impurity-based Random Forest feature importance is model-specific and may be affected by correlated predictors. It is not causal evidence. Model coefficients are also affected by correlations among mix constituents.

## Verification
- Reproducible split and model random seed: 42.
- Group overlap assertion between train and test.
- Five-fold GroupKFold using training mixture groups only.
- Offline unit tests use artificial fixtures only to test validation logic; they are not research data.
- GitHub Actions runs unit tests and a separate training workflow downloads the official dataset and executes the real pipeline.

## Limitations and engineering relevance
The dataset is historical and relatively small. It cannot guarantee performance on unfamiliar cement sources, aggregates, admixtures, curing environments, standards or production processes. The application is an academic estimation prototype, not a replacement for laboratory testing, material qualification, structural design or safety decisions.
