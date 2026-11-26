# ConcreteAI

**Concrete Compressive Strength Prediction Using Linear Regression and Random Forest: A Data-Driven Engineering Solution.**

Academic regression prototype predicting concrete compressive strength (MPa) from seven mix ingredients and concrete age. Models: Linear Regression baseline and Random Forest comparison.

## Dataset
I-Cheng Yeh, *Concrete Compressive Strength*, UCI Machine Learning Repository, DOI [10.24432/C5PK67](https://doi.org/10.24432/C5PK67), CC BY 4.0. 1,030 observations, eight quantitative inputs and one target. Preserve data values and standardize only headers.

## Run
```bash
pip install -r requirements.txt
python src/prepare_concrete_dataset.py
python -m pytest -q
python src/train.py
streamlit run app.py
```

Training creates metrics, grouped cross-validation, predictions, EDA charts, diagnostics, coefficients, feature importance and serialized models.

## Validation
Fixed seed 42. Holdout uses GroupShuffleSplit (80/20) grouped by the seven ingredient quantities, ensuring different ages/replicates of the same mixture cannot cross train/test. Five-fold GroupKFold runs on training mixtures only. Metrics: MAE, RMSE and R². Exact duplicates are reported, not silently removed.

## Verified experiment
The current UCI training run is documented in [`docs/analysis.md`](docs/analysis.md). Actual outputs are stored in `results/`: metrics, model comparison, test predictions, descriptive statistics, correlation matrix, diagnostic charts, coefficients and feature importance. Trained model files are stored in `models/`. The training workflow downloads the official UCI CSV at runtime and publishes model/results artifacts.

## Mathematical basis
Linear Regression minimizes ordinary least squares: min Σ(yᵢ − β₀ − xᵢᵀβ)². Random Forest averages decision-tree predictions trained with bootstrap samples and randomized feature subsets. Feature importance is descriptive, not causal.

## Safety and limitations
Historical dataset and limited sample size constrain generalization to unseen materials, suppliers, curing regimes and modern construction practice. This app is an academic prototype, not a substitute for laboratory testing, applicable standards, structural design or engineering judgement.

## Team contribution
Record each member's work using actual commits, pull requests and peer confirmation. Never fabricate contribution history.
