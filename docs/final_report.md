# FINAL PROJECT REPORT — ConcreteAI

**Course:** Artificial Intelligence  
**Project theme:** Data-driven Engineering Solution  
**Academic title:** *Concrete Compressive Strength Prediction Using Linear Regression and Random Forest: A Data-Driven Engineering Solution*  
**Team:** [Fill in three member names]  
**Institution:** [Fill in institution / study program]  
**Year:** 2026

> Draft report based on the executed UCI dataset pipeline and the artifacts currently stored in this repository. Complete the team identity and institutional details before submission.

## Abstract

Concrete compressive strength is an important engineering property that is commonly determined through laboratory testing. This project develops ConcreteAI, an academic machine-learning prototype that estimates concrete compressive strength from seven mixture ingredients and concrete age. The study uses the Concrete Compressive Strength dataset by I-Cheng Yeh from the UCI Machine Learning Repository, consisting of 1,030 observations, eight numerical input features and one target measured in MPa. Linear Regression is used as the baseline model and Random Forest Regressor as the nonlinear comparison model. To reduce data leakage, observations are split by the seven ingredient quantities, excluding age, using GroupShuffleSplit with a random seed of 42. The resulting training and testing partitions contain 815 and 215 observations, respectively. Five-fold GroupKFold is applied to the training groups. On the held-out test set, Linear Regression obtains MAE 8.3832 MPa, RMSE 10.7361 MPa and R² 0.6453, while Random Forest obtains MAE 4.0332 MPa, RMSE 5.8886 MPa and R² 0.8933. Random Forest also performs better in grouped cross-validation. The results indicate that the ensemble model captures nonlinear patterns in this dataset more effectively than the linear baseline. However, a train-test performance gap and several large residuals remain. Therefore, ConcreteAI is an academic estimation prototype and cannot replace laboratory testing or engineering safety decisions.

**Keywords:** Artificial Intelligence, Machine Learning, Concrete Compressive Strength, Linear Regression, Random Forest, Regression, Data-driven Engineering.

---

## 1. Introduction

### 1.1 Background

Concrete is widely used in civil engineering and construction. Its compressive strength is a key property for evaluating concrete performance. Conventional compressive-strength testing requires concrete specimens, curing time and laboratory equipment. Machine Learning offers a data-driven approach for estimating strength from historical measurements of mixture composition and concrete age.

ConcreteAI applies supervised regression to estimate compressive strength from the quantities of cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate and concrete age.

### 1.2 Problem statement

The project addresses the following questions:

1. Can concrete compressive strength be estimated from mixture ingredients and curing age using supervised Machine Learning?
2. How does Linear Regression perform as a baseline?
3. Does Random Forest improve prediction performance by capturing nonlinear relationships?
4. Which input features are influential in the fitted Random Forest model?
5. How reliable are the models when observations from the same mixture are kept together during evaluation?

### 1.3 Objectives

- Prepare and validate the UCI Concrete Compressive Strength dataset.
- Perform exploratory data analysis.
- Train Linear Regression and Random Forest models.
- Compare both models using MAE, RMSE and R².
- Analyze feature importance, residuals and possible overfitting.
- Provide a Streamlit application for interactive demonstration.

### 1.4 Scope and safety

The project is limited to the historical UCI dataset and the two selected regression models. It does not validate concrete mix designs against construction standards and does not replace laboratory compressive-strength testing, structural design or professional engineering judgement.

## 2. Dataset and theoretical foundation

### 2.1 Dataset source

The dataset was created by I-Cheng Yeh and is distributed through the UCI Machine Learning Repository. It contains 1,030 observations and nine columns: eight quantitative predictors and one quantitative target. UCI metadata reports no missing attribute values. The dataset is licensed under CC BY 4.0.

Dataset DOI: https://doi.org/10.24432/C5PK67

### 2.2 Features and target

| Feature | Description | Unit |
|---|---|---|
| cement_kg_m3 | Cement | kg/m³ |
| slag_kg_m3 | Blast Furnace Slag | kg/m³ |
| fly_ash_kg_m3 | Fly Ash | kg/m³ |
| water_kg_m3 | Water | kg/m³ |
| superplasticizer_kg_m3 | Superplasticizer | kg/m³ |
| coarse_aggregate_kg_m3 | Coarse Aggregate | kg/m³ |
| fine_aggregate_kg_m3 | Fine Aggregate | kg/m³ |
| age_days | Concrete age | days |
| strength_mpa | Concrete compressive strength (target) | MPa |

### 2.3 Linear Regression

Linear Regression estimates a linear relationship between input variables and a continuous target. The prediction equation is:

\[
\hat y=\beta_0+\sum_{j=1}^{p}\beta_jx_j
\]

The Ordinary Least Squares method estimates the coefficients by minimizing the sum of squared residuals:

\[
\min_{\beta_0,\beta}\sum_{i=1}^{n}
\left(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\right)^2
\]

Linear Regression is used as the baseline because it is simple, interpretable and provides a reference for evaluating a nonlinear model.

### 2.4 Random Forest Regressor

Random Forest is an ensemble learning method that combines multiple decision trees. Each tree is trained using bootstrap samples and randomized feature subsets. For regression, the final prediction is the average of the tree predictions:

\[
\hat f_{RF}(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
\]

This approach can represent nonlinear relationships and interactions among mixture ingredients without requiring a manually specified equation. The implementation uses 500 estimators and random seed 42.

### 2.5 Evaluation metrics

**Mean Absolute Error (MAE):**

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|
\]

MAE measures the average absolute prediction error in MPa.

**Root Mean Squared Error (RMSE):**

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}
\]

RMSE penalizes large errors more strongly than MAE.

**Coefficient of Determination (R²):**

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}
\]

R² compares model errors with a constant prediction based on the test-set mean.

## 3. Methodology

### 3.1 Data acquisition and validation

The project downloads the original CSV from the official UCI endpoint. The preparation script checks the original column header, the expected 1,030 observations, nine values per row, numeric conversion, finite values and basic physical domains. It standardizes column names while retaining observation values.

### 3.2 Data quality analysis

The validated dataset contains:
- 1,030 observations and nine columns.
- Zero missing or non-finite values.
- 25 exact duplicate rows.
- 38 repeated feature-plus-age profiles.
- 427 unique ingredient mixtures.

Rows were retained. Duplicate counts were reported rather than automatically removing observations.

### 3.3 Exploratory Data Analysis

Descriptive statistics, feature distributions, a correlation matrix and a correlation heatmap were generated. The target has a mean of 35.818 MPa, a standard deviation of 16.706 MPa and an observed range from 2.33 to 82.60 MPa.

Pearson correlations with compressive strength include cement (0.498), superplasticizer (0.366), age (0.329) and water (-0.290). Correlation describes linear association and does not establish causality.

### 3.4 Leakage-aware data splitting

A group key is constructed from the seven ingredient quantities; concrete age is excluded. This ensures that observations from the same ingredient mixture, including different curing ages, cannot be split between training and testing.

- Random seed: 42.
- Split method: GroupShuffleSplit.
- Target test fraction: 20%.
- Training set: 815 rows and 341 unique mixtures.
- Test set: 215 rows and 86 unique mixtures.
- Total unique mixtures: 427.

The actual row proportion is approximately 79.1% training and 20.9% testing because complete groups are kept together.

### 3.5 Model training and validation

Both models are trained on the training partition. Five-fold GroupKFold cross-validation is performed only on training data, with the same mixture grouping principle. Test metrics are calculated on the held-out test partition.

The project stores trained models, metrics, test predictions, model comparison, descriptive statistics, correlation data, diagnostic plots, feature importance and Linear Regression coefficients.

## 4. Results and discussion

### 4.1 Held-out test results

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---:|---:|---:|
| Linear Regression | 8.3832 | 10.7361 | 0.6453 |
| Random Forest | **4.0332** | **5.8886** | **0.8933** |

Random Forest reduces test MAE by approximately 51.9% and test RMSE by approximately 45.2% relative to Linear Regression. Its R² is higher by approximately 0.248. Based on these held-out metrics, Random Forest is the stronger model in this experiment.

### 4.2 Group cross-validation

| Model | Mean MAE (MPa) | MAE standard deviation | Mean RMSE (MPa) | Mean R² |
|---|---:|---:|---:|---:|
| Linear Regression | 8.3408 | 0.5040 | 10.4693 | 0.5761 |
| Random Forest | **4.7932** | 0.4915 | **6.5153** | **0.8366** |

The five-fold grouped validation confirms the same model ranking as the held-out test. This provides additional evidence that the Random Forest advantage is not limited to one particular holdout partition.

### 4.3 Overfitting analysis

| Model | Training MAE | Test MAE | Training R² | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 8.2174 | 8.3832 | 0.6015 | 0.6453 |
| Random Forest | 1.2294 | 4.0332 | 0.9855 | 0.8933 |

Random Forest has a noticeable train-test performance gap. Its training error is much lower than its test error, indicating some overfitting. However, the model still performs substantially better than Linear Regression on both the holdout test and grouped cross-validation. This result supports using Random Forest as the preferred model for this dataset while clearly communicating its generalization limitations.

### 4.4 Feature importance

Random Forest impurity-based feature importance is:

| Feature | Importance |
|---|---:|
| Age | 34.36% |
| Cement | 28.59% |
| Water | 10.86% |
| Superplasticizer | 8.71% |
| Blast Furnace Slag | 7.70% |
| Fine Aggregate | 4.88% |
| Coarse Aggregate | 2.81% |
| Fly Ash | 2.09% |

Age and cement have the highest importance in the fitted model. These scores describe how the model uses features; they do not establish that changing one ingredient alone will cause a corresponding change in real concrete strength.

### 4.5 Residual analysis

For the 215 held-out test observations, Random Forest residuals are calculated as actual strength minus predicted strength.

- Mean residual: -0.43 MPa.
- Median absolute error: 2.56 MPa.
- 90th percentile absolute error: 9.52 MPa.
- Maximum absolute error: 31.97 MPa.
- Underpredictions: 90 observations.
- Overpredictions: 125 observations.

The largest absolute error is an underprediction for an observation with actual strength 81.75 MPa and predicted strength 49.78 MPa. This shows that a good overall R² does not guarantee accurate estimates for every concrete mixture, particularly at the high-strength end.

### 4.6 Linear Regression coefficients

Selected fitted coefficients are cement +0.1219, water -0.1218, superplasticizer +0.4503 and age +0.1051, with target measured in MPa and inputs in their original units. The coefficients are conditional on all other inputs and should not be interpreted as causal material effects. Correlations among mixture ingredients can influence coefficient magnitudes and signs.

## 5. Conclusion

ConcreteAI successfully implements a data-driven regression pipeline using the UCI Concrete Compressive Strength dataset. The pipeline validates the data, reports duplicate observations, uses mixture-aware splitting to reduce leakage, trains Linear Regression and Random Forest, and evaluates both models using MAE, RMSE, R² and grouped cross-validation.

Random Forest performs better than the Linear Regression baseline on the held-out test set, achieving MAE 4.0332 MPa, RMSE 5.8886 MPa and R² 0.8933. The same ranking appears in five-fold GroupKFold validation. The result suggests that the nonlinear ensemble model is better able to represent patterns in this dataset than the linear baseline.

However, Random Forest shows a train-test performance gap and has a large worst-case residual. The model is therefore not guaranteed to generalize to unfamiliar materials, suppliers, mix designs, curing regimes or construction conditions. ConcreteAI must be treated as an academic prototype. Laboratory testing, applicable standards and qualified engineering judgement remain essential.

## 6. Future work

- Evaluate permutation feature importance and partial dependence to complement impurity-based importance.
- Compare additional models only if time and rubric allow.
- Evaluate uncertainty and error behavior across strength ranges.
- Test generalization on independently collected concrete data.
- Record each member's actual contribution using commits, pull requests and peer confirmation.

## References

1. Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research, 28*(12). https://doi.org/10.1016/S0008-8846(98)00165-3
2. Yeh, I.-C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67. License: CC BY 4.0.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
