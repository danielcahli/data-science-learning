# Principal Component Analysis (PCA)

This folder contains my solution to the **Principal Component Analysis** exercise from the [Kaggle Feature Engineering course](https://www.kaggle.com/learn/feature-engineering).

PCA is a powerful technique for uncovering important relationships in data and creating more informative features. While clustering partitions a dataset based on proximity, PCA partitions the variation in the data, helping to reveal its underlying structure.

---

## Workflow Overview

- Imported essential libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, and `xgboost`
- Set default Matplotlib plot style for consistent visuals
- Defined custom functions:
      - `apply_pca()`: Standardizes data, computes principal components, and returns loadings
      - `plot_variance()`: Plots explained and cumulative variance
      - `make_mi_scores()`: Computes mutual information scores for features
      - `score_dataset()`: Evaluates model performance using cross-validation and `XGBRegressor`
- Loaded the dataset and separated the target variable (`SalePrice`)
- Selected a subset of features highly correlated with `SalePrice`:
      - `GarageArea`, `YearRemodAdd`, `TotalBsmtSF`, `GrLivArea`
- Printed the correlation of these features with `SalePrice`:

```
GarageArea      0.640
YearRemodAdd    0.533
TotalBsmtSF     0.633
GrLivArea       0.707
```

- Applied PCA and printed the loadings:

|               |   PC1   |   PC2   |   PC3   |   PC4   |
|---------------|---------|---------|---------|---------|
| GarageArea    | 0.541   | -0.102  | -0.038  | 0.834   |
| YearRemodAdd  | 0.427   | 0.887   | -0.049  | -0.171  |
| TotalBsmtSF   | 0.510   | -0.361  | -0.667  | -0.406  |
| GrLivArea     | 0.514   | -0.271  | 0.743   | -0.333  |

- Engineered two new features and evaluated model performance:
      - `Feature1 = GrLivArea + TotalBsmtSF`
      - `Feature2 = YearRemodAdd * TotalBsmtSF`
- Model performance:  
      `Your score: 0.13792 RMSLE`

- Created boxen plots to visualize the distribution of each principal component:

![Distribution Plot](result_1.png)

- Analyzed the top rows for the first principal component (PC1), sorted in descending order, displaying key columns and selected features.

---

## Explanation of `dtype: float64`

The line `dtype: float64` at the beginning of the file appears to be an unintended artifact, possibly from a code snippet or output that was copied into the README. It is not relevant to the documentation and can be removed unless it serves a specific purpose, such as illustrating the data type of a variable in the PCA workflow.

## Notes

- **PCA is sensitive to feature scale.** With standardized data (mean 0, variance 1), PCA analyzes the correlation matrix; with unstandardized data, it uses the covariance matrix. Standardization ensures all features contribute equally.
- **Principal components** are linear combinations (weighted sums) of the original features. The weights are called loadings.
- **PCA transforms the data** from the original feature space to a new space defined by axes of maximum variation.

---

## Files

```
principal-component-analysis/
├── dataset.csv
├── principal_component_analysis.py
├── result_1.png
└── README.md
```