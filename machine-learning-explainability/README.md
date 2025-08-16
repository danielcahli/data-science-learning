# Machine Learning Explainability Projects

This repository contains solutions to the exercises from the **Machine Learning Explainability** course on Kaggle.  

This micro-course teachs techniques to extract insights from sophisticated machine learning models.

Each exercise focuses on a key concept in practical machine learning, such as partial plots, permutation importance, and shap values.

## Exercises Summary

### 1. Partial Plots

Partial dependence plots show how a feature affects predictions. E.g: how would similarly sized houses be priced in different areas.

Like Permutation Importance, Partial dependence plots are calculated after a model has been fit.

In this exercise was used `PartialDependenceDisplay.from_estimator` to analyze the base features. Shows how each feature individually
influences the model predictions and Visualizes the Partial Dependence Plots (PDP).

### 2. Permutation importance

Permutation importance helps answer: "If I randomly shuffle a single column in the validation data (keeping the target and other columns unchanged),
how does it affect model accuracy?" This allows us to evaluate how important each feature is to the model’s predictive power.

In this exercise was created a `PermutationImportance` object on the baseline model and displayed importances using `eli5.explain_weights()` and `format_as_text`.

### 3. SHAP Values

SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value

In this exercise was visualized the Partial Dependence Plots (PDP) to show how a feature affects the model's predictions.

### 4. Advanced Uses of SHAP Values

SHAP plots show the distribution and direction of feature effects across samples. 

In this exercise: 

- Initialized a SHAP explainer and generated SHAP values to quantify the contribution of each feature to the model’s predictions.

- Generated a SHAP beeswarm plot to visualize the distribution and relative importance of feature contributions across multiple validation sample

- Created the SHAP dependence contribution plots for `um_medications`