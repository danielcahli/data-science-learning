## Permutation importance

This folder contains my solution to the **"Permutation Importance"** exercise from the [Kaggle Machine Learning Explainability Course](https://www.kaggle.com/learn/machine-learning-explainability).

Permutation importance is calculated after a model has been trained. It doesn’t change the model or its predictions, but helps answer:

"If I randomly shuffle a single column in the validation data (keeping the target and other columns unchanged), how does it affect model accuracy?"

This allows us to evaluate how important each feature is to the model’s predictive power.


## Workflow Summary

- Imported all required libraries

- Loaded and cleaned the dataset:
  - Removed extreme outliers and invalid fare values

- Extracted the target variable `fare_amount` into a separate series `y`

- Defined a list of base features and created the feature matrix `X`

- Split the dataset into training and validation sets using `train_test_split`

- Trained a baseline model with `RandomForestRegressor` using `fit()`

- Displayed the first rows of the cleaned dataset for inspection

**Sample of cleaned data:**

| fare_amount | pickup_datetime       | pickup_longitude | pickup_latitude | dropoff_longitude | dropoff_latitude | passenger_count |
|-------------|------------------------|------------------|------------------|--------------------|------------------|------------------|
| 5.7         | 2011-08-18 00:35:00 UTC | -73.982738       | 40.761270        | -73.991242         | 40.750562        | 2                |
| 7.7         | 2012-04-21 04:30:42 UTC | -73.987130       | 40.733143        | -73.991567         | 40.758092        | 1                |
| ...         | ...                    | ...              | ...              | ...                | ...              | ...              |

- Used `describe()` to understand feature distribution in the cleaned dataset

## Baseline Feature Importances

- Created a `PermutationImportance` object on the baseline model and displayed importances using `eli5.explain_weights()` and `format_as_text`.

0.8426 ± 0.0168 dropoff_latitude
0.8269 ± 0.0211 pickup_latitude
0.5943 ± 0.0436 pickup_longitude
0.5387 ± 0.0273 dropoff_longitude
-0.0020 ± 0.0013 passenger_count

## Feature Engineering

- Created two new features:
  - `abs_lat_change`: Absolute change in latitude
  - `abs_lon_change`: Absolute change in longitude
- Trained a new model with these features
- - Evaluated feature importances again using permutation importance

**Result:**
0.5952 ± 0.0575 abs_lat_change
0.4485 ± 0.0493 abs_lon_change
0.0799 ± 0.0241 pickup_latitude
0.0770 ± 0.0121 dropoff_latitude
0.0694 ± 0.0115 pickup_longitude
0.0596 ± 0.0131 dropoff_longitude

## Key Takeaways

- Permutation importance provides a **model-agnostic** way to measure feature relevance.
- Features like latitude and longitude had high baseline importance.
- After creating engineered features (absolute coordinate changes), they proved to be even more important than the original coordinates.
