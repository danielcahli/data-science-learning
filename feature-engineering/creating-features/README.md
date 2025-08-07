
# Creating Features Project

This folder contains my solution to the **"Creating Features"** exercise from the [Kaggle Feature Engineering course](https://www.kaggle.com/learn/feature-engineering).

The goal of this exercise is to enrich the dataset by **creating new features** that provide useful signals to the model. These features are derived through mathematical transformations, feature interactions, and aggregations.

---

## Overview of Steps

1. **Import libraries and define a scoring function**
   - Uses `cross_val_score` with `XGBRegressor` and RMSLE as the evaluation metric.

2. **Load dataset**
   - The dataset is loaded as a DataFrame, and the target variable (`SalePrice`) is separated.

3. **Create new features**:

### Mathematical Transforms (`X_1`)
- `LivLotRatio`: Ratio of living area to lot area.  
- `Spaciousness`: Floor area per above ground room.  
- `TotalOutsideSF`: Combined square footage of all porch-related features.

### Feature Interactions (`X_2`)
- One-hot encode `BldgType` using `get_dummies()`.
- Multiply each dummy column by `GrLivArea` to capture interactions.

### Porch Count (`X_3`)
- `PorchTypes`: Number of porch-related features (out of 5) that are greater than 0.

### Class Extraction (`X_4`)
- `MSClass`: Extracts the numeric part of `MSSubClass` by splitting at the underscore.

### Neighborhood Median Area (`X_5`)
- `MedNhbdArea`: Median of `GrLivArea` grouped by `Neighborhood`.

4. **Join all created features** to the base dataset.

5. **Evaluate model performance** using the `score_dataset()` function.
