# Feature Engineering Projects

This repository contains solutions to the exercises from the Feature Engineering course on [Kaggle Feature Engineering course](https://www.kaggle.com/learn/feature-engineering).
Each notebook focuses on key concepts and practical techniques for creating effective features for machine learning models.

## Exercises Summary

### 1. Mutual Information

In this exercise:

A function was created to convert all discrete features to integer values.

Seaborn was used to create faceted scatter plots of SalePrice vs selected features.

Mutual Information scores were computed for the Ames dataset and visualized in a horizontal bar chart (top 20 features).

A boxen plot was used to compare the distribution of SalePrice across BldgType categories.

Interaction between GrLivArea and BldgType was analyzed using lmplot, showing different trend lines by category.

In contrast, MoSold showed similar trends across categories, suggesting no interaction.

### 2. Creating Features

The goal of this exercise is to enrich the dataset by creating new features that provide useful signals to the model. 

These features are derived through mathematical transformations, feature interactions, and aggregations.

Overview of Steps
Import libraries and define a scoring function

Uses cross_val_score with XGBRegressor and RMSLE as the evaluation metric.
Load dataset

The dataset is loaded as a DataFrame, and the target variable (SalePrice) is separated.

Mathematical Transforms was used to create new features

Join all created features to the base dataset.

Evaluate model performance using the score_dataset() function.

### 3. Clustering With K-Means

