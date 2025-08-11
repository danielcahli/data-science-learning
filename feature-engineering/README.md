## Feature Engineering Projects
This repository contains my solutions to the exercises from the **Kaggle Feature Engineering course**.
Each project focuses on key concepts and practical techniques for creating effective features for machine learning models.

## Key Skills Demonstrated
Feature Engineering Techniques: Mutual Information, Target Encoding, PCA, Feature Interactions, Mathematical Transformations, Aggregations.

Categorical Encoding: M-estimate encoding, frequency encoding, label encoding.

Visualization: Seaborn facet grids, boxen plots, lmplots, scatter plots, and distribution plots for feature analysis.

Model Evaluation: RMSLE scoring, 5-fold cross-validation, XGBRegressor modeling.

Data Leakage Prevention: Proper encoding split strategies and holdout validation.

Dimensionality Reduction: PCA loadings analysis, principal component feature creation.

## Exercises Summary

## Creating Features
Goal: Enrich the dataset with new features to provide stronger predictive signals.

Steps:

Imported libraries and defined a score_dataset() function using cross_val_score with XGBRegressor and RMSLE.

Loaded dataset and separated SalePrice as target.

Applied mathematical transformations, feature interactions, and aggregations to create new features.

Joined all engineered features to the base dataset.

Evaluated model performance before and after feature creation.

## Mutual Information 
Further evaluation of individual feature predictive power using Mutual Information.

Visualized top 20 features with a horizontal bar chart.

Created a faceted scatter plot of SalePrice vs selected features.

Used a boxen plot to show SalePrice distribution by BldgType.

Plotted trend lines with sns.lmplot:

GrLivArea by BldgType → interaction found.

MoSold by BldgType → no interaction found.

Emphasized that MI is a univariate metric — it measures individual feature relevance without considering feature combinations.

## Clustering With k-Means

Applied K-Means clustering as a feature engineering technique to the Ames housing dataset.
Two approaches were tested:

Adding cluster labels as a categorical feature.

Adding distances to all centroids as numerical features.

Workflow:

Selected numeric features related to property size.

Standardized the features before clustering.

Trained KMeans with 10 clusters.

Visualized clusters and added results to dataset.

Evaluated model performance with and without clustering features using XGBRegressor + cross-validation.

Key insight:
Both cluster labels and centroid distances can enrich the feature set. Centroid distances provide more nuanced information, often improving predictive performance more than labels alone.


## Principal Component Analysis (PCA)
PCA was applied to partition variation in the dataset and reveal underlying structure.

Analyzed correlations between features and target variable.

Applied PCA and printed component loadings.

Engineered two new features from principal components.

Evaluated model performance with the new features.

Created boxen plots for each principal component.

Analyzed top rows for the first principal component (PC1), sorted in descending order, highlighting important features.

## Target Encoding
Implemented M-estimate target encoding to replace categorical values with numbers based on the mean target value per category.

Evaluated performance impact using XGBRegressor with RMSLE:

Encoding helped when the feature had predictive signal.

No improvement when the signal was weak.

Demonstrated data leakage risk when fitting encoder on full dataset.

Showed how smoothing (m) helps stabilize encodings for rare categories.