# Intermediate Machine Learning Projects

This repository contains my solutions to the 7 exercises from the **Intermediate Machine Learning** course on Kaggle.  
Each exercise focuses on a key concept in practical machine learning, such as missing values, pipelines, and model tuning.

## Exercises Summary

### 1. Introduction

In this exercise, a training and a test dataset were used to predict house sale prices based on seven selected features:  
`LotArea`, `YearBuilt`, `1stFlrSF`, `2ndFlrSF`, `FullBath`, `BedroomAbvGr`, and `TotRmsAbvGrd`.

Five different models were created using `RandomForestRegressor`, each with varying hyperparameters.  
These models were evaluated using **Mean Absolute Error (MAE)** on a validation set to compare their performance.

After selecting the preferred configuration, the final model was trained on the full training dataset,  
used to generate predictions on the test dataset, and the results were saved in a CSV file suitable for competition submission.

### 2. Missing Values
In this exercise, missing values were handled using two different approaches.  
A helper function was created to evaluate each method by calculating the **Mean Absolute Error (MAE)** on a validation set.

- In the first approach, columns with missing values were dropped entirely.  
- In the second approach, missing values were filled using **`SimpleImputer`** from scikit-learn.

This comparison helps determine which strategy performs better in terms of model accuracy.

### 2. Categorical Variables
The dataset was split into training and test sets.  
Categorical features were handled in three different ways:  
1. Removing them entirely  
2. Applying Ordinal Encoding  
3. Applying One-Hot Encoding (for low-cardinality features)  

Model performance for each approach was evaluated using Mean Absolute Error (MAE).

### 3. Pipelines
Built a preprocessing pipeline to combine numerical and categorical transformations.  
Used `ColumnTransformer` and `Pipeline` to streamline data preparation.

### 4. Cross-Validation
Implemented cross-validation using `cross_val_score` to get more reliable model performance estimates.  
Learned the trade-offs between validation sets and cross-validation.

### 5. XGBoost
Introduced the `XGBoostRegressor`, a powerful model for tabular data.  
Compared its performance with Random Forest using MAE.

### 6. Data Leakage
Identified different types of data leakage (target leakage, train-test leakage).  
Prevented leakage by correctly splitting data and avoiding certain features.

### 7. Final Project (Model Selection)
Built and compared multiple models using different hyperparameters.  
Trained the best model on the full training data and generated predictions on the test set.  
Saved results to a `.csv` file ready for submission.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Jupyter Notebooks

## How to Run

Open any of the Jupyter Notebooks inside the corresponding exercise folders and run the cells sequentially.  
Make sure required packages are installed (`pandas`, `scikit-learn`, `xgboost`).
