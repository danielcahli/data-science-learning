# Intermediate Machine Learning Projects

This repository contains solutions to the exercises from the **Intermediate Machine Learning** course on Kaggle.  
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

### 3. Categorical Variables
The dataset was split into training and test sets.  
Categorical features were handled in three different ways:  
1. Removing them entirely  
2. Applying **'Ordinal Encoding'**  
3. Applying **'One-Hot Encoding'** (for low-cardinality features)  

Model performance for each approach was evaluated using Mean Absolute Error (MAE).

### 4. Pipelines
Pipeline is a powerful utility in scikit-learn that allows you to chain multiple preprocessing steps together and apply them sequentially. 
This not only keeps your code clean and organized but also ensures consistency during model training and evaluation.
Pipelines are especially useful when combined with tools like ColumnTransformer and are essential for building robust machine learning workflows.

Why use Pipelines?

Cleaner Code: Preprocessing steps can become messy and error-prone when handled manually. Pipelines abstract these steps into a single, well-structured object
eliminating the need to track training and validation transformations separately.

Fewer Bugs: By encapsulating preprocessing logic, pipelines reduce the risk of accidentally omitting a step or misapplying a transformation.

Easier to Productionize: Deploying machine learning models often involves replicating preprocessing exactly as done during training. Pipelines package preprocessing
and modeling steps into one object, simplifying deployment in production environments.

Improved Model Validation: Pipelines integrate seamlessly with cross-validation techniques, allowing you to validate preprocessing and model fitting in a single step.
You'll see this in action in the next section on cross-validation.

### 5. Cross-Validation
Implemented cross-validation using `cross_val_score` to get more reliable model performance estimates.  

In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.

For small datasets, where extra computational burden isn't a big deal, you should run cross-validation.

For larger datasets, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

### 6. XGBoost
The data was One-hot encoded using **get_dummies()** from Pandas.
A prediction model was created using a **XGBRegressor** Class.

XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting
with several additional features focused on performance and speed. 

Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. 
(Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

## Data leakage 
Leakage happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. 
This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.
There are two main types of leakage: target leakage and train-test contamination.

*Target leakage* occurs when your predictors include data that will not be available at the time you make predictions. It is important to think about target leakage in terms 
of the timing or chronological order that data becomes available, not merely whether a feature helps make good predictions. To prevent this type of data leakage, any variable
updated (or created) after the target value is realized should be excluded.

*Train-Test Contamination* occurs when you aren't careful to distinguish training data from validation data. Recall that validation is meant to be a measure of how the model does
on data that it hasn't considered before. You can corrupt this process in subtle ways if the validation data affects the preprocessing behavior. For example, imagine you run preprocessing 
(like fitting an imputer for missing values) before calling train_test_split(). The end result? Your model may get good validation scores, giving you great confidence in it, but perform poorly
when you deploy it to make decisions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Jupyter Notebooks

## How to Run

Open any of the Jupyter Notebooks inside the corresponding exercise folders and run the cells sequentially.  
Make sure required packages are installed (`pandas`, `scikit-learn`, `xgboost`).
