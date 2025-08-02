import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Read the data
train_data  = pd.read_csv(Path("C:/Users/danie/py/progs/machine-learning-intermediate/dataset/train.csv"), index_col='Id')
test_data = pd.read_csv(Path("C:/Users/danie/py/progs/machine-learning-intermediate/dataset/test.csv"), index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

#use the function above to evaluate the model performance corresponding to eight different
# values for the number of trees in the random forest: 50, 100, 150, ..., 300, 350, 400. 
#Store the results in a Python dictionary 
results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

#visualize the results
import matplotlib.pyplot as plt

plt.plot(list(results.keys()), list(results.values()))
plt.show()
