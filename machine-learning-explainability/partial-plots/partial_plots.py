import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
from numpy.random import rand

data = pd.read_csv('machine-learning-explainability/train.csv', nrows=50000)

data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )

y = data.fare_amount

base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude']

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)

print("Data sample:")

print(data.head())

print(data.describe())

# Create Partial Dependence Plots
for feat_name in base_features:
    g = PartialDependenceDisplay.from_estimator(first_model, val_X, [feat_name])
    g.figure_.savefig(f"result_{feat_name}.png")

fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('pickup_longitude', 'dropoff_longitude')]
g=PartialDependenceDisplay.from_estimator(first_model, val_X, f_names, ax=ax)
g.figure_.savefig('result_2D_plot_longitude.png')



# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

# Create Partial Dependence Plots with new features
features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

feat_name = 'pickup_longitude'
g = PartialDependenceDisplay.from_estimator(second_model, new_val_X, [feat_name])
g.figure_.savefig('result_new_pickup_longitude.png')


#Creates two features, `X1` and `X2`, having random values in the range [-2, 2].
n_samples = 20000

# Create array holding predictive feature
X1 = 4 * rand(n_samples) - 2
X2 = 4 * rand(n_samples) - 2

# Modify the initialization of `y` so that our PDP plot has a positive slope in the range [-1,1],
y = -2 * X1 * (X1<-1) + X1 - 2 * X1 * (X1>1) - X2

# create dataframe 
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

predictors_df = my_df.drop(['y'], axis=1)

#Trains a `RandomForestRegressor` model to predict `y` given `X1` and `X2`.
my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)

#Creates a PDP plot for `X1` and a scatter plot of `X1` vs. `y`.
g = PartialDependenceDisplay.from_estimator(my_model, predictors_df, ['X1'])
g.figure_.savefig('result_PDP_X1.png')
