import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance

data = pd.read_csv('machine-learning-explainability/train.csv', nrows=50000)

# Remove data with extreme outlier coordinates or negative fares
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
                 'dropoff_latitude',
                 'passenger_count']

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)

# show data
print("Data sample:")
print(data.head())

print(train_X.describe())

print(train_y.describe())

# Create a `PermutationImportance` object called `perm` to show the importances from `first_model`.  Fit it with the appropriate data and show the weights.

perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

text = eli5.format_as_text(
    eli5.explain_weights(perm, feature_names=val_X.columns.tolist())
)
print(text)

# creates new features for longitudinal and latitudinal distance

data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)

text = eli5.format_as_text(
    eli5.explain_weights(perm2, feature_names=new_val_X.columns.tolist())
)
print(text)

