# This script demonstrates how to combine the strengths of two forecasters


# -------------------------
# Imports
# -------------------------
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor

# -------------------------
# Load datasets
# -------------------------
comp_dir = Path('time-series/dataset')

# Store sales dataset: keep only needed columns and types
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
)
# Use a daily PeriodIndex to simplify time-based slicing/joining
store_sales['date'] = store_sales.date.dt.to_period('D')

# MultiIndex: (store_nbr, family, date)
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

# Daily mean by family, then select 2017 for analysis
family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)
# -------------------------
# Create a boosted hybrid for the Store Sales . 
# -------------------------

# Create the initial class definition
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None

# Define fit method for boosted hybrid
def fit(self, X_1, X_2, y):
    self.model_1.fit(X_1, y)

    y_fit = pd.DataFrame(
        self.model_1.predict(X_1),
        index=X_1.index, columns=y.columns,
    )

    y_resid = (y - y_fit).stack().squeeze()

    self.model_2.fit(X_2, y_resid)

    self.y_columns = y.columns
    self.y_fit = y_fit
    self.y_resid = y_resid

# Add method to class
BoostedHybrid.fit = fit

# Define predict method for boosted hybrid

def predict(self, X_1, X_2):
    # Predict with model_1
    y_pred = pd.DataFrame(
    self.model_1.predict(X_1), 
    index=X_1.index, columns=self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()  # wide to long

    # Add model_2 predictions to model_1 predictions
    y_pred += self.model_2.predict(X_2)

    return y_pred.unstack()

# Add method to class
BoostedHybrid.predict = predict

# -------------------------
# Set up the data for training.
# -------------------------

# Target series
y = family_sales.loc[:, 'sales']

# X_1: Features for Linear Regression
dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()

# X_2: Features for XGBoost
X_2 = family_sales.drop('sales', axis=1).stack()  # onpromotion feature

# Label encoding for 'family'
le = LabelEncoder()  # from sklearn.preprocessing
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])

# Label encoding for seasonality
X_2["day"] = X_2.index.day  # values are day of the month

# -------------------------
# Train boosted hybrid
# -------------------------

# Create the hybrid model by initializing a BoostedHybrid class with LinearRegression() and XGBRegressor() instances.
model = BoostedHybrid(
    model_1=LinearRegression(),
    model_2=XGBRegressor(),
)
model.fit(X_1, X_2, y)

y_pred = model.predict(X_1, X_2)
y_pred = y_pred.clip(0.0)

# See the predictions
y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
X1_train, X1_valid = X_1[: "2017-07-01"], X_1["2017-07-02" :]
X2_train, X2_valid = X_2.loc[:"2017-07-01"], X_2.loc["2017-07-02":]

model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

families = y.columns[0:6]
axs = y.loc(axis=1)[families].plot(
    subplots=True, sharex=True, figsize=(11, 9), alpha=0.5,
)
_ = y_fit.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C3', ax=axs)
for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family)
plt.savefig("prediction_hybrid.png", dpi=200, bbox_inches="tight")
plt.close()