# This script demonstrates how to use trends to make predictions 

# -------------------------
# Import required libraries
# -------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess

# -------------------------
# Load and preprocess "us-retail-sales" dataset
# -------------------------
# - Read retail sales data with "Month" as the index (parsed as datetime).
# Convert index to a PeriodIndex with daily frequency (good for time series)
retail_sales = pd.read_csv(
    "time-series/dataset/us-retail-sales.csv",
    parse_dates=['Month'],
    index_col='Month',
).to_period('D')

# Features FoodAndBeverage and Automobiles
food_sales = retail_sales.loc[:, 'FoodAndBeverage']
auto_sales = retail_sales.loc[:, 'Automobiles']

# -------------------------
# Load and preprocess "store_sales" dataset
# -------------------------
dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    'time-series/dataset/train.csv', # this dataset can be found on https://www.kaggle.com/code/ryanholbrook/trend
    dtype=dtype,
    parse_dates=['date'],
)
# Convert index to a PeriodIndex with daily frequency (good for time series)
store_sales = store_sales.set_index('date').to_period('D')

# Add store number and family as additional levels of a MultiIndex
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)

# Aggregate mean sales across all stores/families per day
average_sales = store_sales.groupby('date').mean()['sales']

# -------------------------
# 1. Moving Average on Food and Beverage Sales
# -------------------------
# Make a moving average 
trend = food_sales.rolling(window=12, center=True, min_periods=6).mean()

# Make a plot
ax = food_sales.plot(figsize=(10, 4), lw=2, label="Food and Beverage Sales")
trend.plot(ax=ax, lw=2, label="12-Month Rolling Average")
ax.set(title="US Food and Beverage Sales", ylabel="Millions of Dollars")
ax.legend()
plt.savefig("result_1.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# 2. Moving Average on Avarege Sales
# -------------------------
# Make a moving average plot of average_sales 
trend = average_sales.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()
# Make a plot
ax = average_sales.plot(figsize=(10, 4), lw=3, label="Avarege Sales",  alpha=0.5)
trend.plot(ax=ax, lw=3, label="Date")
ax.set(title="Avarege Sales", ylabel="Millions of Dollars")
ax.legend()
plt.savefig("result_2.png", dpi=200, bbox_inches="tight")
plt.close()


# -------------------------
# 3. Create a cubic trend forecast
# -------------------------
# Use `DeterministicProcess` to create a feature set for a cubic trend model
y = average_sales.copy()

dp = DeterministicProcess(index=y.index, order=3)

# create features for a 90-day forecast.
X = dp.in_sample()

X_fore = dp.out_of_sample(steps=90)

# Fit regression model to predict sales over time
model = LinearRegression()
model.fit(X, y)

# Predict sales values as a time series
y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

# Make a plot
ax = y.plot(figsize=(10, 4), alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend();

plt.savefig("result_3.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# 3. Create an order 11 trend forecast
# -------------------------
dp = DeterministicProcess(index=y.index, order=11)
X = dp.in_sample()

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

# Make a plot
ax = y.plot(figsize=(10, 4), alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax.legend();
plt.savefig("result_4.png", dpi=200, bbox_inches="tight")
plt.close()

# create features for a 90-day forecast.
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

# Make a plot
ax = y.plot(figsize=(10, 4), alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, lab

el="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend();
plt.savefig("result_5.png", dpi=200, bbox_inches="tight")
plt.close()

