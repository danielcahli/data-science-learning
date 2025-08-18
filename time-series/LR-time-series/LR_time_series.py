# This script demonstrates how to use linear regression models 
# to analyze and predict time series data.

# -------------------------
# Import required libraries
# -------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

# -------------------------
# Load and preprocess "book_sales" dataset
# -------------------------
# - Read book sales data with "Date" as the index (parsed as datetime).
# - Remove the "Paperback" column (we only analyze "Hardcover").

book_sales = pd.read_csv(
    'time-series/book_sales.csv',
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

# Add a time index (0, 1, 2, …) to represent the passage of time.
book_sales['Time'] = np.arange(len(book_sales.index))

# Create a lagged feature: previous day's Hardcover sales (shift by 1).
book_sales['Lag_1'] = book_sales['Hardcover'].shift(1)

# Reorder columns for clarity.
book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])

# -------------------------
# Load autoregressive dataset (two toy series)
# -------------------------
ar = pd.read_csv('time-series/ar.csv')

# -------------------------
# Load and preprocess "store_sales" dataset
# -------------------------
dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
# Load daily sales data
store_sales = pd.read_csv(
    'time-series/train.csv',
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
# 1. Linear regression on Hardcover sales
# -------------------------
# Fit regression line to visualize trend in Hardcover book sales
fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=book_sales, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=book_sales, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
fig.savefig("hardcover_sales.png", dpi=300, bbox_inches="tight")

# -------------------------
# 2. Plot autoregressive toy series
# -------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar['ar1'])
ax1.set_title('Series 1')
ax2.plot(ar['ar2'])
ax2.set_title('Series 2');
fig.savefig("Series.png", dpi=300, bbox_inches="tight")

# -------------------------
# 3. Linear regression with time as a feature (trend model)
# -------------------------
df = average_sales.to_frame()

# Create a "time" feature that increases by 1 each day
df['time'] = np.arange(len(df.index))

# Features (time) and target (sales)
X = df[['time']] 
y = df['sales']

# Fit regression model to predict sales over time
model = LinearRegression()
model.fit(X, y)

# Predicted sales values as a time series
y_pred = pd.Series(model.predict(X).flatten(), index=X.index)

# Plot actual vs predicted sales
fig, ax = plt.subplots(figsize=(10,6))
y.plot(ax=ax, alpha=0.5, linewidth=3, label="Actual")
y_pred.plot(ax=ax, linewidth=3, label="Predicted")
ax.set_title('Time Plot of Total Store Sales')
ax.legend()
fig.savefig("Store_Sales.png", dpi=300, bbox_inches="tight")

# -------------------------
# 4. Linear regression with lag feature (autoregression model)
# -------------------------
df = average_sales.to_frame()

# Create lagged sales (yesterday's sales as predictor)
lag_1 = df['sales'].shift(1)

df['lag_1'] = lag_1

X = df.loc[:, ['lag_1']]

X.dropna(inplace=True)  # drop missing values in the feature set

y = df.loc[:, 'sales']  # create the target

y, X = y.align(X, join='inner')  # drop corresponding values in target

# Fit regression model to predict today's sales from yesterday's sales
model = LinearRegression()
model.fit(X, y)

# Predicted sales
y_pred = pd.Series(model.predict(X), index=X.index)

# Plot actual vs predicted in lag space
fig, ax = plt.subplots()
ax.plot(X['lag_1'], y, '.', color='0.25')
ax.plot(X['lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='lag_1', title='Lag Plot of Average Sales');
fig.savefig("Average_Sales.png", dpi=300, bbox_inches="tight")
