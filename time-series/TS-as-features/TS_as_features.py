# This script demonstrates forecasting with time-based features:
# - Lag/lead features (lag embedding)
# - Seasonal components (dummy seasonality + Fourier terms)
# - Simple rolling/statistical features
# - A holdout (last 30 periods) for evaluation

# -------------------------
# Imports
# -------------------------
from pathlib import Path
from learntools.time_series.utils import plot_lags, make_lags, make_leads
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.model_selection import train_test_split


# -------------------------
# Plot style configuration
# -------------------------
def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rc("figure", figsize=(10, 5))
    plt.rc("axes", labelsize=12, titlesize=14)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("font", size=12)

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
        'onpromotion': 'uint32',
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
    .loc['2017', ['sales', 'onpromotion']]
)
# -------------------------
# Deseasonalize School & Office Supplies
# -------------------------
# Slice the "SCHOOL AND OFFICE SUPPLIES" family; y is the 'sales' Series
supply_sales = family_sales.loc(axis=1)[:, 'SCHOOL AND OFFICE SUPPLIES']
y = supply_sales.loc[:, 'sales'].squeeze()

# Deterministic time features:
# - constant + linear trend (order=1)
# - seasonal dummies (seasonal=True)
# - Fourier terms for smooth intra-year cyclesfourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True, # dummy seasonal indicators
    drop=True,   # drop a baseline category to avoid collinearity
    additional_terms=[fourier],  # sine/cosine pairs
)
X_time = dp.in_sample()
X_time['NewYearsDay'] = (X_time.index.dayofyear == 1)  # holiday effect

# Fit linear model without intercept (the DP already supplies a constant)
model = LinearRegression(fit_intercept=False)
model.fit(X_time, y)
# “Deseasonalized” = original minus fitted seasonal/trend/holiday component
y_deseason = y - model.predict(X_time)
y_deseason.name = 'sales_deseasoned'

ax = y_deseason.plot()
ax.set_title("Sales of School and Office Supplies (deseasonalized)");
plt.savefig("deseasonalized_sales.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Cycles (smoothing + diagnostics)
# -------------------------
# 7-day moving average to smooth short-term fluctuations
y_ma = y.rolling(7, center=True).mean()

ax = y_ma.plot()
ax.set_title("Seven-Day Moving Average");
plt.savefig("Moving_Average.png", dpi=200, bbox_inches="tight")
plt.close()

# Partial autocorrelation (PACF) of deseasonalized series (8 lags)
plot_pacf(y_deseason, lags=8);
plt.savefig("AC.png", dpi=200, bbox_inches="tight")
plt.close()
# Lag plots of deseasonalized series (visual lag relationships)
plot_lags(y_deseason, lags=8, nrows=2);
plt.savefig("Lags.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Promotions vs sales: leads/lags
# -------------------------
onpromotion = supply_sales.loc[:, 'onpromotion'].squeeze().rename('onpromotion')
# Visualize how leading/lagging promo relates to deseasonalized sales
# (filter to days with at least some promotions to reduce zeros)
plot_lags(x=onpromotion.loc[onpromotion > 1], y=y_deseason.loc[onpromotion > 1], lags=3, leads=3, nrows=1);
plt.savefig("leading_lagging.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Feature set for a simple regression
# -------------------------
# From deseasonalized sales: 1 lag
X_lags = make_lags(y_deseason, lags=1)

# From promotions: previous day (lag), same day, and next day (lead)
X_promo = pd.concat([
    make_lags(onpromotion, lags=1),
    onpromotion,
    make_leads(onpromotion, leads=1),
], axis=1)

# Combine deterministic time features + lagged y + promo features
X = pd.concat([X_time, X_lags, X_promo], axis=1).dropna()
y, X = y.align(X, join='inner')

# Hold out the last 30 periods (no shuffle) for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=30, shuffle=False)

# Fit and evaluate; RMSLE requires non-negative predictions -> clip at 0
model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
y_fit = pd.Series(model.predict(X_train), index=X_train.index).clip(0.0)
y_pred = pd.Series(model.predict(X_valid), index=X_valid.index).clip(0.0)

rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
rmsle_valid = mean_squared_log_error(y_valid, y_pred) ** 0.5
print(f'Training RMSLE: {rmsle_train:.5f}')
print(f'Validation RMSLE: {rmsle_valid:.5f}')

# Plot observed vs fitted and forecast (valid)
ax = y.plot(alpha=0.5, title="Average Sales", ylabel="items sold", figsize=(10, 5))
ax = y_fit.plot(ax=ax, label="Fitted", color='C0')
ax = y_pred.plot(ax=ax, label="Forecast", color='C3')
ax.legend();
plt.savefig("forecast.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Rolling/statistical features (exploratory)
# -------------------------
y_lag = supply_sales.loc[:, 'sales'].shift(1)
onpromo = supply_sales.loc[:, 'onpromotion']

mean_7 = y_lag.rolling(7).mean() # 7-day rolling mean
median_14 = y_lag.rolling(14).median()  # 14-day rolling median
std_7 = y_lag.rolling(7).std() # 7-day rolling std
promo_7 = onpromo.rolling(7, center=True).sum()  # 7-day centered promo sum

fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

y_lag.plot(ax=axes[0], lw=2, title="Lagged Sales (y_lag)")
axes[0].set_ylabel("Sales")

mean_7.plot(ax=axes[1], lw=2, color="C1", title="7-Day Rolling Mean")
axes[1].set_ylabel("Mean")

median_14.plot(ax=axes[2], lw=2, color="C2", title="14-Day Rolling Median")
axes[2].set_ylabel("Median")

std_7.plot(ax=axes[3], lw=2, color="C3", title="7-Day Rolling Std")
axes[3].set_ylabel("Std")

promo_7.plot(ax=axes[4], lw=2, color="C4", title="7-Day Centered Sum of Promotions")
axes[4].set_ylabel("Promos")

plt.tight_layout()
plt.savefig("rolling_features.png", dpi=200, bbox_inches="tight")
plt.close()