# This script demonstrates how to use indicator (dummy) variables and Fourier features 
# to capture periodic and holiday effects in a time series.

# -------------------------
# Import required libraries
# -------------------------
from pathlib import Path
from learntools.time_series.utils import plot_periodogram, seasonal_plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.preprocessing import OneHotEncoder

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

# Holidays dataset
holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
)
# Use PeriodIndex (daily frequency) for easier time-based filtering
holidays_events = holidays_events.set_index('date').to_period('D')

# Store sales dataset
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
)
# Convert date column to PeriodIndex
store_sales['date'] = store_sales.date.dt.to_period('D')
# Use a MultiIndex (store, family, date) for structured access
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
# -------------------------
# Aggregate and preprocess
# -------------------------
# Compute average daily sales across all stores/families in 2017
average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)
# Convert PeriodIndex → DatetimeIndex (needed for isocalendar/week extraction)
X = average_sales.to_timestamp().to_frame(name="sales")
X["week"] = X.index.isocalendar().week.astype(int) # ISO week number
X["day"]  = X.index.dayofweek.astype(int)   # Day of week (0=Mon, 6=Sun) 

# -------------------------
# Seasonal plot (week vs day)
# -------------------------
fig, ax = plt.subplots(figsize=(10, 4))   
seasonal_plot(X, y="sales", period="week", freq="day", ax=ax)
plt.tight_layout()
plt.savefig("seasonal_plot.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Periodogram (dominant frequencies)
# -------------------------
fig, ax = plt.subplots(figsize=(10, 4))   
plot_periodogram(average_sales, ax=ax)
plt.tight_layout()
plt.savefig("periodogram.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Deterministic process with Fourier terms
# -------------------------
y = average_sales.copy()

# Fourier terms capture smooth seasonal patterns
fourier = CalendarFourier(freq='ME', order=4)

# Build deterministic regressors: trend, seasonality, Fourier
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()

# -------------------------
# Fit seasonal model (trend + Fourier + dummies)
# -------------------------
model = LinearRegression().fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index, name="Fitted")

# Compare observed vs fitted values
ax = y.plot(figsize=(10, 4), lw=2, alpha=0.5, title="Average Sales", ylabel="Items Sold", label="Observed")
y_pred.plot(ax=ax, lw=2, label="Seasonal", color="C1")
ax.legend()
plt.savefig("avg_sales.png", dpi=200, bbox_inches="tight")
plt.close()


# -------------------------
# Check deseasonalization effect
# -------------------------
y_deseason = y - y_pred

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 4))
ax1 = plot_periodogram(y, ax=ax1)
ax1.set_title("Product Sales Frequency Components")
ax2 = plot_periodogram(y_deseason, ax=ax2);
ax2.set_title("Deseasonalized");
plt.savefig("Deseasonalized.png", dpi=200, bbox_inches="tight")
plt.close()

# -------------------------
# Holidays effect
# -------------------------
# Keep only national & regional holidays in training period

holidays = (
    holidays_events
    .query("locale in ['National', 'Regional']")
    .loc['2017':'2017-08-15', ['description']]
    .assign(description=lambda x: x.description.cat.remove_unused_categories())
)

print(holidays)

# Plot deseasonalized series with holidays highlighted
fig, ax = plt.subplots(figsize=(10, 4))   
y_deseason.plot(ax=ax)
plt.plot_date(holidays.index, y_deseason[holidays.index], color="C3")
ax.set_title("National and Regional Holidays")
plt.savefig("Holidays.png", dpi=200, bbox_inches="tight")
plt.close()


# -------------------------
# Add holidays as regressors
# -------------------------
# One-hot encode holiday descriptions

X_holidays = pd.get_dummies(holidays)

# Join holiday dummies to design matrix
X2 = X.join(X_holidays, on='date').fillna(0.0)

# Fit regression with holiday features
model = LinearRegression().fit(X2, y)

y_pred = pd.Series(  model.predict(X2), index=X2.index, name='Fitted',)

# Compare observed vs seasonal+holiday fit
fig, ax = plt.subplots(figsize=(10, 4))   # fix size
y.plot(ax=ax, alpha=0.5, title="Average Sales", ylabel="items sold")
y_pred.plot(ax=ax, label="Seasonal")
ax.legend()
plt.savefig("seasonal_with_holiday.png", dpi=200, bbox_inches="tight")
plt.close()
