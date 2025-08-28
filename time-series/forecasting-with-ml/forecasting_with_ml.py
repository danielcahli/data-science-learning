# Apply ML to any forecasting task.

# -------------------------
# Imports
# -------------------------
from pathlib import Path
from learntools.time_series.utils import (create_multistep_example,
                                          load_multistep_data,
                                          make_lags,
                                          make_multistep_target,
                                          plot_multistep)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
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
        'onpromotion': 'uint32',

    },
    parse_dates=['date'],
)
# Use a daily PeriodIndex to simplify time-based slicing/joining
store_sales['date'] = store_sales.date.dt.to_period('D')

# MultiIndex: (store_nbr, family, date)
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)
# Test dataset
test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
)
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['store_nbr', 'family', 'date']).sort_index()

#Visualize:
#a. 3-step forecast using 4 lag features with a 2-step lead time
#b. 1-step forecast using 3 lag features with a 1-step lead time
#c. 3-step forecast using 4 lag features with a 1-step lead time

datasets = load_multistep_data()

for i, styler in enumerate(datasets, start=1):
    df = styler.data   # extract the real DataFrame
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.table(cellText=df.values,
             colLabels=df.columns,
             rowLabels=df.index,
             cellLoc="center",
             loc="center")
    plt.savefig(f"dataset_{i}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

#Show training and test datasets
print("Training Data", "\n" + "-" * 13 + "\n", store_sales)
print("\n")
print("Test Data", "\n" + "-" * 9 + "\n", test)

#-----------------------------------------
# Create multistep dataset for Store Sales
#-----------------------------------------

y = family_sales.loc[:, 'sales']

X = make_lags(y, lags=4).dropna()

y = make_multistep_target(y, steps=16).dropna()

y, X = y.align(X, join='inner', axis=0)

# Apply the DirRec strategy to the multiple time series of Store Sales

# limit rows and columns for readability
N = 20   # first 20 rows
M = 10   # first 10 columns

df_preview = y.head(N).iloc[:, :M]

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

tbl = ax.table(
    cellText=df_preview.values,
    colLabels=df_preview.columns,
    rowLabels=df_preview.index,
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1.2, 1.2)

plt.tight_layout()
plt.savefig("y_preview.png", dpi=200, bbox_inches="tight")
plt.close()

#---------------------------------
#Forecast with the DirRec strategy
#---------------------------------

from lightgbm import LGBMRegressor

model = RegressorChain(
    base_estimator=LGBMRegressor(n_estimators=20, max_depth=3, verbose=-1)
)
model.fit(X, y)

y_pred = pd.DataFrame(
    model.predict(X),
    index=y.index,
    columns=y.columns,
).clip(0.0)

# 16-step predictions

FAMILY = 'BEAUTY'
START = '2017-04-01'
EVERY = 16


if hasattr(y_pred.index, 'levels'):
    
    y_pred_ = y_pred.xs(FAMILY, level='family', axis=0).loc[START:]
else:
    
    y_pred_ = y_pred.loc[START:]

y_ = family_sales.loc[START:, 'sales'].loc[:, FAMILY]

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax = y_.plot(ax=ax, alpha=0.5)
ax = plot_multistep(y_pred_, ax=ax, every=EVERY)
_ = ax.legend([FAMILY, FAMILY + ' Forecast'])
plt.tight_layout()
plt.savefig("forecast.png", dpi=200, bbox_inches="tight")
plt.close()