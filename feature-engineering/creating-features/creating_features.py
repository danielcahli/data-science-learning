import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from pathlib import Path

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


# Prepare data
df = pd.read_csv(Path("feature-engineering/creating-features/dataset.csv"))
X = df.copy()
y = X.pop("SalePrice")

# Create Mathematical Transforms in a new dataframe:
X_1 = pd.DataFrame()
X_1["LivLotRatio"] = X.GrLivArea / X.LotArea
X_1["Spaciousness"] = (X.FirstFlrSF + X.SecondFlrSF) / X.TotRmsAbvGrd
X_1["TotalOutsideSF"] = X.WoodDeckSF + X.OpenPorchSF + X.EnclosedPorch + X.Threeseasonporch + X.ScreenPorch

# Was discovered an interaction between `BldgType` and `GrLivArea` in Exercise 2. 
# Was created their interaction features.
# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")

# Multiply
X_2 = X_2.mul(df.GrLivArea, axis=0)

#Create a feature PorchTypes that counts how many of the following are greater than 0.0:
X_3 = pd.DataFrame()

X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)

#`MSSubClass` describes the type of a dwelling:
df.MSSubClass.unique()

X_4 = pd.DataFrame()

#Create a feature splitting MSSubClass at the first underscore _ and selecting the first word
X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]

print(X_4)
#A feature `MedNhbdArea` was created to describe the *median* of `GrLivArea` grouped on `Neighborhood`.
X_5 = pd.DataFrame()

X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")

#Score the model 
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)
