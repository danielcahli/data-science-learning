import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



data = pd.read_csv('machine-learning-explainability/train.csv')
y = data.readmitted
base_features = ['number_inpatient', 'num_medications', 'number_diagnoses', 'num_lab_procedures', 
                 'num_procedures', 'time_in_hospital', 'number_outpatient', 'number_emergency', 
                 'gender_Female', 'payer_code_?', 'medical_specialty_?', 'diag_1_428', 'diag_1_414', 
                 'diabetesMed_Yes', 'A1Cresult_None']

# Some versions of our shap package error when mixing bools and numerics
X = data[base_features].astype(float)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# For speed, we will calculate shap values on smaller subset of the validation data

small_val_X = val_X.iloc[:150].astype(float)

my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)

# build explainer (pass background = train_X for better expectations)
explainer = shap.Explainer(my_model, train_X.astype(float))

sv = explainer(small_val_X)   # sv.values shape: (n_samples, n_features, n_outputs)
# pick the positive class (adjust if your positive label isn't 1)
cls_idx = list(my_model.classes_).index(1) if hasattr(my_model, "classes_") else 0

# slice the class/output dimension -> Explanation with shape (n_samples, n_features)
sv_class = sv[:, :, cls_idx]

# plot (new API)
shap.plots.beeswarm(sv_class, max_display=len(small_val_X.columns))
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot dependence (new API): one feature column only
shap.plots.scatter(
    sv_class[:, "num_medications"],   # SHAP values for that feature
    color=sv_class,                   # auto-choose correlated feature for color
    show=False
)

plt.gcf().set_size_inches(8, 6)
plt.savefig("dependence_plot.png", dpi=150, bbox_inches="tight")
plt.close()