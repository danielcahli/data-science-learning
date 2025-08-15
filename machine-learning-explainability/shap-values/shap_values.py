
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import eli5
from eli5.sklearn import PermutationImportance
import shap
import matplotlib.pyplot as plt

# ----------------- Load data -----------------
data = pd.read_csv('machine-learning-explainability/train.csv')
print("Columns:", list(data.columns))

# Target (handle common 'Yes'/'No' or 0/1 cases)
y_raw = data['readmitted']
if y_raw.dtype == object:
    y = (y_raw.astype(str).str.upper() == 'YES').astype(int)
else:
    y = y_raw.astype(int)

# Features: NUMERIC ONLY (avoid objects/categoricals)
X = data.drop(columns=['readmitted']).select_dtypes(include=[np.number])

# ----------------- Split & model -----------------
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1)
my_model.fit(train_X, train_y)

# ----------------- Permutation importance -----------------
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
text = eli5.format_as_text(eli5.explain_weights(perm, feature_names=val_X.columns.tolist()))
print(text)

# --- PDP (cast to float to avoid FutureWarning) ---
PartialDependenceDisplay.from_estimator(my_model, val_X.astype(float), ["number_inpatient"]).figure_.savefig('result_1.png')
PartialDependenceDisplay.from_estimator(my_model, val_X.astype(float), ["time_in_hospital"]).figure_.savefig('result_2.png')

# --- Readmission rate by length of stay ---
plt.clf()
pd.concat([train_X[['time_in_hospital']], train_y.rename('readmitted')], axis=1) \
  .groupby('time_in_hospital')['readmitted'].mean().plot()
plt.xlabel('Time in Hospital (days)')
plt.ylabel('Mean Readmission Rate')
plt.title('Readmission Rate by Length of Stay')
plt.grid(True)
plt.savefig('result_3.png'); plt.close()

# --- SHAP: single-row, multi-class safe plotting ---
import shap, numpy as np, matplotlib.pyplot as plt

sample_row = val_X.iloc[[0]].astype(float)      # 1-row DataFrame
explainer  = shap.Explainer(my_model)
sv         = explainer(sample_row)              # shap.Explanation

# Pick the positive class index (assumes target encoded as 0/1)
cls_idx = list(my_model.classes_).index(1) if hasattr(my_model, "classes_") else 0

# 1) Waterfall PNG: select the class output THEN the single sample
sv_class = sv[:, :, cls_idx]                    # shape => (n_samples, n_features) as Explanation
plt.figure(figsize=(8,6))
shap.plots.waterfall(sv_class[0], max_display=20)
plt.tight_layout(); plt.savefig("result_shap.png", dpi=150); plt.close()

# 2) Force plot HTML (interactive)
base = float(np.ravel(sv.base_values)[cls_idx]) if np.ndim(sv.base_values) else float(sv.base_values)
vals = sv.values[0, :, cls_idx]                 # 1D array of SHAP values for class 1
feat = sample_row.iloc[0]                       # feature values as Series
html = shap.force_plot(base_value=base, shap_values=vals, features=feat)
shap.save_html("result_shap.html", html)

print("Saved: result_1.png, result_2.png, result_3.png, result_shap.png, result_shap.html")
