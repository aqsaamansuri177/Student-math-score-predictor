"""
train_model.py
Trains a Linear Regression and a Random Forest on the Students Performance
dataset to predict math score. Compares both on RMSE and R², keeps the
better model, and serialises it together with metadata to disk.
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "StudentsPerformance.csv"
MODEL_PATH = "model.pkl"
META_PATH = "model_meta.json"

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# ── 2. Encode categoricals ────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
NUMERIC_FEATURES = ["reading score", "writing score"]
TARGET = "math score"

encoders: dict[str, LabelEncoder] = {}
df_enc = df.copy()
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_FEATURES
X = df_enc[FEATURE_COLS].values
y = df_enc[TARGET].values

# ── 3. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train both models ──────────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("── Model Comparison ──────────────────────────────")
print(f"  Linear Regression  │ RMSE: {lr_rmse:.3f}  │ R²: {lr_r2:.4f}")
print(f"  Random Forest      │ RMSE: {rf_rmse:.3f}  │ R²: {rf_r2:.4f}")

# ── 5. Pick the better model (lower RMSE) ────────────────────────────────────
if rf_rmse <= lr_rmse:
    best_model = rf
    best_name = "Random Forest"
    best_rmse = rf_rmse
    best_r2 = rf_r2
    importances = rf.feature_importances_.tolist()
else:
    best_model = lr
    best_name = "Linear Regression"
    best_rmse = lr_rmse
    best_r2 = lr_r2
    # Use abs(coef) normalised as a proxy for importance
    coefs = np.abs(lr.coef_)
    importances = (coefs / coefs.sum()).tolist()

print(f"\n  ✓  Keeping: {best_name}  (RMSE {best_rmse:.3f}, R² {best_r2:.4f})")

# ── 6. Persist ────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

# Build encoder label maps so the app can rebuild them without the .csv
label_maps = {col: le.classes_.tolist() for col, le in encoders.items()}

meta = {
    "model_name": best_name,
    "rmse": best_rmse,
    "r2": best_r2,
    "lr_rmse": lr_rmse,
    "lr_r2": lr_r2,
    "rf_rmse": rf_rmse,
    "rf_r2": rf_r2,
    "feature_cols": FEATURE_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "numeric_features": NUMERIC_FEATURES,
    "importances": importances,
    "label_maps": label_maps,
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"  ✓  Saved model  → {MODEL_PATH}")
print(f"  ✓  Saved meta   → {META_PATH}")
