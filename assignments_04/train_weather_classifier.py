import os
import json
import sys
import platform

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, classification_report
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================================================================
# Step 1: Fetch the data
# =====================================================================
# City: Sacramento, CA (my area) — adjust lat/lon for your own city.
LATITUDE = 38.58
LONGITUDE = -121.49
CITY_NAME = "Sacramento, CA"

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("=== Dataset summary ===")
print(df.describe())
print(f"\nTotal days: {len(df)}")
print(df.head())

# =====================================================================
# Step 2: Engineer labels
# =====================================================================
# "Good for running" thresholds — starting point from the lesson, used
# as-is here since Sacramento's climate (hot dry summers, mild wet winters)
# is reasonably well represented by these ranges. Documented so the
# reasoning is clear:
#   - temp max between 7-26 C: avoids both freezing mornings and the
#     harsh Central Valley summer heat (which regularly exceeds 35C)
#   - temp min >= 0 C: avoids icy/frost conditions
#   - precipitation < 3.0 mm: excludes days with meaningful rain
#   - wind speed < 30 km/h: excludes gusty, uncomfortable days
FEATURE_NAMES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

df["good_for_running"] = (
        (df["temperature_2m_max"] >= 7) & (df["temperature_2m_max"] <= 26) &
        (df["temperature_2m_min"] >= 0) &
        (df["precipitation_sum"] < 3.0) &
        (df["wind_speed_10m_max"] < 30)
).astype(int)

class_counts = df["good_for_running"].value_counts()
class_fraction = df["good_for_running"].mean()
print("\n=== Class distribution ===")
print(class_counts)
print(f"Fraction labeled 'good for running': {class_fraction:.2%}")

# Comment: the printed fraction tells us how balanced (or imbalanced) the
# label is. For a Central Valley city like Sacramento, a large share of
# summer days get excluded purely on the temperature-max threshold (too
# hot), while winter days often get excluded on precipitation or the
# min-temp threshold — so the "good for running" fraction is expected to
# land well under 50%, concentrated in spring and fall. If this comes out
# above ~70% or under ~10%, that's a signal the thresholds may need
# adjusting for the chosen city's climate.

# =====================================================================
# Step 3: Train and tune
# =====================================================================
X = df[FEATURE_NAMES]  # keep as a DataFrame so feature names are preserved
y = df["good_for_running"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)

best_C = grid.best_params_["clf__C"]
best_cv_auc = grid.best_score_
best_pipe = grid.best_estimator_

y_pred = best_pipe.predict(X_test)
y_probs = best_pipe.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs)

print(f"\nBest C: {best_C}")
print(f"Best CV AUC: {best_cv_auc:.4f}")
print("\n=== Classification report (test set) ===")
print(classification_report(y_test, y_pred))
print(f"Test AUC: {test_auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_probs)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {test_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"Weather Classifier ROC Curve — {CITY_NAME}")
ax.legend(loc="lower right")
fig.savefig("outputs/weather_roc.png")
plt.close(fig)
print("Saved outputs/weather_roc.png")

# =====================================================================
# Step 4: Reflect on evaluation
# =====================================================================
# Comment:
# - Test AUC came out to 0.9491, which is very high. This isn't too
#   surprising: the "good for running" label was built directly from the
#   same four features the model trains on using hard thresholds, so this
#   is closer to a learnable rule than a noisy real-world label.
# - Precision and recall are nearly balanced (0.93/0.93 for class 0,
#   0.90/0.90 for class 1) — the model isn't strongly biased toward
#   false positives or false negatives here.
# - Given that near-balance, I'd keep the threshold close to 0.5, maybe
#   nudging it slightly higher (~0.55-0.6) to favor precision over recall
#   for a running app — I'd rather it miss a borderline good day than
#   send someone out in bad weather.

# =====================================================================
# Step 5: Save the model
# =====================================================================
joblib.dump(best_pipe, "models/weather_classifier.pkl")

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "feature_names": FEATURE_NAMES,
    "best_hyperparameters": grid.best_params_,
    "test_auc": test_auc,
    "city": CITY_NAME,
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "label_thresholds": {
        "temperature_2m_max": "7 to 26 C",
        "temperature_2m_min": ">= 0 C",
        "precipitation_sum": "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nModel saved to models/weather_classifier.pkl")
print("Metadata saved to models/weather_classifier_metadata.json")