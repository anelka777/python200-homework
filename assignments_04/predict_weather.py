import json

import joblib
import pandas as pd

# =====================================================================
# Task 1: Load and verify
# =====================================================================
model = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

print("=== Model metadata ===")
print(f"City: {metadata['city']}")
print(f"Features: {metadata['feature_names']}")
print(f"Test AUC: {metadata['test_auc']:.4f}")

# =====================================================================
# Task 2: Predict on new data
# =====================================================================
feature_names = metadata["feature_names"]

# Five hypothetical days: two clearly good, two clearly bad, one borderline
new_days = pd.DataFrame(
    [
        # temp_max, temp_min, precip, wind
        [18.0, 8.0, 0.0, 10.0],    # clearly good: mild, dry, calm
        [22.0, 12.0, 0.5, 15.0],   # clearly good: warm, dry, light wind
        [35.0, 20.0, 0.0, 12.0],   # clearly bad: too hot
        [10.0, -3.0, 8.0, 40.0],   # clearly bad: rainy, freezing min, windy
        [26.5, 1.0, 2.5, 28.5],    # borderline: right at the edges of the thresholds
    ],
    columns=feature_names,
)

preds = model.predict(new_days)
probs = model.predict_proba(new_days)[:, 1]

print("\n=== Predictions ===")
for i, row in new_days.iterrows():
    label = "good" if preds[i] == 1 else "skip"
    print(
        f"Day {i}: temp_max={row['temperature_2m_max']}, "
        f"temp_min={row['temperature_2m_min']}, "
        f"precip={row['precipitation_sum']}, "
        f"wind={row['wind_speed_10m_max']} "
        f"-> {label} (confidence: {probs[i]:.3f})"
    )

# =====================================================================
# Task 3: Reflect
# =====================================================================
# Comment:
# - The borderline case (Day 4 above) sits right at several threshold
#   edges at once (temp_max near 26, precip near the 3.0mm cutoff, wind
#   near the 30 km/h cutoff). I expected its predicted probability to land
#   close to 0.5, but it actually came out at 0.000 — a very confident
#   "skip." This shows that with a test AUC this high (0.9491), the model
#   learned a sharp decision boundary rather than a smooth one: even one
#   feature slightly over its threshold (temp_max=26.5) was enough to push
#   the prediction strongly toward "skip," rather than producing genuine
#   uncertainty. If I wanted to see real uncertainty, I'd need a day where
#   ALL four features sit only slightly past their thresholds at once,
#   instead of just one. In a real app, I'd still want a safety margin
#   around the threshold (e.g. don't auto-recommend above 0.9, flag
#   anything below that as "check yourself") since a rule-based label like
#   this can make the model overconfident near edge cases it hasn't
#   actually seen much of.
# - If someone ran predict_weather.py before train_weather_classifier.py,
#   joblib.load() would raise a FileNotFoundError because
#   models/weather_classifier.pkl wouldn't exist yet. The default error
#   message just says the file wasn't found, which doesn't tell the user
#   WHY. A more helpful version would wrap the load call in a try/except
#   and raise a clearer message, e.g.:
#     try:
#         model = joblib.load("models/weather_classifier.pkl")
#     except FileNotFoundError:
#         raise SystemExit(
#             "No trained model found. Run train_weather_classifier.py "
#             "first to create models/weather_classifier.pkl."
#         )
# - To support running daily on a real forecast, predict_weather.py would
#   need to: (1) call a weather FORECAST API (e.g. Open-Meteo's forecast
#   endpoint, not the historical archive endpoint) to get tomorrow's
#   predicted temp_max/temp_min/precipitation/wind instead of hand-typed
#   values, (2) build the same feature DataFrame from that response using
#   the exact column names/order stored in the metadata, and (3) probably
#   run on a schedule (e.g. a cron job or cloud scheduler) rather than
#   being invoked manually, writing its output somewhere persistent (a
#   file, database, or notification) instead of just printing to the
#   console.