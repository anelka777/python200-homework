# Pre-preprocessing observation:
# The CSV uses ';' as the separator, not ','. Values are not quoted.
# G1, G2, G3 are plain numeric values. So pd.read_csv() needs sep=";"
# in addition to the filename.

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("student_performance_math.csv", sep=";")

print(df.shape)
print(df.head())
print(df.dtypes)
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig("outputs/g3_distribution.png")
plt.close()

print("\n--- Task 2 ---")
print("Shape before filtering:", df.shape)

df_clean = df[df["G3"] != 0].copy()

print("Shape after filtering:", df_clean.shape)

# Comment: G3=0 does not represent an actual failing grade — it means the
# student was absent from the final exam and never received a real score.
# If we keep these rows, the model will try to explain G3=0 using features
# like absences, failures, or study time, even though these students might
# have had completely normal academic profiles. This adds noise that has
# nothing to do with real performance, and it weakens or distorts the
# relationships the model is supposed to learn between features and grades.

binary_yn_cols = ["schoolsup", "internet", "higher", "activities"]
for col in binary_yn_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

print(df_clean[["schoolsup", "internet", "higher", "activities", "sex"]].head())

from scipy.stats import pearsonr

corr_before, _ = pearsonr(df["absences"], df["G3"])
corr_after, _ = pearsonr(df_clean["absences"], df_clean["G3"])

print("\nCorrelation (absences vs G3) before filtering:", corr_before)
print("Correlation (absences vs G3) after filtering:", corr_after)

# Comment: In the original dataset, students with G3=0 didn't take the final
# exam at all, so their G3 value has nothing to do with how many classes
# they missed -- some had very high absences, others had very few, yet all
# of them show up as G3=0. Mixing these students in with everyone else adds
# noise that has no real relationship to absences, which weakens (dilutes)
# the true correlation between absences and grades. Once we filter out the
# G3=0 rows, we're only looking at students who actually completed the
# course, so the correlation reflects the real, direct relationship: more
# absences genuinely predicting a lower final grade.

print("\n--- Task 3 ---")

numeric_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]

correlations = {}
for col in numeric_cols:
    corr, _ = pearsonr(df_clean[col], df_clean["G3"])
    correlations[col] = corr

sorted_correlations = sorted(correlations.items(), key=lambda x: x[1])

for name, corr in sorted_correlations:
    print(f"{name:12s}: {corr:+.3f}")


# Comment: failures has the strongest (negative) relationship with G3 --
# students with more past class failures tend to get lower final grades,
# which makes intuitive sense. Surprisingly, schoolsup (extra school
# support) is negatively correlated with G3 -- but this likely reflects
# reverse causation: schools tend to assign extra support to students who
# are already struggling, not the other way around. Medu (mother's
# education) has a noticeably stronger positive relationship with G3 than
# Fedu (father's education), which is worth noting.


# --- Visualization 1: failures vs G3 ---
failures_groups = [df_clean[df_clean["failures"] == val]["G3"] for val in sorted(df_clean["failures"].unique())]

plt.figure()
plt.boxplot(failures_groups, tick_labels=sorted(df_clean["failures"].unique()))
plt.title("Final Grade (G3) by Number of Past Failures")
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/g3_by_failures.png")
plt.close()

# Comment: Students with 0 past failures have a noticeably higher median
# G3 and a wider spread toward higher grades, while students with 2 or 3
# past failures show a clearly lower median grade. This matches the
# negative correlation we computed earlier (-0.294) and confirms that
# failures is one of the most reliable warning signs of a low final grade.

# --- Visualization 2: Medu vs G3 ---
medu_groups = [df_clean[df_clean["Medu"] == val]["G3"] for val in sorted(df_clean["Medu"].unique())]

plt.figure()
plt.boxplot(medu_groups, tick_labels=sorted(df_clean["Medu"].unique()))
plt.title("Final Grade (G3) by Mother's Education Level")
plt.xlabel("Mother's Education (0=none, 4=higher education)")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/g3_by_medu.png")
plt.close()

# Comment: There is a mild upward trend in median G3 as Medu increases,
# consistent with the positive correlation (+0.190) computed earlier.
# Students whose mothers have higher education levels tend to score
# somewhat higher on average, though the effect is modest and there is
# still a lot of overlap between groups -- Medu alone is not a strong
# predictor on its own.

print("\n--- Task 4: Baseline Model ---")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

X_baseline = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)

model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)

y_pred_baseline = model_baseline.predict(X_test)

rmse_baseline = np.sqrt(np.mean((y_pred_baseline - y_test) ** 2))
r2_baseline = model_baseline.score(X_test, y_test)

print("Slope:", model_baseline.coef_[0])
print("RMSE:", rmse_baseline)
print("R^2:", r2_baseline)

# Comment: The slope (-1.43) means that each additional past failure is
# associated with a G3 grade about 1.4 points lower, on a 0-20 scale --
# a fairly noticeable drop for just one extra failure. RMSE of about 2.96
# means our typical prediction is off by roughly 3 points out of 20, which
# is a meaningful margin of error but not huge relative to the scale. R^2
# of 0.089 is low, meaning failures alone explains less than 10% of the
# variation in final grades. This roughly matches what we saw in the EDA --
# failures had the strongest single correlation with G3 (-0.294), but even
# the strongest single predictor leaves most of the variation unexplained,
# which is expected since student performance depends on many factors at
# once, not just past failures.


print("\n--- Task 5: Full Model ---")

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]

X_full = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

train_r2 = model_full.score(X_train_f, y_train_f)
test_r2 = model_full.score(X_test_f, y_test_f)

y_pred_full = model_full.predict(X_test_f)
rmse_full = np.sqrt(np.mean((y_pred_full - y_test_f) ** 2))

print("Train R^2:", train_r2)
print("Test R^2:", test_r2)
print("RMSE:", rmse_full)

print("\nCoefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Comment: Adding all 15 features nearly triples the explanatory power
# compared to the baseline (test R^2 went from 0.089 to 0.263). Train R^2
# (0.235) and test R^2 (0.263) are close to each other, which means the
# model is not overfitting -- it generalizes reasonably well to new data.
#
# The largest coefficient by far is schoolsup (-2.263), which is surprising
# at first, but it likely reflects reverse causation: schools tend to give
# extra support to students who are already struggling, not the other way
# around. internet (+1.037) has a notably large positive effect -- having
# internet access at home may give students more resources for studying.
# failures (-0.800) confirms what we saw in the baseline: past failures are
# a strong negative predictor. goout (-0.313) and Walc (-0.268) both show
# that more time socializing and drinking on weekends is associated with
# lower grades, which fits intuition. sex (+0.402) shows a modest advantage
# for male students in this dataset -- as noted in the feature guide, this
# reflects a social pattern specific to this educational context (research
# shows this gap varies by country), not an inherent ability difference.
#
# If deploying this model in production, I would keep failures, schoolsup,
# internet, goout, and Walc, since they have the largest, most consistent
# effects and align with domain intuition. I would consider dropping
# freetime (+0.014) and activities (+0.061), since their effects are close
# to zero and contribute little predictive value.


print("\n--- Task 6: Evaluate and Summarize ---")

plt.figure()
plt.scatter(y_pred_full, y_test_f, alpha=0.7)

min_val = min(y_pred_full.min(), y_test_f.min())
max_val = max(y_pred_full.max(), y_test_f.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.legend()
plt.savefig("outputs/predicted_vs_actual_g3.png")
plt.close()

# Comment: The errors are not uniform across the grade range -- the model
# struggles most at the extremes. Students with low actual grades (5-8)
# tend to fall below the diagonal, meaning the model overpredicts and
# gives them a higher grade than they actually earned (e.g. predicting
# ~10-12 for a student who actually scored 5-6). Students with high actual
# grades (16-19) tend to fall above the diagonal, meaning the model
# underpredicts them (e.g. predicting ~13 for a student who actually
# scored 18-19). This is a classic regression-to-the-mean pattern: the
# model pulls predictions toward the average grade instead of committing
# to extreme values. Predictions cluster much closer to the diagonal in
# the middle of the grade range (roughly G3 = 9-14), where the model
# performs best.

print("\n--- Neglected Feature: The Power of G1 ---")

feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print("Test R^2 with G1 added:", test_r2_g1)

# Comment: Adding G1 causes test R^2 to jump from 0.263 to 0.765 -- a huge
# increase. A high R^2 here does NOT mean G1 causes G3; it means G1 is a
# very strong indicator of a student's overall trajectory in the course,
# since first-period performance tends to be highly consistent with final
# performance (a student doing well in G1 usually keeps doing well, and
# vice versa). This makes the model much less useful for identifying
# at-risk students early -- G1 already comes from partway through the
# course, so by the time we have it, it's often too late to intervene
# meaningfully. If educators want to identify struggling students before
# G1 is even available, they need to rely on the background/behavioral
# model from Task 5 (failures, schoolsup, absences, goout, etc.), even
# though it's less accurate, because it's the only one that works before
# any grades exist.



# --- Summary ---
# Comment:
# Dataset size: after removing G3=0 rows (students who didn't take the
# final exam), the filtered dataset has 357 students, split into roughly
# 285 training rows and 72 test rows (80/20 split).
#
# Best model performance (full model without G1): RMSE ~2.66 and test R^2
# ~0.263. In plain language, a typical prediction is off by about 2.7
# points on a 0-20 grading scale, and the model explains about 26% of the
# variation in final grades using only background and behavioral features
# -- meaningful, but far from a complete picture of what determines a
# student's final grade.
#
# Largest negative coefficient: schoolsup (-2.263) -- students receiving
# extra school support tend to have notably lower grades, most likely
# because support is given to students who are already struggling, not
# because the support itself hurts performance.
# Largest positive coefficient: internet (+1.037) -- having internet
# access at home is associated with a meaningfully higher final grade,
# possibly reflecting access to study resources.
#
# Most surprising result: the negative coefficient on schoolsup. It would
# be easy to assume that extra academic support should raise grades, but
# the data suggests the opposite sign -- a good reminder that correlation
# (and even regression coefficients) can reflect who receives an
# intervention rather than the effect of the intervention itself.