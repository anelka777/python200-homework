import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# --- ROC and AUC ---
# =====================================================================

# --- Q1 ---
# Logistic regression on raw (unscaled) data
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_probs_lr = lr.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_probs_lr)
print(f"Logistic Regression AUC (raw data): {auc_lr:.4f}")

# Scale data for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_probs_knn = knn.predict_proba(X_test_scaled)[:, 1]
auc_knn = roc_auc_score(y_test, y_probs_knn)
print(f"KNN AUC (scaled data): {auc_knn:.4f}")

# Comment: Whichever model has the higher AUC separates the two classes
# better across ALL thresholds, not just at the default 0.5 cutoff. AUC
# measures the probability that a randomly chosen positive example gets a
# higher predicted score than a randomly chosen negative example — so a
# higher AUC means the model's ranking of "how positive" each point looks
# is more reliable, independent of where you eventually decide to draw the
# decision boundary.

# --- Q2 ---
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_probs_knn)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {auc_knn:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison: Logistic Regression vs KNN")
ax.legend(loc="lower right")
fig.savefig("outputs/roc_comparison.png")
plt.close(fig)
print("Saved outputs/roc_comparison.png")


def fpr_at_tpr(fpr, tpr, target_tpr=0.80):
    """Return the fpr at the first point where tpr >= target_tpr."""
    idx = np.argmax(tpr >= target_tpr)
    return fpr[idx]


fpr_lr_80 = fpr_at_tpr(fpr_lr, tpr_lr)
fpr_knn_80 = fpr_at_tpr(fpr_knn, tpr_knn)
print(f"FPR at TPR=0.80 — Logistic Regression: {fpr_lr_80:.4f}")
print(f"FPR at TPR=0.80 — KNN: {fpr_knn_80:.4f}")

# Comment: at TPR = 0.80, whichever model has the lower FPR is catching the
# same 80% of true positives while raising fewer false alarms. Practically,
# that model would be the better choice for an application (e.g. fraud
# detection or medical screening) where you have a fixed recall target and
# want to minimize how often you incorrectly flag a negative case.

# --- Q3 ---
best_f1 = -1
best_threshold = None
best_tpr = None
best_fpr = None

for thresh in thresholds_lr:
    y_pred_thresh = (y_probs_lr >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
        idx = np.where(thresholds_lr == thresh)[0][0]
        best_tpr = tpr_lr[idx]
        best_fpr = fpr_lr[idx]

print(
    f"Best threshold: {best_threshold:.4f} | TPR: {best_tpr:.4f} | "
    f"FPR: {best_fpr:.4f} | F1: {best_f1:.4f}"
)

# Comment: the F1-optimal threshold is often not exactly 0.5 — it shifts
# depending on the class balance and how precision/recall trade off for
# this particular dataset. In a real application you would choose a
# threshold lower than 0.5 when missing a positive case (false negative) is
# much more costly than a false alarm (false positive) — for example, in
# disease screening, where failing to flag a sick patient is worse than an
# unnecessary follow-up test.

# =====================================================================
# --- GridSearchCV ---
# =====================================================================

# --- GridSearch Q1 ---
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

param_grid_lr = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="roc_auc")
grid_lr.fit(X_train, y_train)

best_C = grid_lr.best_params_["clf__C"]
best_cv_auc_lr = grid_lr.best_score_
test_auc_lr_grid = roc_auc_score(
    y_test, grid_lr.best_estimator_.predict_proba(X_test)[:, 1]
)

print(f"Best C: {best_C}")
print(f"Best CV AUC: {best_cv_auc_lr:.4f}")
print(f"Test AUC of best estimator: {test_auc_lr_grid:.4f}")

# Comment: compare best_C to the sklearn default of C=1.0 — the search may
# or may not land on 1.0. The test AUC difference vs. a pipeline fit with
# the default C shows how much tuning this single hyperparameter actually
# mattered for this dataset; often the gain is modest because logistic
# regression is already fairly robust to C on well-behaved synthetic data.

# --- GridSearch Q2 ---
pipe_tree = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(random_state=42)),
])

param_grid_tree = {"clf__max_depth": [2, 3, 5, 8, None]}

grid_tree = GridSearchCV(pipe_tree, param_grid_tree, cv=5, scoring="roc_auc")
grid_tree.fit(X_train, y_train)

best_depth = grid_tree.best_params_["clf__max_depth"]
best_cv_auc_tree = grid_tree.best_score_
test_auc_tree_grid = roc_auc_score(
    y_test, grid_tree.best_estimator_.predict_proba(X_test)[:, 1]
)

print(f"Best max_depth: {best_depth}")
print(f"Best CV AUC (tree): {best_cv_auc_tree:.4f}")
print(f"Test AUC (tree): {test_auc_tree_grid:.4f}")

# Comment: compare best_cv_auc_lr vs best_cv_auc_tree. Whichever is higher
# and more stable across folds would be the candidate to bring forward —
# but AUC alone doesn't tell the whole story. You'd also want to consider
# interpretability, training/inference speed, how well the model
# generalizes (a deep, unpruned tree can overfit even if CV AUC looks
# fine), and how the model behaves on the specific precision/recall
# trade-off your application cares about.

# --- GridSearch Q3 ---
cv_results = grid_lr.cv_results_
results_summary = sorted(
    zip(cv_results["params"], cv_results["mean_test_score"], cv_results["std_test_score"]),
    key=lambda item: item[1],
    reverse=True,
)

print("Logistic Regression grid search — CV AUC by parameter value:")
for params, mean_score, std_score in results_summary:
    print(f"  {params} -> mean AUC: {mean_score:.4f}, std: {std_score:.4f}")

# Comment: look through the printed list for two C values with similar mean
# scores but different standard deviations. Given the same mean, the value
# with the LOWER standard deviation is the safer pick — it means the model's
# performance is more consistent across the 5 folds, i.e. less sensitive to
# which subset of data it happened to be trained/validated on. A model that
# is occasionally great and occasionally mediocre is a riskier bet than one
# that is reliably decent.

# =====================================================================
# --- joblib ---
# =====================================================================

# --- joblib Q1 ---
best_lr_pipe = grid_lr.best_estimator_
joblib.dump(best_lr_pipe, "models/warmup_model.pkl")

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Comment: if only the LogisticRegression step were saved (without the
# StandardScaler) and then called directly on unscaled X_test, the model
# would still run without an error — but its predictions would be wrong,
# because the coefficients were learned on scaled feature ranges. Passing
# in raw, differently-scaled features silently produces garbage
# probabilities and a degraded decision boundary rather than a crash,
# which is exactly why saving the whole Pipeline (not just the estimator)
# matters.

# --- joblib Q2 ---
# --- Simulated prediction script ---
loaded_for_prediction = joblib.load("models/warmup_model.pkl")

new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

preds = loaded_for_prediction.predict(new_samples)
probs = loaded_for_prediction.predict_proba(new_samples)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    print(f"Sample {i}: predicted class = {pred}, P(class=1) = {prob:.4f}")

# Comment: the all-zeros row sits exactly at the mean of every feature
# (since make_classification generates roughly standardized features), so
# after scaling it lands near the origin of the decision boundary and gets
# no strong push from any single feature. In practice the prediction still
# lands somewhat above 0.5 (around 0.65 here) rather than dead center —
# that's the model's learned intercept term showing through: with all
# feature contributions at ~0, the intercept alone decides which class is
# "more likely by default," reflecting the baseline class balance the
# model saw during training.