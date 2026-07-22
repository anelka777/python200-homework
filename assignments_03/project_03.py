import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix

os.makedirs("outputs", exist_ok=True)

# ============================================================
# Task 1: Load and Explore
# ============================================================

# Loaded via ucimlrepo -- UCI's own officially documented import method for
# this dataset (see the "Import in Python" section on the dataset's page:
# https://archive.ics.uci.edu/dataset/94/spambase). Requires:
#   pip install ucimlrepo
spambase = fetch_ucirepo(id=94)
X = spambase.data.features
y = spambase.data.targets.iloc[:, 0]
y.name = "spam_label"

print("=" * 60)
print("TASK 1: Load and Explore")
print("=" * 60)

n_emails = len(X)
n_spam = int(y.sum())
n_ham = n_emails - n_spam
print(f"Total emails: {n_emails}")
print(f"Spam: {n_spam} ({n_spam / n_emails:.1%})")
print(f"Ham:  {n_ham} ({n_ham / n_emails:.1%})")
# The classes are imbalanced but not severely -- roughly 60/40 ham/spam.
# A raw accuracy score is still fairly informative here (a model predicting
# "always ham" would only get ~60%), but it's still worth looking at
# precision/recall per class rather than trusting accuracy alone, since a
# 60/40 split is unbalanced enough to inflate accuracy on the majority class.

# Boxplots: spam vs ham for 3 key features
features_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feature in features_to_plot:
    plt.figure(figsize=(6, 5))
    spam_vals = X.loc[y == 1, feature]
    ham_vals = X.loc[y == 0, feature]
    plt.boxplot([ham_vals, spam_vals], tick_labels=["Ham", "Spam"])
    plt.ylabel(feature)
    plt.title(f"{feature}: Spam vs Ham")
    safe_name = feature.replace("!", "exclamation").replace("$", "dollar")
    plt.savefig(f"outputs/boxplot_{safe_name}.png")
    plt.close()
# All three features show noticeably higher medians and wider spread for
# spam emails than ham -- spam emails use "free" more, use "!" more, and
# have longer runs of capital letters. The differences are fairly dramatic
# for word_freq_free and char_freq_!, though outliers stretch the boxplots
# in both classes.

# Broader look at scale
print("\nFeature value summary (first few word-frequency columns):")
print(X[features_to_plot].describe())
zero_fraction = (X == 0).mean().mean()
print(f"\nAverage fraction of zero values across all features: {zero_fraction:.2%}")
# Most word-frequency features are zero for most emails, because any single
# word (like "free") only appears in a minority of emails -- the data is
# naturally sparse. The numeric scale varies dramatically across features
# because word/char frequencies are percentages (small fractions, 0-100)
# while capital_run_length_total counts raw character runs and can reach
# into the thousands. This matters a lot for distance-based models like KNN
# and for PCA: without scaling, capital_run_length_total alone would
# dominate any distance calculation or variance-maximizing direction, simply
# because its raw numbers are so much bigger -- not because it's more
# informative.

# ============================================================
# Task 2: Prepare Your Data
# ============================================================

print("\n" + "=" * 60)
print("TASK 2: Prepare Your Data")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale: fit on training data only, to avoid leaking test-set statistics
# into the transformation.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA preprocessing: fit on the scaled training data only (same leakage
# reasoning as the scaler). PCA must run on SCALED data, or
# capital_run_length_total would dominate the variance purely because of
# its raw magnitude, not because it's actually the most informative feature.
pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = int(np.argmax(cumulative_variance >= 0.9) + 1)
print(f"Components needed to reach 90% variance: {n_components_90}")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(y=0.9, color="r", linestyle="--", label="90% variance")
plt.axvline(x=n_components_90, color="g", linestyle="--", label=f"n={n_components_90}")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Spambase: Cumulative Explained Variance vs. Number of Components")
plt.legend()
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

n = n_components_90
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# ============================================================
# Task 3: A Classifier Comparison
# ============================================================

print("\n" + "=" * 60)
print("TASK 3: A Classifier Comparison")
print("=" * 60)

results = {}  # model_name -> (accuracy, y_pred) for later use (best model / confusion matrix)

# --- KNN: unscaled ---
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
pred = knn_unscaled.predict(X_test)
acc = accuracy_score(y_test, pred)
results["KNN (unscaled)"] = (acc, pred)
print(f"\nKNN (unscaled) accuracy: {acc:.4f}")
print(classification_report(y_test, pred))

# --- KNN: scaled ---
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
pred = knn_scaled.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)
results["KNN (scaled)"] = (acc, pred)
print(f"\nKNN (scaled) accuracy: {acc:.4f}")
print(classification_report(y_test, pred))

# --- KNN: PCA-reduced ---
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
pred = knn_pca.predict(X_test_pca)
acc = accuracy_score(y_test, pred)
results["KNN (PCA)"] = (acc, pred)
print(f"\nKNN (PCA-reduced, n={n}) accuracy: {acc:.4f}")
print(classification_report(y_test, pred))
# KNN (scaled) beat KNN (unscaled) by a wide margin in my run (0.9077 vs
# 0.7991) -- exactly the scaling effect predicted in Task 1, since without
# scaling, capital_run_length_total dominates the distance calculation.
# KNN (PCA, n=43) scored 0.9066 -- essentially tied with, but very slightly
# below, the full-scaled version. So in this run PCA-reduced KNN did NOT
# outperform plain scaling; it kept 90% of the variance but still lost a
# hair of accuracy, likely because KNN distances are still sensitive to
# which specific variance gets dropped from the remaining ~10%.

# --- Decision Tree: depth comparison ---
print("\nDecision Tree: train vs test accuracy by max_depth")
depths = [3, 5, 10, None]
depth_results = {}
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    depth_results[depth] = (train_acc, test_acc)
    gap = train_acc - test_acc
    print(f"max_depth={depth}: train acc={train_acc:.4f}, test acc={test_acc:.4f}, gap={gap:.4f}")
# As depth increases, training accuracy climbs toward 1.0 (the tree
# memorizes the training data), while test accuracy keeps improving too but
# far more slowly -- the train/test gap (printed above for each depth)
# widens sharply with depth, which is the signature of overfitting even
# without test accuracy actually declining.

# Choose the depth with the best test accuracy among the depths that don't
# show a runaway train/test gap -- i.e. the best real generalization
# performance actually observed above, not an assumption.
best_finite_depth = max((d for d in depths if d is not None), key=lambda d: depth_results[d][1])
best_finite_test_acc = depth_results[best_finite_depth][1]
unlimited_test_acc = depth_results[None][1]
print(
    f"\nBest capped depth by test accuracy: {best_finite_depth} "
    f"(test acc={best_finite_test_acc:.4f}) vs unlimited depth "
    f"(test acc={unlimited_test_acc:.4f})"
)

chosen_depth = best_finite_depth
# Chosen depth is read directly from the comparison above: max_depth=10 gave
# the best test accuracy among the capped depths (3, 5, 10), and it was
# within a fraction of a point of the unlimited tree's test accuracy while
# keeping the train/test gap far smaller -- so it captures nearly all of
# the achievable performance without memorizing noise the way the
# unlimited-depth tree does.
tree = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)
acc = accuracy_score(y_test, pred)
results[f"Decision Tree (depth={chosen_depth})"] = (acc, pred)
print(f"\nDecision Tree (max_depth={chosen_depth}) accuracy: {acc:.4f}")
print(classification_report(y_test, pred))

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
acc = accuracy_score(y_test, pred)
results["Random Forest"] = (acc, pred)
print(f"\nRandom Forest accuracy: {acc:.4f}")
print(classification_report(y_test, pred))

# Feature importances: Decision Tree vs Random Forest
tree_importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 10 features -- Decision Tree:")
print(tree_importances.head(10))
print("\nTop 10 features -- Random Forest:")
print(rf_importances.head(10))
# The two models largely agree on the most important features (things like
# char_freq_$, char_freq_!, word_freq_remove, word_freq_free, and
# capital_run_length_total tend to show up near the top for both), which
# matches intuition -- dollar signs, exclamation points, and words like
# "remove" or "free" are classic spam signals. The Random Forest importances
# tend to be spread more evenly across features than the single Decision
# Tree (whose top feature alone can account for well over a third of total
# importance), since each tree in the forest only sees a random subset of
# features at each split.

plt.figure(figsize=(8, 6))
top10_rf = rf_importances.head(10)
plt.barh(top10_rf.index[::-1], top10_rf.values[::-1])
plt.xlabel("Feature importance")
plt.title("Random Forest: Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

# --- Logistic Regression: scaled ---
logreg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
logreg_scaled.fit(X_train_scaled, y_train)
pred = logreg_scaled.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)
results["Logistic Regression (scaled)"] = (acc, pred)
print(f"\nLogistic Regression (scaled) accuracy: {acc:.4f}")
print(classification_report(y_test, pred))

# --- Logistic Regression: PCA-reduced ---
logreg_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
logreg_pca.fit(X_train_pca, y_train)
pred = logreg_pca.predict(X_test_pca)
acc = accuracy_score(y_test, pred)
results["Logistic Regression (PCA)"] = (acc, pred)
print(f"\nLogistic Regression (PCA-reduced, n={n}) accuracy: {acc:.4f}")
print(classification_report(y_test, pred))
# Logistic Regression (scaled) scored 0.9294, and Logistic Regression
# (PCA, n=43) scored 0.9186 -- about 1 point lower. So, same as with KNN,
# PCA-reduced Logistic Regression did NOT outperform the plain scaled
# version in this run; it scored measurably lower. This complicates the
# Task 2 hypothesis that PCA would help magnitude-sensitive models: here,
# scaling alone already captures most of the benefit, and cutting to n
# components loses a bit of real signal rather than adding value -- PCA's
# actual payoff in this run is dimensionality/compute reduction (43
# components instead of 57), not a boost in accuracy.

print("\nSummary of test accuracies:")
for name, (acc, _) in results.items():
    print(f"  {name}: {acc:.4f}")
# Random Forest is typically the strongest performer here -- it captures
# non-linear interactions between features without needing scaling or PCA,
# and averaging across 100 trees reduces variance compared to a single
# Decision Tree.
#
# For a spam filter specifically, accuracy is NOT the metric I would
# optimize alone. A false positive (legitimate email marked as spam) can
# mean a person misses something important -- a job offer, a bill, a
# message from a client -- often without ever knowing it happened. A false
# negative (spam that gets through) is just an annoyance the user deletes
# in two seconds. Because the cost of the two error types is so
# asymmetric, I would rather minimize false positives, even if it means
# tolerating somewhat more spam getting through -- i.e. I'd prioritize
# precision on the "spam" class over recall.

best_model_name = max(results, key=lambda k: results[k][0])
best_acc, best_pred = results[best_model_name]
print(f"\nBest model by accuracy: {best_model_name} ({best_acc:.4f})")

cm_disp = ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, display_labels=["Ham", "Spam"]
)
plt.title(f"Confusion Matrix: {best_model_name}")
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

# Read the exact error counts directly from the confusion matrix itself,
# rather than estimating from precision/recall -- this is the ground truth
# for which error type the best model actually makes more often.
tn, fp, fn, tp = confusion_matrix(y_test, best_pred).ravel()
print(f"\nConfusion matrix for best model ({best_model_name}):")
print(f"  True Negatives (Ham correctly identified):  {tn}")
print(f"  False Positives (Ham marked as Spam):        {fp}")
print(f"  False Negatives (Spam that got through):     {fn}")
print(f"  True Positives (Spam correctly identified):  {tp}")
if fp < fn:
    print(f"  -> {best_model_name} makes MORE false negatives ({fn}) than false "
          f"positives ({fp}): it leans toward letting spam through rather than "
          f"blocking legitimate email -- the cheaper mistake given the priority "
          f"argued above.")
elif fp > fn:
    print(f"  -> {best_model_name} makes MORE false positives ({fp}) than false "
          f"negatives ({fn}): it leans toward blocking legitimate email more "
          f"often than letting spam through -- the costlier mistake given the "
          f"priority argued above, worth addressing (e.g. by adjusting the "
          f"classification threshold) before deploying it.")
else:
    print(f"  -> {best_model_name} makes an equal number of both error types "
          f"({fp} each).")

# ============================================================
# Task 4: Cross-Validation
# ============================================================

print("\n" + "=" * 60)
print("TASK 4: Cross-Validation")
print("=" * 60)

cv_models = {
    "KNN (unscaled)": Pipeline([
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "KNN (scaled)": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "KNN (PCA)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    f"Decision Tree (depth={chosen_depth})": Pipeline([
        ("classifier", DecisionTreeClassifier(max_depth=chosen_depth, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "Logistic Regression (scaled)": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ]),
    "Logistic Regression (PCA)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ]),
}

cv_summary = {}
for name, pipeline in cv_models.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    cv_summary[name] = (scores.mean(), scores.std())
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
# The most accurate model by mean CV score and the most stable (lowest std)
# model can be read directly off the printed values above. Random Forest
# has the highest mean CV accuracy by a clear margin, and it is noticeably
# more stable than the single Decision Tree (lower std), confirming that
# averaging across 100 trees smooths out fold-to-fold variance. However,
# Random Forest is NOT the single most stable model overall -- Logistic
# Regression (scaled) has an even lower std, so it is the more consistent
# performer across folds even though its mean accuracy trails Random
# Forest's. This generally matches the single train/test split ranking
# from Task 3, though the exact order of the middle-of-the-pack models
# (the KNN variants especially) can shift slightly since CV uses multiple
# splits instead of just one.

# ============================================================
# Task 5: Building a Prediction Pipeline
# ============================================================

print("\n" + "=" * 60)
print("TASK 5: Building a Prediction Pipeline")
print("=" * 60)

# Best tree-based model: Random Forest, no scaling needed (trees are scale-
# insensitive).
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
tree_pipeline.fit(X_train, y_train)
tree_pipeline_pred = tree_pipeline.predict(X_test)
print("\nTree-based pipeline (Random Forest) -- test set classification report:")
print(classification_report(y_test, tree_pipeline_pred))

# Best non-tree-based model: Logistic Regression. In Task 3, plain scaled
# Logistic Regression (0.9294) beat the PCA-reduced version (0.9186), so
# PCA is deliberately left OUT of this pipeline -- adding it would only
# reproduce the weaker result. (If Task 3 had shown PCA winning instead,
# a "pca" step would go here between "scaler" and "classifier", using
# PCA(n_components=n) as in the Task 2/4 setup.)
non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])
non_tree_pipeline.fit(X_train, y_train)
non_tree_pipeline_pred = non_tree_pipeline.predict(X_test)
print("\nNon-tree-based pipeline (Logistic Regression) -- test set classification report:")
print(classification_report(y_test, non_tree_pipeline_pred))

print("\nPipeline scores (pipeline.score == accuracy):")
print(f"  Tree pipeline:     {tree_pipeline.score(X_test, y_test):.4f}")
print(f"  Non-tree pipeline: {non_tree_pipeline.score(X_test, y_test):.4f}")
# These should match the manual Task 3 results for the same models.
#
# The two pipelines do NOT have the same structure: the tree-based pipeline
# skips scaling entirely (Random Forest splits on raw thresholds and
# doesn't care about feature magnitude), while the non-tree pipeline needs
# a scaler because Logistic Regression's coefficients and regularization
# are sensitive to feature scale. The practical value of packaging each
# model this way is that all preprocessing logic travels with the model --
# anyone (including future me) can call .fit()/.predict() on raw data
# without needing to remember the right sequence of manual steps, and
# there's no risk of accidentally scaling the test set with test-set
# statistics or forgetting a step when handing the model off or deploying
# it.

print("\nDone. All figures saved to outputs/")