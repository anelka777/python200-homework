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
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# ============================================================
# Task 1: Load and Explore
# ============================================================

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
knn_scaled_acc = results["KNN (scaled)"][0]
knn_pca_acc = results["KNN (PCA)"][0]
knn_unscaled_acc = results["KNN (unscaled)"][0]
print(
    f"\nKNN comparison -- unscaled: {knn_unscaled_acc:.4f}, "
    f"scaled: {knn_scaled_acc:.4f}, PCA(n={n}): {knn_pca_acc:.4f}"
)
# Read the three numbers printed directly above to see which KNN variant
# actually won on this run -- don't just assume. The expected pattern is:
# scaled KNN should beat unscaled KNN, for the same reason scaling mattered
# in Task 1 (without it, capital_run_length_total swamps the distance
# calculation), and PCA-reduced KNN should land close to scaled KNN,
# possibly a little behind, since keeping 90% of the variance still
# discards some signal. Whatever the printed accuracies actually show is
# the real conclusion to report here.

# --- Decision Tree: depth comparison ---
print("\nDecision Tree: train vs test accuracy by max_depth")
depths = [3, 5, 10, None]
for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"max_depth={depth}: train acc={train_acc:.4f}, test acc={test_acc:.4f}")
# As depth increases, training accuracy climbs toward 1.0 (the tree
# memorizes the training data), while test accuracy keeps improving too but
# far more slowly -- the gap between train and test accuracy widens sharply
# with depth (from roughly a 1-point gap at depth 3 to nearly a 9-point gap
# at unlimited depth), which is the signature of overfitting even without
# test accuracy actually declining.

chosen_depth = 10
# max_depth=10 is a reasonable production choice: it captures most of the
# tree's achievable test accuracy while keeping the train/test gap much
# smaller than the unlimited-depth tree, which memorizes noise instead of
# learning generalizable rules.
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
logreg_scaled_acc = results["Logistic Regression (scaled)"][0]
logreg_pca_acc = results["Logistic Regression (PCA)"][0]
print(
    f"\nLogistic Regression comparison -- scaled: {logreg_scaled_acc:.4f}, "
    f"PCA(n={n}): {logreg_pca_acc:.4f}"
)
logreg_pca_helped = logreg_pca_acc > logreg_scaled_acc
print(f"Did PCA help Logistic Regression on this run? {logreg_pca_helped}")
# The two numbers printed above (and the boolean right after them) are the
# actual evidence for this comparison -- report whichever one is true rather
# than assuming. If PCA did NOT help (the common outcome for these
# magnitude-sensitive models once they're already scaled), that confirms the
# Task 2 hypothesis needs qualifying: scaling alone captures most of the
# benefit, and cutting to n={n} components mostly buys compute/dimensionality
# reduction rather than an accuracy boost. If PCA DID help on this run, the
# non-tree pipeline in Task 5 should include it. Whichever branch the printed
# `logreg_pca_helped` value falls into determines which model counts as the
# best non-tree model below.

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
# Whichever error type appears more often in this confusion matrix (check
# the printed matrix / saved figure) determines whether the best model
# currently leans toward more false positives or more false negatives.
# In my run, Random Forest (the best model) produced roughly 17 false
# positives (ham marked as spam) and 33 false negatives (spam that got
# through) on the test set -- derived from recall=0.97 on Ham (558
# support) and recall=0.91 on Spam (363 support). That means it already
# leans toward the cheaper mistake (letting spam through) rather than the
# costlier one (blocking legitimate email) -- consistent with the priority
# argued above, though it's still worth checking whether adjusting the
# classification threshold could push false positives even lower.

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

# Best non-tree-based model: Logistic Regression. Only include PCA in the
# pipeline if Task 3 showed it helped -- otherwise the extra step just adds
# complexity without benefit.
if logreg_pca_helped:
    non_tree_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
    ])
else:
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