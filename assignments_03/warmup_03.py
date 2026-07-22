import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Q1: shapes")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nQ2: mean of each column in X_train_scaled")
print(X_train_scaled.mean(axis=0))
# We fit the scaler on X_train only so that test-set statistics never leak into
# training (fitting on the full dataset would let information about the test
# set influence the scaling and make the evaluation overly optimistic).

# --- KNN ---

# Q1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_unscaled = knn.predict(X_test)
print("\nKNN Q1: accuracy (unscaled)")
print(accuracy_score(y_test, y_pred_unscaled))
print(classification_report(y_test, y_pred_unscaled))

# Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("\nKNN Q2: accuracy (scaled)")
print(accuracy_score(y_test, y_pred_scaled))
# Scaling makes little to no difference here (often the exact same accuracy),
# because the four Iris features are already measured in similar units (cm)
# and similar ranges, so no single feature dominates the distance calculation
# even without scaling.

# Q3
cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=5)
print("\nKNN Q3: 5-fold CV scores (unscaled, k=5)")
print("Fold scores:", cv_scores)
print("Mean:", cv_scores.mean())
print("Std:", cv_scores.std())
# This is more trustworthy than a single train/test split, because it averages
# performance over 5 different train/test partitions, so the result is less
# sensitive to which particular rows happened to land in the test set.

# Q4
print("\nKNN Q4: CV accuracy for different k")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
    print(f"k={k}, mean CV score={scores.mean():.4f}")
# k=5 (tied with k=7 in this run at the highest mean CV score) looks like a
# good choice: it balances flexibility against overfitting -- very small k
# can be noisy/overfit, while very large k can oversmooth the decision boundary.

# --- Classifier Evaluation ---

# Q1
cm = confusion_matrix(y_test, y_pred_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()
# Looking at the confusion matrix, the model most often confuses versicolor
# and virginica (setosa is always perfectly separated) -- these two species
# have overlapping petal/sepal measurements.

# --- The sklearn API: Decision Trees ---

# Q1
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("\nDecision Tree Q1: accuracy")
print(accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
# Compared to KNN, the Decision Tree accuracy is similar (often identical or very
# close on this small, easy dataset).
# Decision Trees split on raw feature thresholds rather than computing distances,
# so scaling would not affect the result at all -- the tree would produce the
# exact same splits and predictions on scaled or unscaled data.

# --- Logistic Regression and Regularization ---

# Q1
print("\nLogistic Regression Q1: coefficient magnitude vs C")
for C in [0.01, 1.0, 100]:
    logreg = OneVsRestClassifier(LogisticRegression(C=C, max_iter=1000, solver='liblinear'))
    logreg.fit(X_train_scaled, y_train)
    coef_sum = np.abs(logreg.estimators_[0].coef_).sum() + \
               np.abs(logreg.estimators_[1].coef_).sum() + \
               np.abs(logreg.estimators_[2].coef_).sum()
    print(f"C={C}: total |coef| = {coef_sum:.4f}")
# As C increases, the total coefficient magnitude increases. C is the inverse
# of the regularization strength, so a smaller C penalizes large coefficients
# more heavily (stronger regularization, simpler model), while a larger C
# allows the model to fit the training data more closely with larger weights.

# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1
print("\nPCA Q1: shapes")
print("X_digits:", X_digits.shape)
print("images:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")
plt.savefig("outputs/sample_digits.png")
plt.close()

# Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("outputs/pca_2d_projection.png")
plt.close()
# Same-digit images do tend to cluster together (e.g. 0s and 4s form fairly
# distinct groups), though some digits overlap heavily in just 2D (like 1/7/9
# or 3/5/8), since 2 components can't capture all 64 dimensions of variation.

# Q3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.axhline(y=0.8, color="r", linestyle="--")
plt.savefig("outputs/pca_variance_explained.png")
plt.close()
n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
print(f"\nPCA Q3: components needed for 80% variance = {n_components_80}")
# Roughly 10-13 components are needed to explain 80% of the variance
# (see the printed value above for the exact number).

# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]
n_digits_to_show = 5

fig, axes = plt.subplots(len(n_values) + 1, n_digits_to_show, figsize=(10, 12))

for col in range(n_digits_to_show):
    axes[0, col].imshow(images[col], cmap="gray_r")
    axes[0, col].axis("off")
axes[0, 0].set_ylabel("Original", rotation=0, labelpad=40)

for row, n in enumerate(n_values, start=1):
    for col in range(n_digits_to_show):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap="gray_r")
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"n={n}", rotation=0, labelpad=40)

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()
# Digits become clearly recognizable around n=15-40 components. This roughly
# matches where the cumulative variance curve starts leveling off (well past
# the 80% mark reached around 10-13 components) -- more components are needed
# for visual clarity than for capturing the bulk of the statistical variance.