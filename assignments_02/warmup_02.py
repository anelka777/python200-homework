import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)

# ============================================================
# --- scikit-learn API ---
# ============================================================

# --- Q1 ---

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model_q1 = LinearRegression()          # create
model_q1.fit(years, salary)            # fit

pred_4 = model_q1.predict([[4]])[0]    # predict
pred_8 = model_q1.predict([[8]])[0]

print("--- Q1 ---")
print("Slope (coef_):", model_q1.coef_[0])
print("Intercept:", model_q1.intercept_)
print("Predicted salary for 4 years experience:", pred_4)
print("Predicted salary for 8 years experience:", pred_8)

# --- Q2 ---
x = np.array([10, 20, 30, 40, 50])
print("\n--- Q2 ---")
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped shape:", x_2d.shape)


# --- Q3 ---
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)   # create
kmeans.fit(X_clusters)                            # fit
labels = kmeans.predict(X_clusters)               # predict

print("\n--- Q3 ---")
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.figure(figsize=(6, 5))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", s=30)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="black", marker="X", s=200, label="Centers")
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

# ============================================================
# --- Linear Regression ---
# ============================================================

np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# --- LR Q1 ---
plt.figure(figsize=(6, 5))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.close()


# --- LR Q2 ---
X_age = age.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X_age, cost, test_size=0.2, random_state=42
)
print("\n--- LR Q2 ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --- LR Q3 ---
model_age = LinearRegression()
model_age.fit(X_train, y_train)

print("\n--- LR Q3 ---")
print("Slope:", model_age.coef_[0])
print("Intercept:", model_age.intercept_)

y_pred = model_age.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model_age.score(X_test, y_test)
print("RMSE:", rmse)
print("R^2:", r2)


# --- LR Q4 ---
X_full = np.column_stack([age, smoker])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)
r2_full = model_full.score(X_test_f, y_test_f)

print("\n--- LR Q4 ---")
print("R^2 (age only):", r2)
print("R^2 (age + smoker):", r2_full)
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])


# --- LR Q5 ---
y_pred_full = model_full.predict(X_test_f)

plt.figure(figsize=(6, 5))
plt.scatter(y_pred_full, y_test_f, alpha=0.7)
min_val = min(y_pred_full.min(), y_test_f.min())
max_val = max(y_pred_full.max(), y_test_f.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.legend()
plt.savefig("outputs/predicted_vs_actual_cost.png")
plt.close()
