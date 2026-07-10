import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


import sys
print(sys.executable)

# --- Pandas ---

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

# Pandas Q1

print("First three rows:")
print(df.head(3))

print("DataFrame shape:")
print(df.shape)

print("Data types:")
print(df.dtypes)

# Pandas Q2

print("Students who passed and have grade above 80:")
filtered_df = df[(df["passed"]) & (df["grade"] > 80)]
print(filtered_df)

# Pandas Q3

print("DataFrame with curved grades:")
df["grade_curved"] = df["grade"] + 5
print(df)

# Pandas Q4

df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# Pandas Q5

print("Average grade by city:")
city_grades = df.groupby("city")["grade"].mean()
print(city_grades)

# Pandas Q6

print("Name and city after replacing Austin with Houston:")
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# Pandas 07

sorted_df = df.sort_values("grade", ascending=False)
print(sorted_df.head(3))

# --- NumPy ---

# NumPy Q1

arr = np.array([10, 20, 30, 40, 50])

print("Shape:")
print(arr.shape)

print("Dtype:")
print(arr.dtype)

print("Number of dimensions:")
print(arr.ndim)

# NumPy Q2

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])


print("Shape:")
print(arr.shape)

print("Total number of elements(size):")
print(arr.size)

# NumPy Q3

print("Slice out the top-left 2x2 block:")
print(arr[0:2, 0:2])

# NumPy Q4

zeros = np.zeros((3, 4))
ones = np.ones((2, 5))

print("Create 3x4 aray of zeros:")
print(zeros)
print("Create 2x5 array of ones:")
print(ones)

# NumPy Q5

arr = np.arange(0, 50, 5)
print("The array:")
print(arr)

print("Shape:")
print(arr.shape)

print("Mean:")
print(arr.mean())

print("Sum:")
print(arr.sum())

print("Standard deviation:")
print(arr.std())

# NumPy Q6

arr = np.random.normal(0, 1, 200)

print("Mean:")
print(arr.mean())

print("Standard:")
print(arr.std())

# --- Matplotlib Review ---

# Matplotlib Q1

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)

plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Matplotlib Q2

subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()


# Matplotlib Q3

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="blue", label="Dataset 1")
plt.scatter(x2, y2, color="red", label="Dataset 2")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Matplotlib Q4

fig, ax = plt.subplots(1, 2)

ax[0].plot(x, y)
ax[1].bar(subjects, scores)

ax[0].set_title("Squares")
ax[1].set_title("Subject Scores")

ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

ax[1].set_xlabel("Subjects")
ax[1].set_ylabel("Scores")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")

plt.tight_layout()
plt.show()


# --- Descriptive Statistics ---

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print("Mean:")
print(np.mean(data))

print("Median:")
print(np.median(data))

print("Variance:")
print(np.var(data))

print("Standard deviation:")
print(np.std(data))


# Descriptive Stats Q2

# np.random.normal(mean, std, size)
scores = np.random.normal(65, 10, 500)

plt.hist(scores, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.show()


# Descriptive Stats Q3

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot(
    [group_a, group_b],
    tick_labels=["Group A", "Group B"]
)

plt.title("Score Comparison")
plt.show()


# Descriptive Stats Q4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot(
    [normal_data, skewed_data],
    tick_labels=["Normal", "Exponential"]
)

plt.title("Distribution Comparison")
plt.show()

# The Exponential distribution is more skewed.
# Mean is appropriate for the Normal distribution, while median is better for the Exponential distribution because of skewness.


# Descriptive Stats Q5

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print("Data1 mean:")
print(np.mean(data1))

print("Data1 median:")
print(np.median(data1))

print("Data1 mode:")
print(stats.mode(data1).mode)

print("Data2 mean:")
print(np.mean(data2))

print("Data2 median:")
print(np.median(data2))

print("Data2 mode:")
print(stats.mode(data2).mode)

# The mean and median are very different for data2 because 150 is an outlier.
# The large value pulls the mean upward, while the median is not affected as much.


# --- Hypothesis ---

# Hypothesis Q1

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

result = stats.ttest_ind(group_a, group_b)

print("t-statistic:")
print(result.statistic)

print("p-value:")
print(result.pvalue)

# Hypothesis Q2

# Check statistical significance using alpha = 0.05
if result.pvalue < 0.05:
    print("The result is statistically significant.")
else:
    print("The result is not statistically significant.")


# Hypothesis Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]
# Paired t-test: compare scores before and after for the same students
result = stats.ttest_rel(before, after)

print("Paired t-test results:")
print("t-statistic:", result.statistic)
print("p-value:", result.pvalue)

# Hypothesis Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]
# One-sample t-test: compare sample mean to the benchmark value of 70
result = stats.ttest_1samp(scores, 70)

print("One-sample t-test results:")
print("t-statistic:", result.statistic)
print("p-value:", result.pvalue)


# Hypothesis Q5

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
# One-tailed independent t-test:
# Check whether Group A scores are less than Group B scores
result = stats.ttest_ind(
    group_a,
    group_b,
    alternative="less"
)
print("One-tailed t-test p-value:")
print(result.pvalue)


# Hypothesis Q6

print(
    "Group B scores were significantly higher than Group A scores, "
    "and this difference is unlikely to be due to chance."
)


# --- Correlation Review ---

# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)

print("Correlation matrix:")
print(corr_matrix)

print("Pearson correlation coefficient:")
print(corr_matrix[0,1])

# I expect the correlation to be 1 because y increases proportionally with x,
# showing a perfect positive linear relationship.


# Correlation Q2

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

correlation, p_value = pearsonr(x, y)

print("Pearson correlation coefficient:")
print(correlation)

print("p-value:")
print(p_value)


# Correlation Q3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

correlation_matrix = df.corr()

print("Correlation matrix:")
print(correlation_matrix)


# Correlation Q4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Q5

sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()



# --- Pipelines ---

# Pipeline Q1

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
def data_pipeline(arr):
    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)

    return summary

result = data_pipeline(arr)
print("Pipeline summary:")
for key, value in result.items():
    print(f"{key}: {value}")