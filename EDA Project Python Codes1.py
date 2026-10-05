# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 2. LOAD THE DATASET
# ============================================================

# Read Red Wine dataset
red = pd.read_csv("winequality-red.csv", sep=";")

# Read White Wine dataset
white = pd.read_csv("winequality-white.csv", sep=";")

# ============================================================
# 3. ADD WINE TYPE
# ============================================================

red["wine_type"] = "Red"
white["wine_type"] = "White"

# ============================================================
# 4. COMBINE BOTH DATASETS
# ============================================================

df = pd.concat([red, white], ignore_index=True)

# ============================================================
# 5. DISPLAY FIRST 5 RECORDS
# ============================================================

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

# ============================================================
# 6. DISPLAY LAST 5 RECORDS
# ============================================================

print("\n========== LAST 5 RECORDS ==========")
print(df.tail())

# ============================================================
# 7. DATASET SHAPE
# ============================================================

print("\n========== DATASET SHAPE ==========")
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# ============================================================
# 8. COLUMN NAMES
# ============================================================

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# ============================================================
# 9. DATA TYPES
# ============================================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# ============================================================
# 10. DATASET INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")
df.info()

# ============================================================
# 11. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ============================================================
# 12. CHECK DUPLICATE VALUES
# ============================================================

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())

# ============================================================
# 13. REMOVE DUPLICATE ROWS
# ============================================================

df = df.drop_duplicates()

print("\nRows after removing duplicates:", len(df))

# ============================================================
# 14. DESCRIPTIVE STATISTICS
# ============================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

# ============================================================
# 15. MEAN, MEDIAN, MINIMUM AND MAXIMUM QUALITY
# ============================================================

print("\n========== QUALITY STATISTICS ==========")

print("Mean Quality:", df["quality"].mean())
print("Median Quality:", df["quality"].median())
print("Minimum Quality:", df["quality"].min())
print("Maximum Quality:", df["quality"].max())

# ============================================================
# 16. MODE OF QUALITY
# ============================================================

print("\n========== MODE OF QUALITY ==========")

print("Mode Quality:")
print(df["quality"].mode())

# ============================================================
# 17. STANDARD DEVIATION
# ============================================================

print("\n========== STANDARD DEVIATION ==========")

print(df.select_dtypes("number").std())

# ============================================================
# 18. QUALITY COUNT
# ============================================================

print("\n========== QUALITY COUNT ==========")
print(df["quality"].value_counts().sort_index())

# ============================================================
# 19. WINE TYPE COUNT
# ============================================================

print("\n========== WINE TYPE COUNT ==========")

print(df["wine_type"].value_counts())

# ============================================================
# 20. UNIVARIATE ANALYSIS
#     QUALITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="quality")

plt.title("Distribution of Wine Quality")
plt.xlabel("Quality Score")
plt.ylabel("Number of Samples")

plt.show()

# ============================================================
# 21. ALCOHOL DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="alcohol",
    bins=30,
    kde=True
)

plt.title("Distribution of Alcohol")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")

plt.show()

# ============================================================
# 22. pH DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="pH",
    bins=30,
    kde=True
)

plt.title("Distribution of pH")
plt.xlabel("pH")
plt.ylabel("Frequency")

plt.show()

# ============================================================
# 23. FIXED ACIDITY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="fixed acidity",
    bins=30,
    kde=True
)

plt.title("Distribution of Fixed Acidity")
plt.xlabel("Fixed Acidity")
plt.ylabel("Frequency")

plt.show()

# ============================================================
# 24. RESIDUAL SUGAR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="residual sugar",
    bins=30,
    kde=True
)

plt.title("Distribution of Residual Sugar")
plt.xlabel("Residual Sugar")
plt.ylabel("Frequency")

plt.show()

# ============================================================
# 25. BIVARIATE ANALYSIS
#     ALCOHOL VS QUALITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="quality",
    y="alcohol"
)

plt.title("Alcohol vs Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Alcohol")

plt.show()

# ============================================================
# 26. VOLATILE ACIDITY VS QUALITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="quality",
    y="volatile acidity"
)

plt.title("Volatile Acidity vs Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Volatile Acidity")

plt.show()

# ============================================================
# 27. SULPHATES VS QUALITY
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="quality",
    y="sulphates"
)

plt.title("Sulphates vs Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Sulphates")

plt.show()

# ============================================================
# 28. DENSITY VS ALCOHOL
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="density"
)

plt.title("Alcohol vs Density")
plt.xlabel("Alcohol")
plt.ylabel("Density")

plt.show()

# ============================================================
# 29. ALCOHOL VS QUALITY SCATTER PLOT
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="quality"
)

plt.title("Alcohol vs Quality")
plt.xlabel("Alcohol")
plt.ylabel("Quality")

plt.show()

# ============================================================
# 30. GROUPING AND AGGREGATION
#     AVERAGE VALUES BY QUALITY
# ============================================================

print("\n========== AVERAGE VALUES BY QUALITY ==========")

quality_summary = df.groupby("quality").agg({
    "alcohol": "mean",
    "pH": "mean",
    "density": "mean",
    "volatile acidity": "mean",
    "residual sugar": "mean"
})

print(quality_summary)

# ============================================================
# 31. QUALITY-WISE COUNT
# ============================================================

print("\n========== QUALITY-WISE COUNT ==========")

quality_count = df.groupby("quality").size()

print(quality_count)

# ============================================================
# 32. WINE TYPE SUMMARY
# ============================================================

print("\n========== WINE TYPE SUMMARY ==========")

type_summary = df.groupby("wine_type").agg({
    "quality": ["count", "mean", "median", "std"],
    "alcohol": "mean",
    "pH": "mean",
    "density": "mean"
})

print(type_summary)

# ============================================================
# 33. QUALITY BY WINE TYPE
# ============================================================

print("\n========== QUALITY BY WINE TYPE ==========")

quality_type = pd.crosstab(
    df["quality"],
    df["wine_type"]
)

print(quality_type)

# ============================================================
# 34. WINE TYPE VS QUALITY VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="wine_type",
    y="quality"
)

plt.title("Quality Distribution by Wine Type")
plt.xlabel("Wine Type")
plt.ylabel("Quality")

plt.show()

# ============================================================
# 35. CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION MATRIX ==========")

correlation = df.select_dtypes("number").corr()

print(correlation)

# ============================================================
# 36. CORRELATION WITH QUALITY
# ============================================================

print("\n========== CORRELATION WITH QUALITY ==========")

quality_corr = correlation["quality"].sort_values(
    ascending=False
)

print(quality_corr)

# ============================================================
# 37. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================================
# 38. PAIRPLOT
# ============================================================

# Selected important variables
selected_columns = [
    "alcohol",
    "pH",
    "density",
    "volatile acidity",
    "sulphates",
    "quality"
]

sns.pairplot(
    df[selected_columns]
)

plt.show()

# ============================================================
# 39. TOP CORRELATIONS WITH QUALITY
# ============================================================

print("\n========== CORRELATION WITH QUALITY ==========")

for column in quality_corr.index:
    if column != "quality":
        print(
            column,
            ":",
            round(quality_corr[column], 3)
        )

# ============================================================
# 40. HIGHEST QUALITY RECORDS
# ============================================================

print("\n========== HIGHEST QUALITY RECORDS ==========")

highest_quality = df[
    df["quality"] == df["quality"].max()
]

print(highest_quality.head())

# ============================================================
# 41. LOWEST QUALITY RECORDS
# ============================================================

print("\n========== LOWEST QUALITY RECORDS ==========")

lowest_quality = df[
    df["quality"] == df["quality"].min()
]

print(lowest_quality.head())

# ============================================================
# 42. AVERAGE PHYSICOCHEMICAL VALUES
# ============================================================

print("\n========== AVERAGE PHYSICOCHEMICAL VALUES ==========")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    print(
        column,
        ":",
        round(df[column].mean(), 3)
    )

# ============================================================
# 43. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("             WINE QUALITY ANALYSIS")
print("=" * 60)

print("Total Records:", len(df))
print("Total Variables:", len(df.columns))
print("Average Quality:", round(df["quality"].mean(), 2))
print("Median Quality:", df["quality"].median())
print("Minimum Quality:", df["quality"].min())
print("Maximum Quality:", df["quality"].max())
print("Average Alcohol:", round(df["alcohol"].mean(), 2))
print("Average pH:", round(df["pH"].mean(), 2))

print("=" * 60)
print("                 ANALYSIS COMPLETED")
print("=" * 60)