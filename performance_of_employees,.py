# data_analysis_pipeline.py
# This script reads a CSV, cleans the data, creates plots, and generates a PDF report with embedded plots.
# It was generated inside the assistant environment. Replace DATA_PATH with your CSV path when running locally.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = Path("C:\\Users\\Rahul\\Desktop\\internship project\\Test_data.csv")
OUTPUT_DIR = Path('C:\\Users\\Rahul\\Desktop\\internship project\\report_output')
PLOTS_DIR = Path ('C:\\Users\\Rahul\\Desktop\\internship project\\report_output\\plot')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("\n---- DATA INFO (df.info()) ----")
print(df.info())

# Shows first 5 rows of the dataset
print("\n---- FIRST 5 ROWS (df.head()) ----")
print(df.head())

# Optional: preview shape
print("\n✔ Dataset Shape:", df.shape)

# -------------------------------------------
# 4. Treat NULL Values (if any)
# -------------------------------------------

print("\n---- NULL VALUES BEFORE TREATMENT ----")
print(df.isnull().sum())

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill numeric NULLs with median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical NULLs with mode
for col in categorical_cols:
    if df[col].mode().size > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\n---- NULL VALUES AFTER TREATMENT ----")
print(df.isnull().sum())


# -------------------------------------------
# 5. Remove Duplicate Data
# -------------------------------------------

duplicates_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows found: {duplicates_count}")

# Drop duplicates
df.drop_duplicates(inplace=True)

print("Duplicates removed.")
print("New dataset shape:", df.shape)


# -------------------------------------------
# 6. Check Columns & Identify Useful Metrics
# -------------------------------------------

print("\n---- COLUMN DATA TYPES ----")
print(df.dtypes)

print("\n---- SUMMARY STATISTICS ----")
print(df.describe(include="all"))

# Identify potential modeling features
useful_metrics = {
    "numeric_features": list(numeric_cols),
    "categorical_features": list(categorical_cols),
    "possible_target_columns": [col for col in df.columns
                                if "target" in col.lower() or
                                   "label" in col.lower() or
                                   "y" == col.lower()]
}

print("\n---- USEFUL METRICS FOR MODELING ----")
for key, value in useful_metrics.items():
    print(f"{key}: {value}")

# -------------------------------------------
# 7. Visualizing Features Using Count Plots
# -------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

categorical_cols = df.select_dtypes(include=['object']).columns

print("\n---- GENERATING COUNT PLOTS ----")

for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col, palette="pastel")
    plt.title(f"Count Plot of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------------------
# 8. Visualizing Numeric Feature Distributions
# -------------------------------------------

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("\n---- GENERATING HISTOGRAMS ----")

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=20, kde=True, color="skyblue")
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.show()

# -------------------------------------------
# 9. Identify Trends & Insights From Plots
# -------------------------------------------

print("\n---- INSIGHTS FROM PLOTS ----")

insights = []

# Categorical insights
for col in categorical_cols:
    counts = df[col].value_counts()
    top = counts.idxmax()
    top_val = counts.max()
    pct = (top_val / len(df)) * 100

    insights.append(f"In '{col}', '{top}' is the most common category ({top_val} records, {pct:.1f}%).")

# Numeric insights
for col in numeric_cols:
    mean_val = df[col].mean()
    median_val = df[col].median()
    min_val = df[col].min()
    max_val = df[col].max()
    skew_val = df[col].skew()

    insights.append(
        f"'{col}' → Mean: {mean_val:.2f}, Median: {median_val:.2f}, Range: [{min_val}, {max_val}], Skewness: {skew_val:.2f}."
    )

# Print insights
for i in insights:
    print("•", i)
