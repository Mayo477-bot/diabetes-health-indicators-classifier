import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/diabetes_health_indicators.csv")

# Basic information
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# Target distribution
print("\n--- DIABETES DISTRIBUTION ---")
print(df["Diabetes_binary"].value_counts())

print("\n--- DIABETES PERCENTAGES ---")
print(df["Diabetes_binary"].value_counts(normalize=True) * 100)

# Visualize target distribution
sns.countplot(data=df, x="Diabetes_binary")
plt.title("Distribution of Diabetes Status")
plt.xlabel("Diabetes Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.savefig("reports/diabetes_distribution.png")
plt.show()