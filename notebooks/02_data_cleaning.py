import pandas as pd

# Load original dataset
df = pd.read_csv("data/diabetes_health_indicators.csv")

print("Original shape:", df.shape)

# Count duplicates
duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

# Remove exact duplicates
df = df.drop_duplicates().copy()

print("Shape after removing duplicates:", df.shape)

# Check missing values
print("Missing values:", df.isnull().sum().sum())

# Check target distribution
print("\nTarget distribution:")
print(df["Diabetes_binary"].value_counts())

# Save cleaned dataset
df.to_csv("data/diabetes_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")