from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Fetch the CDC Diabetes Health Indicators dataset
dataset = fetch_ucirepo(id=891)

# Combine features and target
df = pd.concat(
    [dataset.data.features, dataset.data.targets],
    axis=1
)

# Save dataset
output_path = Path("data/diabetes_health_indicators.csv")
df.to_csv(output_path, index=False)

print("Dataset downloaded successfully!")
print(f"Shape: {df.shape}")
print(f"Saved to: {output_path}")
print(df.head())