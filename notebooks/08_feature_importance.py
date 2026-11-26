import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import joblib

# Load model and test data
model = joblib.load("models/random_forest.pkl")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Use a sample to reduce computation time
X_sample = X_test.sample(
    n=5000,
    random_state=42
)
y_sample = y_test.loc[X_sample.index]

# Calculate permutation importance
print("Calculating feature importance...")

results = permutation_importance(
    model,
    X_sample,
    y_sample,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

# Organize results
importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Importance": results.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_df.head(10).to_string(index=False))

# Visualize
top_features = importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Features - Random Forest")
plt.xlabel("Mean decrease in ROC-AUC after permutation")
plt.tight_layout()

plt.savefig("reports/feature_importance.png")
plt.show()

# Save results
importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("\nFeature importance analysis completed!")