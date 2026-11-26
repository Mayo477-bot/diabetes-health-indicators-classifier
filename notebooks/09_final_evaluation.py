import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Load original logistic regression
logistic = joblib.load("models/logistic_regression.pkl")

# Load random forest
forest = joblib.load("models/random_forest.pkl")

# Load optimized logistic regression
optimized = joblib.load(
    "models/optimized_logistic_regression.pkl"
)

# Define models and thresholds
model_configs = {
    "Logistic Regression": (logistic, 0.5),
    "Random Forest": (forest, 0.5),
    "Optimized Logistic Regression": (
        optimized["model"],
        optimized["threshold"]
    )
}

results = []

for name, (model, threshold) in model_configs.items():

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

# Create comparison table
report = pd.DataFrame(results)

print("\nFINAL MODEL EVALUATION")
print(report.round(4).to_string(index=False))

# Save report
report.to_csv(
    "reports/final_model_evaluation.csv",
    index=False
)

print("\nFinal evaluation report saved!")