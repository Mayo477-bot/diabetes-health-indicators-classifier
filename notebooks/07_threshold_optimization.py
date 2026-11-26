import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Create training and validation sets
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# Build model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Train on subtraining data
print("Training model...")
model.fit(X_subtrain, y_subtrain)

# Find best threshold on validation data
val_prob = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_prob
)

# Select the threshold with the highest precision
# while maintaining at least 85% recall
valid_indices = [
    i for i, r in enumerate(recall[:-1])
    if r >= 0.85
]

if not valid_indices:
    raise ValueError("No threshold achieved 85% validation recall.")

best_index = max(
    valid_indices,
    key=lambda i: precision[i]
)

best_threshold = thresholds[best_index]

print("\nSelected threshold:", best_threshold)
print("Validation precision:", precision[best_index])
print("Validation recall:", recall[best_index])

# Retrain on all training data
model.fit(X_train, y_train)

# Evaluate once on untouched test data
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n--- FINAL TEST PERFORMANCE ---")
print("\nClassification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["No diabetes", "Prediabetes/diabetes"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nROC-AUC:", roc_auc_score(y_test, test_prob))

# Save model and threshold
joblib.dump(
    {
        "model": model,
        "threshold": float(best_threshold)
    },
    "models/optimized_logistic_regression.pkl"
)

print("\nOptimized model saved successfully!")