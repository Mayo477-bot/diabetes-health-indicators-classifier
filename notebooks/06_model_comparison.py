import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models = ["Logistic Regression", "Random Forest"]

metrics = {
    "Accuracy": [0.7143, 0.8253],
    "Precision": [0.32, 0.43],
    "Recall": [0.76, 0.41],
    "F1-score": [0.45, 0.42],
    "ROC-AUC": [0.8106, 0.8012]
}

df = pd.DataFrame(metrics, index=models)

print("\nMODEL COMPARISON")
print(df.round(4))

# Plot comparison
ax = df.T.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.75
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Evaluation Metrics")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Models")
plt.tight_layout()

plt.savefig("reports/model_comparison.png")
plt.show()

print("\nComparison chart saved successfully!")