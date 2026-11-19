
import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
data = pd.read_csv("models/model_comparison.csv")

# Select metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Plot graph
data.set_index("Algorithm")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    width=0.8
)

plt.title("ML Algorithm Performance Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Performance (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.legend(title="Evaluation Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save graph
plt.savefig("models/model_comparison.png", dpi=300)

plt.show()

print("Comparison graph saved successfully!")
