
import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load test dataset
test_data = pd.read_csv("dataset/test_data.csv")

X_test = test_data.drop("actual_scheme", axis=1)
y_test = test_data["actual_scheme"]

models = [
    "Logistic_Regression",
    "Perceptron",
    "Neural_Network",
    "Decision_Tree"
]

results = []

for name in models:

    with open(f"models/{name}.pkl", "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print("\n", name)
    print(classification_report(y_test, predictions, zero_division=0))

# Display comparison
comparison = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

comparison.to_csv("models/model_comparison.csv", index=False)
print("\nComparison saved successfully!")
