
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("dataset/farmer_data.csv")

X = data.drop("scheme", axis=1)
y = data["scheme"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

numerical = ["age", "land_area", "income"]
categorical = ["crop", "irrigation", "category"]

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Perceptron": Perceptron(max_iter=1000, random_state=42),
    "Neural_Network": MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=42
    ),
    "Decision_Tree": DecisionTreeClassifier(
        criterion="gini", random_state=42
    )
}

os.makedirs("models", exist_ok=True)

for name, algorithm in models.items():

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    with open(f"models/{name}.pkl", "wb") as file:
        pickle.dump(pipeline, file)

    print(name, "trained successfully")

# Save test data for evaluation
test_data = X_test.copy()
test_data["actual_scheme"] = y_test
test_data.to_csv("dataset/test_data.csv", index=False)

print("\nAll models trained successfully!")
print("Test data saved successfully!")
