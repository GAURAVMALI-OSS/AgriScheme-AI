import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load trained Decision Tree model
with open("models/Decision_Tree.pkl", "rb") as file:
    model = pickle.load(file)

# Extract preprocessing and classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Clean feature names
feature_names = [
    name.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
    for name in feature_names
]

# Get scheme class names
class_names = [str(name) for name in classifier.classes_]

# Plot actual CART Decision Tree
plt.figure(figsize=(30, 18))

plot_tree(
    classifier,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    fontsize=8
)

plt.title(
    "AgriScheme AI - CART Decision Tree (Gini Impurity)",
    fontsize=20
)

plt.savefig(
    "models/decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Decision Tree visualization saved successfully!")
print("Location: models/decision_tree.png")