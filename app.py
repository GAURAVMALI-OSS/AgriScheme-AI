
from flask import Flask, render_template, request
import pickle
import pandas as pd

from schemes import SCHEMES, MAHARASHTRA_SCHEMES

app = Flask(__name__)


# Load trained ML models
model_names = [
    "Logistic_Regression",
    "Perceptron",
    "Neural_Network",
    "Decision_Tree"
]

models = {}

for name in model_names:
    path = f"models/{name}.pkl"

    with open(path, "rb") as file:
        models[name] = pickle.load(file)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Farmer scheme prediction
@app.route("/predict", methods=["POST"])
def predict():

    try:
        farmer = {
            "age": int(request.form["age"]),
            "land_area": float(request.form["land_area"]),
            "income": float(request.form["income"]),
            "crop": request.form["crop"],
            "irrigation": request.form["irrigation"],
            "category": request.form["category"]
        }

        input_data = pd.DataFrame([farmer])

        results = {}

        # Predictions from all four algorithms
        for name, model in models.items():
            prediction = model.predict(input_data)[0]
            results[name] = prediction

        # Get predicted scheme information
        scheme_details = {}

        for scheme_code in set(results.values()):

            if scheme_code in SCHEMES:
                scheme_details[scheme_code] = SCHEMES[scheme_code]

        # Send all scheme information to HTML
        return render_template(
            "index.html",
            results=results,
            farmer=farmer,
            scheme_details=scheme_details,
            maharashtra_schemes=MAHARASHTRA_SCHEMES
        )

    except Exception as error:

        return render_template(
            "index.html",
            error=str(error),
            maharashtra_schemes=MAHARASHTRA_SCHEMES
        )


if __name__ == "__main__":
    app.run(debug=True)
