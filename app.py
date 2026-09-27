from flask import Flask, render_template, request
import joblib
import os

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)


# ==========================================
# Load / auto-create KNN model
# ==========================================

KNN_MODEL_PATH = "iris_knn_model.pkl"
RF_MODEL_PATH  = "iris_rf_model.pkl"


def _train_and_save_knn():
    """
    Train KNN using the EXACT same pipeline as iris_model.py and save the
    resulting model to iris_knn_model.pkl.  This runs only once when the
    pickle file is missing.
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species"] = df["species"].map({
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    })

    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    joblib.dump(knn, KNN_MODEL_PATH)
    print(f"[app] KNN model trained and saved to {KNN_MODEL_PATH}")
    return knn


# Load Random Forest (always from saved pkl)
rf_model = joblib.load(RF_MODEL_PATH)

# Load KNN — create automatically if pkl is missing
if os.path.exists(KNN_MODEL_PATH):
    knn_model = joblib.load(KNN_MODEL_PATH)
    print(f"[app] KNN model loaded from {KNN_MODEL_PATH}")
else:
    print(f"[app] {KNN_MODEL_PATH} not found — training KNN now...")
    knn_model = _train_and_save_knn()


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Dual Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    sepal_length = float(request.form["sepal_length"])
    sepal_width  = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width  = float(request.form["petal_width"])

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    # --- KNN prediction ---
    knn_prediction = knn_model.predict(input_data)[0]

    # --- Random Forest prediction + confidence ---
    rf_prediction = rf_model.predict(input_data)[0]

    rf_proba = rf_model.predict_proba(input_data)[0]
    rf_classes = rf_model.classes_          # ['setosa', 'versicolor', 'virginica']
    rf_confidence = round(
        rf_proba[list(rf_classes).index(rf_prediction)] * 100,
        1
    )

    # Build per-class confidence dict for template
    rf_proba_dict = {
        cls: round(prob * 100, 1)
        for cls, prob in zip(rf_classes, rf_proba)
    }

    # Are the two models in agreement?
    models_agree = (knn_prediction == rf_prediction)

    return render_template(
        "index.html",
        knn_prediction   = knn_prediction,
        rf_prediction    = rf_prediction,
        rf_confidence    = rf_confidence,
        rf_proba_dict    = rf_proba_dict,
        models_agree     = models_agree,
        # Keep legacy key so any leftover template refs still work
        prediction       = rf_prediction,
    )


# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)