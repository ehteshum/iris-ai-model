from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib
import os


# ==========================================
# 1. Load Iris Dataset
# ==========================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("Dataset loaded successfully!")
print(df.head())

print("\nSpecies Count:")
print(df["species"].value_counts())


# ==========================================
# 2. Create static folder
# ==========================================

os.makedirs("static", exist_ok=True)


# ==========================================
# 3. Species Distribution Graph
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="species"
)

plt.title("Number of Samples per Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.savefig("static/species_distribution.png")
plt.close()


# ==========================================
# 4. Petal Length vs Petal Width
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species"
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.tight_layout()
plt.savefig("static/petal_scatter.png")
plt.close()


# ==========================================
# 5. Prepare Data
# ==========================================

X = df.drop("species", axis=1)
y = df["species"]


# ==========================================
# 6. Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. KNN Model
# ==========================================

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

print("\nKNN Accuracy:", knn_accuracy)


# ==========================================
# 8. Random Forest Model
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("Random Forest Accuracy:", rf_accuracy)


# ==========================================
# 9. Save Random Forest Model
# ==========================================

joblib.dump(
    rf,
    "iris_rf_model.pkl"
)

print("\nRandom Forest model saved!")


# ==========================================
# 10. Accuracy Comparison Graph
# ==========================================

models = [
    "KNN",
    "Random Forest"
]

accuracies = [
    knn_accuracy,
    rf_accuracy
]

plt.figure(figsize=(7, 5))

plt.bar(
    models,
    accuracies
)

plt.title("KNN vs Random Forest Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig("static/accuracy_comparison.png")
plt.close()


# ==========================================
# 11. Feature Importance
# ==========================================

feature_importance = rf.feature_importances_

print("\nFeature Importance:")

for feature, importance in zip(
    X.columns,
    feature_importance
):
    print(
        feature,
        ":",
        importance
    )


plt.figure(figsize=(8, 5))

sns.barplot(
    x=feature_importance,
    y=X.columns
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()
plt.savefig("static/feature_importance.png")
plt.close()


# ==========================================
# 12. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    rf_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "setosa",
        "versicolor",
        "virginica"
    ],
    yticklabels=[
        "setosa",
        "versicolor",
        "virginica"
    ]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()


print("\nAll graphs generated successfully!")