"""Train the same two models as the notebook and save them for the app.

The split, scaler, and models match task1_iris_classification.ipynb:
80/20 split, stratify by species, random_state=42, StandardScaler fit
on the training rows only.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train():
    df = pd.read_csv(ROOT / "iris.csv")
    X = df[FEATURE_COLS]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(random_state=42)
    tree = DecisionTreeClassifier(random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    tree.fit(X_train_scaled, y_train)

    log_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))
    tree_acc = accuracy_score(y_test, tree.predict(X_test_scaled))

    bundle = {
        "scaler": scaler,
        "log_reg": log_reg,
        "tree": tree,
        "feature_cols": FEATURE_COLS,
        "classes": list(log_reg.classes_),
    }
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, out_dir / "iris_pipeline.joblib")
    return bundle, log_acc, tree_acc


if __name__ == "__main__":
    _, log_acc, tree_acc = train()
    print(f"Test accuracy, logistic regression: {log_acc:.3f}")
    print(f"Test accuracy, decision tree: {tree_acc:.3f}")
