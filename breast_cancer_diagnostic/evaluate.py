import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from breast_cancer_diagnostic.models import Model


def evaluate(model: Model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="M")
    recall = recall_score(y_test, y_pred, pos_label="M")

    return {"accuracy": accuracy, "precision": precision, "recall": recall}
