import pandas as pd

from breast_cancer_diagnostic.models import Model


def train(model: Model, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Model:
    model.fit(X_train, y_train)
    return model
