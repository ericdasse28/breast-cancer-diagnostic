import numpy as np

from breast_cancer_diagnostic.models import Model


def train(model: Model, X: np.ndarray, y: np.ndarray) -> Model:
    model.fit(X, y)
    return model
