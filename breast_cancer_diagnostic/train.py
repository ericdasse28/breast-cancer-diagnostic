from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def train(X: np.ndarray, y: np.ndarray) -> Model:
    model = LogisticRegression()
    model.fit(X, y)
    return model
