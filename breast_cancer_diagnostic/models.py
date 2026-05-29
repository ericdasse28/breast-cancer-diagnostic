from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def create_logistic_regression():
    return LogisticRegression()
