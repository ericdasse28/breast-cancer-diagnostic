from __future__ import annotations

from typing import Protocol

import numpy as np


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Model: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def train(model: Model, X: np.ndarray, y: np.ndarray) -> Model:
    model.fit(X, y)
    return model
