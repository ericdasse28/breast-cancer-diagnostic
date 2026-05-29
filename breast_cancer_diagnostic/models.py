from __future__ import annotations

from typing import Protocol

import pandas as pd
from sklearn.linear_model import LogisticRegression


class Model(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Model: ...
    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...


def create_logistic_regression():
    return LogisticRegression()
