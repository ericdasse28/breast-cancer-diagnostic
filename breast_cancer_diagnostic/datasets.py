from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


@dataclass
class Dataset:
    features: pd.DataFrame
    targets: pd.Series


def load_dataset() -> Dataset:
    aspirates = fetch_ucirepo(id=17)
    X = aspirates.data.features
    y = aspirates.data.targets["Diagnosis"]
    return Dataset(X, y)


def split_dataset(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Splits dataset into training
    and test sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features, dataset.targets, test_size=0.2, random_state=123
    )

    return Dataset(X_train, y_train), Dataset(X_test, y_test)
