from dataclasses import dataclass

import pandas as pd
from ucimlrepo import fetch_ucirepo


@dataclass
class Dataset:
    features: pd.DataFrame
    targets: pd.Series


def load_dataset() -> Dataset:
    aspirates = fetch_ucirepo(id=17)
    X = aspirates.data.features
    y = aspirates.data.targets

    return Dataset(X, y)
