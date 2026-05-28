import pandas as pd


def preprocess_data(aspirates: pd.DataFrame) -> pd.DataFrame:
    # First, drop all "worst" columns
    cols = [
        "radius3",
        "texture3",
        "perimeter3",
        "area3",
        "smoothness3",
        "compactness3",
        "concavity3",
        "concave_points3",
        "symmetry3",
        "fractal_dimension3",
    ]
    cleaned_aspirates = aspirates.drop(cols, axis=1)

    # Then, drop all columns related to the "perimeter" and "area" attributes
    cols = ["perimeter1", "perimeter2", "area1", "area2"]
    cleaned_aspirates = cleaned_aspirates.drop(cols, axis=1)

    # Lastly, drop all columns related to the "concavity" and "concave points" attributes
    cols = ["concavity1", "concavity2", "concave_points1", "concave_points2"]
    cleaned_aspirates = cleaned_aspirates.drop(cols, axis=1)

    return cleaned_aspirates
