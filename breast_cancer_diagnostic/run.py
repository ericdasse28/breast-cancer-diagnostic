from sklearn.metrics import accuracy_score, precision_score, recall_score
from ucimlrepo import fetch_ucirepo

from breast_cancer_diagnostic.data_preprocessing import preprocess_data
from breast_cancer_diagnostic.train import train


def main():
    aspirates = fetch_ucirepo(id=17)
    X, y = aspirates.data.features, aspirates.data.targets
    X_cleaned = preprocess_data(X)

    model = train(X_cleaned.values, y.values.ravel())
    y_pred = model.predict(X_cleaned.values)

    print(f"===> Accuracy: {accuracy_score(y, y_pred)}")
    print(f"===> Precision: {precision_score(y, y_pred, pos_label='M')}")
    print(f"===> Recall: {recall_score(y, y_pred, pos_label='M')}")


if __name__ == "__main__":
    main()
