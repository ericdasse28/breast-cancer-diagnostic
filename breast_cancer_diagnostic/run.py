from breast_cancer_diagnostic.data_preprocessing import preprocess_data
from breast_cancer_diagnostic.datasets import load_dataset, split_dataset
from breast_cancer_diagnostic.evaluate import evaluate
from breast_cancer_diagnostic.models import create_logistic_regression
from breast_cancer_diagnostic.train import train


def main():
    breast_cancer_dataset = load_dataset()
    train_dataset, test_dataset = split_dataset(breast_cancer_dataset)

    X_train = preprocess_data(train_dataset.features)
    X_test = preprocess_data(test_dataset.features)

    model = create_logistic_regression()
    model = train(model, X_train, train_dataset.targets)

    metrics = evaluate(model, X_test, test_dataset.targets)

    print(f"===> Accuracy: {metrics['accuracy']}")
    print(f"===> Precision: {metrics['precision']}")
    print(f"===> Recall: {metrics['recall']}")


if __name__ == "__main__":
    main()
