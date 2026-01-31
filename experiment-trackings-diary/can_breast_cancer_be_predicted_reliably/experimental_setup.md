# Experimental setup

## Question

> Can we automatically whether a fine-needle aspirate (FNA) sample is _benign_
> or _malignant_ in a reliable way using ML?

In this experimental setting, we are trying to find out if a machine learning model can
predict whether a fine-needle aspirate sample is benign or malignant, and be reliable
enough to be of use to medical professionals.

## Background research

### Dataset

[Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Exploratory data analysis (EDA)

See [here](./Breast%20cancer%20data%20exploration.ipynb)

## Metrics

According to EDA, the dataset consists of around 62.7% benign observations and
37.3% malignant ones. Its distribution is skewed. Therefore, we picked the following metrics:

- Precision
- Recall
- Accuracy
