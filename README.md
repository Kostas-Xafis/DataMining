# Project Title

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
  - [Installation](#installation)
  - [Usage](#usage)
  - [CLI Arguments](#cli-arguments)

## Summary

This project focuses on data mining and classification using various machine learning techniques. It includes data preprocessing, feature selection, model training, and evaluation.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To use the project, follow these steps:

1. Train and evaluate the model:

    ```bash
    python classification.py --classifier
    ```

2. Test with unlabeled data:

    ```bash
    python classification.py --unlabeled
    ```

## CLI Arguments

The `classification.py` script accepts the following arguments:

- `--simple`: Run a simple test.
- `--grid_search`: Perform a grid search for hyperparameter tuning.
- `--classifier`: Train and evaluate the classifier.
- `--top10`: Test with the top 10 features.
- `--unlabeled`: Test with unlabeled data.

Example:

```bash
python classification.py --classifier
```
