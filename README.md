# Project Title

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
  - [Installation](#installation)
  - [Usage](#usage)
  - [CLI Arguments](#cli-arguments)

## Summary

This project focuses on developing methods for preprocessing and model selection in data mining.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Kostas-Xafis/DataMining
    cd DataMining
    ```

2. Create a virtual environment and activate it (if needed):

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:

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

3. Extract the top 10 features and compare with new evaluation:

    ```bash
    python classification.py --top10
    ```

4. Extract the top 10 features and perform regression analysis:

    ```bash
    python regression.py
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
