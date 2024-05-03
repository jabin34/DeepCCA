

# Deep Canonical Correlation Analysis (DCCA)

## Overview

Python code for Deep Canonical Correlation Analysis (DCCA), which finds correlations in various kinds of data sources, is available in this repository. DCCA is a potent method for learning joint representations that optimise correlation across various data perspectives.

## Requirements

- Python 3.x
- PyTorch
- pandas
- scikit-learn

## Usage

1. **Data Preparation**: In the provided code, pathway data is stored in a CSV file named `kegg_legacy_ensembl.csv`.

2. **Run the Code**: Execute the Python script `DeepCCA.py` to train the DCCA model and evaluate it on the test set:

    ```bash
    python DeepCCA.py
    ```

3. **Output**: The script will print the validation loss for each epoch during training and the final test loss. Additionally, it will perform early stopping to prevent overfitting.

## Code Structure

- `DeepCCA.py`: Main Python script containing the DCCA model definition, training loop, and evaluation.
- `kegg_legacy_ensembl.csv`:  Pathway data in CSV format.
