# Transaction Auto Coder

This program trains a model on bank transaction data. It then uses this model to assign budget codes to transactions.

## Input Files

- Training data:
  - training/training.xlsx
- Checking transactions:
  - input/to_code_checking.csv
- Credit transactions:
  - input/to_code_credit.csv

## Output Files

- Checking transactions with codes:
  - output/to_code_checking_scored.xlsx
- Credit transactions with codes:
  - output/to_code_credit_scored.xlsx

## Usage

1. Ensure files are in the expected locations:
   - training/training.xlsx
   - input/to_code_checking.csv
   - input/to_code_credit.csv
   - autocode.py

2. Run the program
    ```bash
    python autocode.py
    ```

## Training model cleanup
- Perform after updating training algorithm

    ```bash
    rm -f models/model_embeddings.xgb models/model_embeddings_classes.json
    rm -f models/model_embeddings.ubj models/model_embeddings_classes.json
    ```
