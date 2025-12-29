# Transaction Auto Coder

This program trains a model on bank transaction data. It then uses this model to assign budget codes to transactions.

## Input Files

- **training.xlsx**: contains training data
- **to_code_checking**.xlsx: contains checking transactions which are to be coded
- **to_credit_checking**.xlsx: contains credit transactions which are to be coded

## Usage

1. Put Excel files and the script in the same folder:
   - training.xlsx
   - to_code_checking.xlsx
   - to_code_credit.xlsx
   - autocode.py

2. Run the program
    ```bash
    python autocode.py
    ```

## Output File

- **to_code_checking_scored.xlsx**: checking transactions with codes
- **to_code_credit_scored.xlsx**: credit transactions with codes

## Training model cleanup (after updating training algorithm)

```bash
rm -f models/model_embeddings.xgb models/model_embeddings_classes.json
rm -f models/model_embeddings.ubj models/model_embeddings_classes.json
```
