# Transaction Auto Coder

This program trains a model on bank transaction data. It then uses this model to assign budget codes to transactions.

## Input Files

- **training.xlsx**: contains training data
- **to_code**.xlsx: contains transactions which are to be coded

## Usage

1. Put both Excel files and the script in the same folder:
   - training.xlsx
   - to_code.xlsx
   - autocode.py

2. Run the program
    ```bash
    python autocode.py
    ```

## Output File

- **to_code_scored.xlsx**: transactions with codes

## Training model cleanup

```bash
rm -f models/model_embeddings.xgb models/model_embeddings_classes.json
```
