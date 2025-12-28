# Install Python

```bash
brew install python
```

## Create a project folder and venv

```bash
mkdir transaction-autocoder
cd transaction-autocoder
python3 -m venv .venv
source .venv/bin/activate
```

## One-time install (macOS / Apple Silicon):

```bash
pip install -U pip
pip install pandas openpyxl numpy scikit-learn xgboost joblib
pip install sentence-transformers torch
```

### Quick Test

```bash
python -c "import xgboost, torch; print('OK')"
```
