## Copilot / Agent Instructions

This repository is a small Python utility that trains or loads a classifier to auto-assign transaction codes from Excel spreadsheets. Keep guidance concise and actionable for code edits or feature work.

- **Entry point:** [autocode.py](autocode.py). Inspect `main()` for end-to-end flow.
- **Primary files:** [README.md](README.md) (env & run steps), [headers.txt](headers.txt) (expected CSV/Excel header), and the `models/` directory (persisted artifacts).

- **Big picture:** `autocode.py` reads `training.xlsx` and `to_code.xlsx`, builds text and numeric features, trains or loads either a TF-IDF + XGBoost pipeline (`FEATURE_MODE='tfidf'`) or local sentence-transformer embeddings + XGBoost (`FEATURE_MODE='embeddings'`), then writes predictions to `to_code_scored.xlsx`.

- **Data contract / conventions:** The code expects exact Excel headers defined as constants in `autocode.py`: `COL_DATE`, `COL_MERCHANT`, `COL_DESCRIPTION`, `COL_AMOUNT`, `COL_CODE`. Use [headers.txt](headers.txt) for the canonical order.

- **Feature engineering:** Text is normalized by `normalize_text()` and combined via `build_text_column()`. Numeric features are added by `add_numeric_features()` producing `AmountNum`, `AbsAmount`, and `IsDebit`.

- **Model storage & retraining:** Models are persisted under `models/`:
  - TF-IDF pipeline: `models/model_tfidf.joblib` + `models/model_tfidf_classes.json`
  - Embeddings: `models/model_embeddings.xgb` + `models/model_embeddings_classes.json`
  - To force retrain either set `FORCE_RETRAIN = True` in `autocode.py` or delete the model files (see `README.md` cleanup section).

- **Embeddings & caching:** Local embeddings use `sentence-transformers` and are cached in `.emb_cache` keyed by a stable fingerprint (`_texts_fingerprint`). The code auto-detects device via `get_local_device()` (supports Apple's `mps` when available). When changing `EMBEDDING_MODEL_NAME` or source texts, cache invalidation occurs automatically via the fingerprint.

- **Training behavior & caveats:** Both training branches train on the full dataset with no validation split (explicit design). Labels must be present in `training.xlsx` under `Code`; training filters out unlabeled rows. This affects model changes — expect to retrain when label sets change.

- **XGBoost settings:** The XGBoost wrapper is created by `build_xgb_classifier()` (700 trees, depth 8, hist tree method, `multi:softprob`). For embeddings the code saves the underlying booster to `models/model_embeddings.xgb`.

- **Run & setup:** Follow [README.md](README.md): create a `venv`, install dependencies (pandas, openpyxl, numpy, scikit-learn, xgboost, joblib, sentence-transformers, torch), then run:

```
python autocode.py
```

- **Quick developer notes / examples:**
  - Switch feature mode: set `FEATURE_MODE = 'tfidf'` or `'embeddings'` in `autocode.py`.
  - Inspect a saved TF-IDF pipeline: open `models/model_tfidf.joblib` via joblib.load for debugging.
  - Recompute embeddings for the scoring set by removing `.emb_cache` files (or change `EMBEDDING_MODEL_NAME`).

- **Safety & expectations for edits:**
  - Preserve the exact Excel header constants unless also updating `headers.txt` and downstream consumers.
  - Avoid adding a validation split automatically; if you add one, update docstrings and `README.md` to reflect the new training flow.

- **Where to look for patterns / examples in repo:**
  - [autocode.py](autocode.py) — canonical implementation for normalization, features, training, persistence, and thresholds.
  - [README.md](README.md) — environment setup and run examples.

If anything above is unclear or you want more detail about a particular section (e.g., the embedding fingerprinting, TF-IDF pipeline internals, or adding a CI test), tell me which part to expand and I will iterate.
