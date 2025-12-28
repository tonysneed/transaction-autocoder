import os
import re
import json
import hashlib
import numpy as np
import pandas as pd

import xgboost as xgb

# =========================
# CONFIG
# =========================
TRAIN_FILE = "training.xlsx"
SCORE_FILE_CHECKING = "to_code_checking.csv"
SCORE_FILE_CREDIT = "to_code_credit.csv"
OUTPUT_FILE_CHECKING = "to_code_checking_scored.xlsx"
OUTPUT_FILE_CREDIT = "to_code_credit_scored.xlsx"

# Train persistence (models)
FORCE_RETRAIN = False
MODEL_DIR = "models"

# Save embeddings model as UBJSON for stable Booster load/save
EMBED_MODEL_PATH = os.path.join(MODEL_DIR, "model_embeddings.ubj")
EMBED_CLASSES_PATH = os.path.join(MODEL_DIR, "model_embeddings_classes.json")

# Auto-code threshold
AUTO_ASSIGN_THRESHOLD = 0.90  # consider 0.92+ for fully automatic posting

# Your EXACT Excel headers
COL_DATE = "Date"
COL_MERCHANT = "Merchant"
COL_DESCRIPTION = "Description"
COL_AMOUNT = "Amount"
COL_CODE = "Code"

# Local embeddings settings (only used if FEATURE_MODE="embeddings")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_CACHE_DIR = ".emb_cache"
EMBED_BATCH_SIZE = 256

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"\b\d{4,}\b", " ", s)
    s = re.sub(
        r"\b(DEBIT\s+CARD\s+PURCHASE|DEBIT\s+CARD|POS|ACH|ONLINE|PURCHASE|PAYMENT)\b",
        " ",
        s
    )
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def build_text_column(df: pd.DataFrame) -> pd.Series:
    desc = df[COL_DESCRIPTION].fillna("").astype(str).map(normalize_text)
    merch = df[COL_MERCHANT].fillna("").astype(str).map(normalize_text)
    return (desc + " " + merch).str.strip()

# =========================
# NUMERIC FEATURES
# =========================
def add_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["AmountNum"] = pd.to_numeric(out[COL_AMOUNT], errors="coerce").fillna(0.0)
    out["AbsAmount"] = out["AmountNum"].abs()
    # Assumes debits are negative, credits positive
    out["IsDebit"] = (out["AmountNum"] < 0).astype(int)
    return out

def parse_money(series: pd.Series) -> pd.Series:
    """Convert currency-formatted strings like '$1,234.56' to float."""
    return (
        series.astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .replace("nan", "0")
        .astype(float)
    )

def normalize_score_checking_csv(score_df: pd.DataFrame) -> pd.DataFrame:
    """Map bank-export CSV columns into the internal schema used by the model.

    Expected CSV columns:
      Date, Status, Type, CheckNumber, Description, Withdrawal, Deposit, RunningBalance

    Mappings:
      Date -> Date
      Description -> Merchant
      Withdrawal/Deposit -> Amount (Withdrawal treated as negative, Deposit positive)

    Internal columns added if missing:
      Merchant, Description, Amount
    """
    out = score_df.copy()

    # Merchant comes from the bank 'Description' field
    if "Description" in out.columns:
        out[COL_MERCHANT] = out["Description"].astype(str)
    else:
        out[COL_MERCHANT] = ""

    # Internal Description column (model uses it as part of text). If absent, leave blank.
    if COL_DESCRIPTION not in out.columns:
        out[COL_DESCRIPTION] = ""

    # Compute Amount from Withdrawal/Deposit if Amount not already present
    if COL_AMOUNT not in out.columns:
        w = parse_money(out.get("Withdrawal", 0)).fillna(0.0)
        d = parse_money(out.get("Deposit", 0)).fillna(0.0)
        # Withdrawal is money out (negative), Deposit money in (positive)
        out[COL_AMOUNT] = d - w

    return out

def normalize_score_credit_csv(score_df: pd.DataFrame) -> pd.DataFrame:
    """Map credit-card CSV columns into the internal schema used by the model.

    Expected CSV columns:
      Transaction Date, Description, Amount

    Mappings:
      Transaction Date -> Date
      Description -> Merchant
      Amount -> Amount (already signed; debits are negative)

    Internal columns added if missing:
      Merchant, Description, Amount, Date
    """
    out = score_df.copy()

    # Date
    if COL_DATE not in out.columns:
        if "Transaction Date" in out.columns:
            out[COL_DATE] = out["Transaction Date"]
        else:
            out[COL_DATE] = ""

    # Merchant comes from the credit 'Description' field
    if "Description" in out.columns:
        out[COL_MERCHANT] = out["Description"].astype(str)
    else:
        out[COL_MERCHANT] = ""

    # Internal Description column (model uses it as part of text). If absent, leave blank.
    if COL_DESCRIPTION not in out.columns:
        out[COL_DESCRIPTION] = ""

    # Amount comes directly from the credit 'Amount' column (may be currency formatted)
    if COL_AMOUNT not in out.columns:
        if "Amount" in out.columns:
            out[COL_AMOUNT] = parse_money(out["Amount"]).fillna(0.0)
        else:
            out[COL_AMOUNT] = 0.0

    return out

# =========================
# HASH-BASED EMBEDDING CACHE
# =========================
def _texts_fingerprint(texts: list[str], model_name: str) -> str:
    """
    Produces a stable fingerprint for the *content* of texts + embedding model.
    Changes if any text changes, order changes, or model_name changes.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\n")
    for t in texts:
        b = t.encode("utf-8", errors="ignore")
        h.update(str(len(b)).encode("ascii"))
        h.update(b":")
        h.update(b)
        h.update(b"\n")
    return h.hexdigest()

def _embed_cache_path(fingerprint: str) -> str:
    return os.path.join(EMBED_CACHE_DIR, f"{fingerprint}.npy")

def get_local_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def embed_texts_local(texts: list[str]) -> np.ndarray:
    """
    Embeds texts locally, caching by fingerprint.
    """
    fp = _texts_fingerprint(texts, EMBEDDING_MODEL_NAME)
    path = _embed_cache_path(fp)

    if os.path.exists(path):
        return np.load(path).astype(np.float32)

    from sentence_transformers import SentenceTransformer

    device = get_local_device()
    print(f"Embedding with SentenceTransformer on device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    emb = np.array(emb, dtype=np.float32)

    np.save(path, emb)
    return emb

# =========================
# XGBOOST
# =========================
def build_xgb_classifier(num_classes: int):
    return xgb.XGBClassifier(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1
    )

# =========================
# Embeddings: Train or Load (Booster only for stability)
# =========================
def train_or_load_embeddings(train_df: pd.DataFrame):
    if os.path.exists(EMBED_MODEL_PATH) and os.path.exists(EMBED_CLASSES_PATH) and not FORCE_RETRAIN:
        print("Loading saved Embeddings model...")
        booster = xgb.Booster()
        booster.load_model(EMBED_MODEL_PATH)
        with open(EMBED_CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = np.array(json.load(f), dtype=object)
        return booster, classes

    print("Training Embeddings model (full dataset, no validation split)...")

    train_texts = train_df["Text"].tolist()
    train_emb = embed_texts_local(train_texts)

    train_num = train_df[["AmountNum", "AbsAmount", "IsDebit"]].to_numpy(dtype=np.float32)
    X = np.hstack([train_emb, train_num]).astype(np.float32)

    y = train_df[COL_CODE].astype(str).to_numpy()
    classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int32)

    clf = build_xgb_classifier(num_classes=len(classes))
    clf.fit(X, y_idx)
    print("Trained on full dataset (no validation split).")

    booster = clf.get_booster()
    booster.save_model(EMBED_MODEL_PATH)
    with open(EMBED_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes.tolist(), f)

    print(f"Saved model → {EMBED_MODEL_PATH}")
    print(f"Saved classes → {EMBED_CLASSES_PATH}")
    return booster, classes

def score_transactions(score_df: pd.DataFrame, booster: xgb.Booster, classes: np.ndarray) -> pd.DataFrame:
    """Return a copy of score_df with SuggestedCode/Confidence/AutoCode/NeedsReview columns."""
    score_df = add_numeric_features(score_df)
    score_df["Text"] = build_text_column(score_df)

    if score_df["Text"].str.len().sum() == 0:
        raise ValueError("All 'Text' values are empty after normalization. Check Merchant/Description fields.")

    score_texts = score_df["Text"].tolist()
    score_emb = embed_texts_local(score_texts)

    score_num = score_df[["AmountNum", "AbsAmount", "IsDebit"]].to_numpy(dtype=np.float32)
    X_new = np.hstack([score_emb, score_num]).astype(np.float32)

    dmat = xgb.DMatrix(X_new)
    probs = booster.predict(dmat)

    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    best_idx = np.argmax(probs, axis=1)
    best_prob = probs[np.arange(len(best_idx)), best_idx]
    suggested = np.array(classes, dtype=object)[best_idx]

    auto_code = np.where(best_prob >= AUTO_ASSIGN_THRESHOLD, suggested, "")
    needs_review = auto_code == ""

    out = score_df.copy()
    out["SuggestedCode"] = suggested
    out["Confidence"] = best_prob
    out["AutoCode"] = auto_code
    out["NeedsReview"] = needs_review
    return out

# =========================
# MAIN
# =========================
def main():
    train_df = pd.read_excel(TRAIN_FILE)
    score_df_checking = pd.read_csv(SCORE_FILE_CHECKING)
    score_df_credit = pd.read_csv(SCORE_FILE_CREDIT)

    # Validate required columns
    for col in [COL_MERCHANT, COL_DESCRIPTION, COL_AMOUNT, COL_CODE]:
        if col not in train_df.columns:
            raise ValueError(f"Training file missing required column: '{col}'")
    # Score CSV schema validation (bank export)
    required_score_cols = ["Date", "Description", "Withdrawal", "Deposit"]
    missing = [c for c in required_score_cols if c not in score_df_checking.columns]
    if missing:
        raise ValueError(
            f"Scoring file missing required column(s): {missing}. "
            "Expected: Date, Description, Withdrawal, Deposit (plus optional other columns)."
        )
    required_credit_cols = ["Transaction Date", "Description", "Amount"]
    missing_credit = [c for c in required_credit_cols if c not in score_df_credit.columns]
    if missing_credit:
        raise ValueError(
            f"Credit scoring file missing required column(s): {missing_credit}. "
            "Expected: Transaction Date, Description, Amount (plus optional other columns)."
        )

    # Guard: ensure there is at least 1 row to score
    if len(score_df_checking) == 0:
        raise ValueError(
            f"'{SCORE_FILE_CHECKING}' contains 0 rows to score. "
            "Make sure it has data rows under the header row."
        )
    if len(score_df_credit) == 0:
        raise ValueError(
            f"'{SCORE_FILE_CREDIT}' contains 0 rows to score. "
            "Make sure it has data rows under the header row."
        )

    # Map bank-export CSV columns into internal schema
    score_df_checking = normalize_score_checking_csv(score_df_checking)
    score_df_credit = normalize_score_credit_csv(score_df_credit)

    # Prep numeric + text for training
    train_df = add_numeric_features(train_df)
    train_df["Text"] = build_text_column(train_df)

    # Only labeled rows in training
    train_df = train_df[train_df[COL_CODE].notna()].copy()
    train_df[COL_CODE] = train_df[COL_CODE].astype(str)

    if len(train_df) == 0:
        raise ValueError(f"No labeled rows found in '{TRAIN_FILE}'. Column '{COL_CODE}' must contain codes.")

    # Embeddings-only workflow
    booster, classes = train_or_load_embeddings(train_df)

    out_checking = score_transactions(score_df_checking, booster, classes)
    out_credit = score_transactions(score_df_credit, booster, classes)

    out_checking.to_excel(OUTPUT_FILE_CHECKING, index=False)
    print(f"Wrote → {OUTPUT_FILE_CHECKING}")

    out_credit.to_excel(OUTPUT_FILE_CREDIT, index=False)
    print(f"Wrote → {OUTPUT_FILE_CREDIT}")

    print(f"Auto threshold: {AUTO_ASSIGN_THRESHOLD} | Force retrain: {FORCE_RETRAIN}")

if __name__ == "__main__":
    main()