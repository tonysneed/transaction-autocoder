import os
import re
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

# =========================
# CONFIG
# =========================
TRAIN_FILE = "training.xlsx"
SCORE_FILE = "to_code.xlsx"
OUTPUT_FILE = "to_code_scored.xlsx"

# Choose: "tfidf" or "embeddings"
FEATURE_MODE = "embeddings"

# Train persistence (models)
FORCE_RETRAIN = False
MODEL_DIR = "models"

TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, "model_tfidf.joblib")
TFIDF_CLASSES_PATH = os.path.join(MODEL_DIR, "model_tfidf_classes.json")

EMBED_MODEL_PATH = os.path.join(MODEL_DIR, "model_embeddings.xgb")
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
# TF-IDF: Train or Load (Label-encoded)
# =========================
def train_or_load_tfidf(train_df: pd.DataFrame):
    if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(TFIDF_CLASSES_PATH) and not FORCE_RETRAIN:
        print("Loading saved TF-IDF model...")
        model = joblib.load(TFIDF_MODEL_PATH)
        with open(TFIDF_CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = np.array(json.load(f), dtype=object)
        return model, classes

    print("Training TF-IDF model (full dataset, no validation split)...")

    X = train_df[["Text", "AmountNum", "AbsAmount", "IsDebit"]]
    y_raw = train_df[COL_CODE].astype(str).to_numpy()

    # Encode string labels to 0..K-1 (required for some XGBoost versions)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # ints
    classes = le.classes_.astype(object)

    pre = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True
            ), "Text"),
            ("num", "passthrough", ["AmountNum", "AbsAmount", "IsDebit"]),
        ],
        remainder="drop"
    )

    clf = build_xgb_classifier(num_classes=len(classes))
    model = Pipeline(steps=[("pre", pre), ("clf", clf)])

    model.fit(X, y)
    print("Trained on full dataset (no validation split).")

    joblib.dump(model, TFIDF_MODEL_PATH)
    with open(TFIDF_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes.tolist(), f)

    print(f"Saved model → {TFIDF_MODEL_PATH}")
    print(f"Saved classes → {TFIDF_CLASSES_PATH}")
    return model, classes

# =========================
# Embeddings: Train or Load (already int-labeled)
# =========================
def train_or_load_embeddings(train_df: pd.DataFrame):
    # Load saved booster + classes
    if os.path.exists(EMBED_MODEL_PATH) and os.path.exists(EMBED_CLASSES_PATH) and not FORCE_RETRAIN:
        print("Loading saved Embeddings model...")
        clf = build_xgb_classifier(num_classes=1)  # placeholder; booster carries the real config
        booster = xgb.Booster()
        booster.load_model(EMBED_MODEL_PATH)
        clf._Booster = booster  # attach booster to sklearn wrapper
        with open(EMBED_CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = np.array(json.load(f), dtype=object)
        return clf, classes

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

    # Save underlying Booster (avoids sklearn metadata issues)
    clf.get_booster().save_model(EMBED_MODEL_PATH)
    with open(EMBED_CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(classes.tolist(), f)

    print(f"Saved model → {EMBED_MODEL_PATH}")
    print(f"Saved classes → {EMBED_CLASSES_PATH}")
    return clf, classes

# =========================
# MAIN
# =========================
def main():
    train_df = pd.read_excel(TRAIN_FILE)
    score_df = pd.read_excel(SCORE_FILE)

    # Validate required columns
    for col in [COL_MERCHANT, COL_DESCRIPTION, COL_AMOUNT, COL_CODE]:
        if col not in train_df.columns:
            raise ValueError(f"Training file missing required column: '{col}'")
    for col in [COL_MERCHANT, COL_DESCRIPTION, COL_AMOUNT]:
        if col not in score_df.columns:
            raise ValueError(f"Scoring file missing required column: '{col}'")

    # Prep numeric + text
    train_df = add_numeric_features(train_df)
    score_df = add_numeric_features(score_df)

    train_df["Text"] = build_text_column(train_df)
    score_df["Text"] = build_text_column(score_df)

    # Guard: ensure there is at least 1 row to score
    if len(score_df) == 0:
        raise ValueError(
            f"'{SCORE_FILE}' contains 0 rows to score. "
            "Make sure it has data rows under the header row."
        )

    # Guard: ensure Text isn't entirely empty (optional but useful)
    if score_df["Text"].str.len().sum() == 0:
        raise ValueError(
            f"All 'Text' values are empty after normalization in '{SCORE_FILE}'. "
            "Check that Merchant/Description cells contain text."
        )

    # Only labeled rows in training
    train_df = train_df[train_df[COL_CODE].notna()].copy()
    train_df[COL_CODE] = train_df[COL_CODE].astype(str)

    if FEATURE_MODE.lower() == "tfidf":
        model, classes = train_or_load_tfidf(train_df)
        X_new = score_df[["Text", "AmountNum", "AbsAmount", "IsDebit"]]
        probs = model.predict_proba(X_new)

    elif FEATURE_MODE.lower() == "embeddings":
        clf, classes = train_or_load_embeddings(train_df)

        score_texts = score_df["Text"].tolist()
        score_emb = embed_texts_local(score_texts)

        score_num = score_df[["AmountNum", "AbsAmount", "IsDebit"]].to_numpy(dtype=np.float32)
        X_new = np.hstack([score_emb, score_num]).astype(np.float32)

        probs = clf.predict_proba(X_new)

    else:
        raise ValueError("FEATURE_MODE must be 'tfidf' or 'embeddings'")

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

    out.to_excel(OUTPUT_FILE, index=False)
    print(f"Wrote → {OUTPUT_FILE}")
    print(f"Mode: {FEATURE_MODE} | Auto threshold: {AUTO_ASSIGN_THRESHOLD} | Force retrain: {FORCE_RETRAIN}")

if __name__ == "__main__":
    main()