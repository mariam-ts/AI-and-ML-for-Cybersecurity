#!/usr/bin/env python3
"""
Train & evaluate a spam classifier (70/30 split) on the provided CSV.

Features:
- Auto-detects schema:
  * If a 'text' column exists -> TF-IDF + LogisticRegression
  * Else -> uses numeric feature columns directly (LogisticRegression)
- Auto-detects label column from common names.
- Saves:
  * artifacts/model.joblib
  * artifacts/vectorizer.joblib (text mode only)
  * artifacts/schema.json (feature names + flags)
  * figures/confusion_matrix.png
  * figures/class_balance.png
  * figures/top_features.png
  * model_coefficients.csv
- Prints accuracy to console.

Usage:
  python train_and_eval.py \
    --csv spam_classifier/data/m_tsirekidze2024_829461.csv \
    --test_size 0.30 --random_state 42
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_CANDIDATES = [
    "label", "class", "target", "is_spam", "spam", "y"
]
POSITIVE_SPAM_VALUES = {"spam", "1", 1, True, "true", "yes"}


def find_label_column(df: pd.DataFrame) -> str:
    # Try common label names first
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    # Otherwise, guess last column if it looks binary/categorical
    last = df.columns[-1]
    return last


def coerce_label(y: pd.Series) -> pd.Series:
    # Map to {0,1} where 1 = spam
    def map_val(v):
        if isinstance(v, str):
            v_low = v.strip().lower()
            return 1 if v_low in POSITIVE_SPAM_VALUES else 0
        if isinstance(v, (int, np.integer, bool, np.bool_)):
            return 1 if v in POSITIVE_SPAM_VALUES else int(bool(v))
        return int(bool(v))
    return y.apply(map_val).astype(int)


def get_text_column(df: pd.DataFrame) -> Optional[str]:
    for name in ["text", "body", "email", "message", "content"]:
        if name in df.columns:
            return name
    # try to find a likely free-text column (heuristic)
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.len().mean() > 30:
            return c
    return None


def make_dirs():
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)


def plot_confusion(y_true, y_pred, outpath: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["legit(0)", "spam(1)"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (Test)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_class_balance(y: pd.Series, outpath: str):
    fig, ax = plt.subplots()
    counts = y.value_counts().sort_index()
    counts.index = ["legit(0)", "spam(1)"]
    ax.bar(counts.index, counts.values)
    ax.set_title("Class Balance")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_top_features(coef: np.ndarray, feat_names: List[str], outpath: str, top_k: int = 20):
    # Show most positive (spam) and most negative (legit) features
    coef = coef.ravel()
    idx_sorted = np.argsort(coef)
    neg_idx = idx_sorted[:top_k]
    pos_idx = idx_sorted[-top_k:]
    names = [feat_names[i] for i in list(neg_idx) + list(pos_idx)]
    vals = list(coef[neg_idx]) + list(coef[pos_idx])
    colors = ["#1f77b4"] * len(neg_idx) + ["#d62728"] * len(pos_idx)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_title("Top Feature Weights (negative=legit, positive=spam)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def save_coefficients_csv(coef: np.ndarray, feat_names: List[str], outpath: str):
    df = pd.DataFrame({"feature": feat_names, "coefficient": coef.ravel()})
    df.sort_values("coefficient", ascending=False, inplace=True)
    df.to_csv(outpath, index=False)


def build_text_pipeline(text_col: str) -> Tuple[Pipeline, List[str]]:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    model = LogisticRegression(max_iter=1000, n_jobs=None)
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", model),
    ])
    return pipe, [text_col]


def build_numeric_pipeline(num_cols: List[str]) -> Tuple[Pipeline, List[str]]:
    # Scale numeric features then LR
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=False), num_cols)],
        remainder="drop",
        sparse_threshold=0.0
    )
    model = LogisticRegression(max_iter=1000)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", model),
    ])
    return pipe, num_cols


def main():
    make_dirs()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV with features and label")
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--label", help="Override label column name")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.label:
        label_col = args.label
    else:
        label_col = find_label_column(df)
    if label_col not in df.columns:
        raise SystemExit(f"Label column '{label_col}' not found in CSV. Available: {list(df.columns)}")

    y = coerce_label(df[label_col])
    X = df.drop(columns=[label_col])

    # Decide text vs numeric mode
    text_col = get_text_column(X)
    text_mode = text_col is not None

    if text_mode:
        print(f"[INFO] Detected text column: {text_col}")
        X_use = X[text_col].astype(str)
        pipe, feat_cols = build_text_pipeline(text_col)
    else:
        # choose numeric columns
        num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
        if not num_cols:
            raise SystemExit("No numeric columns found and no text column detected. Please specify schema.")
        print(f"[INFO] Using numeric columns ({len(num_cols)}): {num_cols[:8]}{' ...' if len(num_cols)>8 else ''}")
        X_use = X[num_cols]
        pipe, feat_cols = build_numeric_pipeline(num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Accuracy (test): {acc:.4f}")

    # Save confusion matrix & class balance
    plot_confusion(y_test, y_pred, "figures/confusion_matrix.png")
    plot_class_balance(y, "figures/class_balance.png")

    # Extract coefficients + feature names
    if text_mode:
        # Recover TF-IDF vocabulary order
        tfidf: TfidfVectorizer = pipe.named_steps["tfidf"]
        feat_names = tfidf.get_feature_names_out().tolist()
        coef = pipe.named_steps["clf"].coef_
    else:
        # ColumnTransformer output order
        pre: ColumnTransformer = pipe.named_steps["pre"]
        num_names = pre.transformers_[0][2]  # list of numeric columns
        feat_names = list(num_names)
        coef = pipe.named_steps["clf"].coef_

    # Save coefficients CSV and top features plot
    save_coefficients_csv(coef, feat_names, "model_coefficients.csv")
    plot_top_features(coef, feat_names, "figures/top_features.png", top_k=15)

    # Save artifacts
    joblib.dump(pipe, "artifacts/model.joblib")
    print("[SAVE] artifacts/model.joblib")

    schema = {
        "text_mode": text_mode,
        "label_col": label_col,
        "feature_columns": feat_cols if not text_mode else [text_col],
        "vectorizer_present": bool(text_mode),
    }
    with open("artifacts/schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print("[SAVE] artifacts/schema.json")

    # If text mode, also save the vectorizer alone (optional convenience)
    if text_mode:
        vec = pipe.named_steps["tfidf"]
        joblib.dump(vec, "artifacts/vectorizer.joblib")
        print("[SAVE] artifacts/vectorizer.joblib")

    print("[DONE] Training + evaluation complete.")


if __name__ == "__main__":
    main()
