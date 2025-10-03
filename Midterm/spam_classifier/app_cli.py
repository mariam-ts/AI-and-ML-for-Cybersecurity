#!/usr/bin/env python3
"""
Console app: classify a single email body as spam/legit using trained artifacts.

Usage:
  python app_cli.py \
    --model_artifact artifacts/model.joblib \
    --text "Your email body here..."

If your model was trained on text, it will use TF-IDF inside the pipeline.
If your model was trained on numeric columns only, this app will try to use
engineered numeric features from feature_extractor.simple_numeric_features
and align them to the training schema (missing features -> 0).
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from feature_extractor import simple_numeric_features


def load_schema(schema_path: Path) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_artifact", default="artifacts/model.joblib")
    ap.add_argument("--vectorizer_artifact", default="artifacts/vectorizer.joblib",
                    help="Only used for text-mode models (optional; pipeline already includes vectorizer).")
    ap.add_argument("--text", required=True, help="Email body text to classify")
    args = ap.parse_args()

    model = joblib.load(args.model_artifact)
    schema = load_schema(Path("artifacts/schema.json"))

    text_mode = schema.get("text_mode", False)

    if text_mode:
        # Pipeline includes TF-IDF -> logistic regression; just pass a list with the text
        X = [args.text]
        proba = model.predict_proba(X)[0][1]
    else:
        # Numeric-only model: build engineered features and align columns
        feats = simple_numeric_features(args.text)
        cols = schema.get("feature_columns", [])
        row = {c: feats.get(c, 0.0) for c in cols}  # fill missing with 0
        X = pd.DataFrame([row], columns=cols)
        proba = model.predict_proba(X)[0][1]

    pred = int(proba >= 0.5)
    label = "SPAM" if pred == 1 else "LEGITIMATE"
    print(f"P(spam) = {proba:.4f}  →  Predicted: {label}")


if __name__ == "__main__":
    main()
