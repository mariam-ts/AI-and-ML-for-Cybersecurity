#!/usr/bin/env python3
"""
Feature extractor for free-text emails (used by app_cli.py when the model was trained on text).
If the model was trained on numeric-only CSV features, app_cli will try to map to these engineered
features by name; any missing columns are filled with 0.
"""
import re
import math
from collections import Counter
from typing import Dict


SPAM_TOKENS = {
    "free","winner","win","bonus","congratulations","urgent","limited","offer",
    "money","cash","prize","click","now","verify","credit","loan","deal","gift",
    "discount","act","immediately","exclusive"
}

URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
DIGIT_RE = re.compile(r"\d")
UPPER_RE = re.compile(r"[A-Z]")
WORD_RE = re.compile(r"\b\w+\b")


def simple_numeric_features(text: str) -> Dict[str, float]:
    t = text or ""
    length = len(t)
    words = WORD_RE.findall(t)
    n_words = len(words)
    n_urls = len(URL_RE.findall(t))
    n_emails = len(EMAIL_RE.findall(t))
    n_digits = len(DIGIT_RE.findall(t))
    n_excl = t.count("!")
    n_q = t.count("?")
    n_upper = len(UPPER_RE.findall(t))
    cap_ratio = (n_upper / length) if length else 0.0

    # spam token counts (lower-cased)
    tokens = [w.lower() for w in words]
    cnt = Counter(tokens)
    spam_hits = sum(cnt[w] for w in SPAM_TOKENS)

    return {
        "len": float(length),
        "n_words": float(n_words),
        "n_urls": float(n_urls),
        "n_emails": float(n_emails),
        "n_digits": float(n_digits),
        "n_exclam": float(n_excl),
        "n_question": float(n_q),
        "cap_ratio": float(cap_ratio),
        "spam_token_hits": float(spam_hits),
    }
