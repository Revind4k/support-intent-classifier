"""Baselines: majority class, and TF-IDF (word + character n-grams) with logistic regression.

    uv run python -m intent.baseline
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from .data import load
from .metrics import report

RESULTS = Path(__file__).resolve().parents[2] / "results"


def tfidf_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    return make_pipeline(features, LogisticRegression(C=10, max_iter=2000))


def main() -> None:
    train, val, test, names = load()
    majority = Counter(train.labels).most_common(1)[0][0]
    out = {"majority": report(test.labels, [majority] * len(test.labels), names)}
    model = tfidf_model().fit(train.texts, train.labels)
    out["tfidf_logreg"] = {"val": report(val.labels, list(model.predict(val.texts)), names),
                           "test": report(test.labels, list(model.predict(test.texts)), names)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"majority      test acc {out['majority']['accuracy']:.4f}")
    t = out["tfidf_logreg"]["test"]
    print(f"tfidf+logreg  test acc {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
