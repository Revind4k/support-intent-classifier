from __future__ import annotations

from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


def report(y_true: list[int], y_pred: list[int], names: list[str], top_confusions: int = 5) -> dict:
    pairs = Counter((names[t], names[p]) for t, p in zip(y_true, y_pred) if t != p)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "n": len(y_true),
        "top_confusions": [{"true": t, "predicted": p, "count": c} for (t, p), c in pairs.most_common(top_confusions)],
    }
