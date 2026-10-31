"""Confusion matrix for a fine-tuned checkpoint, zoomed in on the intents it mixes up most.

    uv run python -m intent.confusion --model bert-base-uncased

Runs the checkpoint over the official test split, writes the full confusion matrix and the
per-intent error counts to results/confusion-<model>.json, and draws the sub-matrix for the
k most-confused intents to docs/confusion-<model>.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from intent.data import load
from intent.train import device as pick_device

ROOT = Path(__file__).resolve().parents[2]


def most_confused(cm: np.ndarray, k: int) -> list[int]:
    """Up to k classes that make up the most-confused pairs, in the order the pairs were picked.

    Pairs are ranked by errors in both directions (true a predicted as b, plus b as a), and both
    classes of each pair are added until k classes are chosen. Picking pairs rather than the
    classes with the most errors overall keeps the mistakes inside the plotted block: a class
    with many errors spread thinly across 20 others makes a mostly empty row. Ties break by index.
    """
    off = cm - np.diag(np.diag(cm))
    sym = off + off.T
    pairs = sorted(((sym[i, j], i, j) for i in range(len(cm)) for j in range(i + 1, len(cm)) if sym[i, j]),
                   key=lambda t: (-t[0], t[1], t[2]))
    chosen: list[int] = []
    for _, i, j in pairs:
        for c in (i, j):
            if c not in chosen and len(chosen) < k:
                chosen.append(c)
        if len(chosen) == k:
            break
    return chosen


def predict(model_name: str, texts: list[str], batch_size: int = 64) -> list[int]:
    ckpt = ROOT / "checkpoints" / model_name
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i : i + batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
            preds += model(**batch).logits.argmax(-1).tolist()
    return preds


def plot(cm: np.ndarray, idx: list[int], names: list[str], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = cm[np.ix_(idx, idx)]
    labels = [names[i] for i in idx]
    # Shade by share of each true intent's test messages, so rows of different sizes compare fairly.
    shares = sub / cm[idx].sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(np.where(np.eye(len(idx), dtype=bool) | (sub == 0), np.nan, shares), cmap="Reds", vmin=0, vmax=max(shares.max(), 0.01))
    for r in range(len(idx)):
        for c in range(len(idx)):
            if sub[r, c]:
                ax.text(c, r, sub[r, c], ha="center", va="center", fontsize=9,
                        color="0.45" if r == c else "black")
    ax.set_xticks(range(len(idx)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(idx)), labels, fontsize=8)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    _, _, test, names = load()
    preds = predict(args.model, test.texts)
    cm = confusion_matrix(test.labels, preds, labels=range(len(names)))
    acc = np.trace(cm) / cm.sum()
    idx = most_confused(cm, args.k)
    errors_in_top = int(cm[np.ix_(idx, idx)].sum() - np.trace(cm[np.ix_(idx, idx)]))
    total_errors = int(cm.sum() - np.trace(cm))

    out = {
        "model": args.model,
        "accuracy": round(float(acc), 4),
        "errors": total_errors,
        "top_k": [names[i] for i in idx],
        "errors_within_top_k": errors_in_top,
        "matrix": cm.tolist(),
        "labels": names,
    }
    (ROOT / "results" / f"confusion-{args.model}.json").write_text(json.dumps(out, indent=1) + "\n")
    title = (f"{args.model}: the {len(idx)} most-confused intents "
             f"({errors_in_top} of {total_errors} test errors fall inside this block)")
    plot(cm, idx, names, title, ROOT / "docs" / f"confusion-{args.model}.png")
    print(f"accuracy {acc:.4f}  errors {total_errors}  within top-{len(idx)}: {errors_in_top}")
    print("top intents:", ", ".join(out["top_k"]))


if __name__ == "__main__":
    main()
