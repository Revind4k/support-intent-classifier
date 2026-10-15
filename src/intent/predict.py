"""Classify messages with the fine-tuned checkpoint.

    uv run python -m intent.predict "my card still hasn't arrived" "how do I top up by bank transfer?"
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CKPT = Path(__file__).resolve().parents[2] / "checkpoints" / "distilbert-base-uncased"


def main(texts: list[str]) -> None:
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT).eval()
    with torch.no_grad():
        probs = model(**tok(texts, padding=True, truncation=True, return_tensors="pt")).logits.softmax(-1)
    for text, p in zip(texts, probs):
        top = p.topk(3)
        best = ", ".join(f"{model.config.id2label[i.item()]} {v.item():.2f}" for v, i in zip(top.values, top.indices))
        print(f"{text!r}\n  -> {best}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["my card still hasn't arrived"])
