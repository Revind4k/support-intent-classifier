"""BANKING77 loading and a fixed, stratified validation split carved out of the training set.

The official test split is touched exactly once, for the final numbers. Model selection
(epochs, the best checkpoint) uses only the validation split.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[2] / "data"
SEED = 13


@dataclass
class Split:
    texts: list[str]
    labels: list[int]


def read_csv(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["text"].strip()]
    return [r["text"].strip() for r in rows], [r["category"] for r in rows]


def load(val_fraction: float = 0.1) -> tuple[Split, Split, Split, list[str]]:
    tr_x, tr_y = read_csv(DATA / "train.csv")
    te_x, te_y = read_csv(DATA / "test.csv")
    names = sorted(set(tr_y))
    idx = {n: i for i, n in enumerate(names)}
    x_train, x_val, y_train, y_val = train_test_split(
        tr_x, [idx[y] for y in tr_y], test_size=val_fraction, random_state=SEED, stratify=tr_y
    )
    return Split(x_train, y_train), Split(x_val, y_val), Split(te_x, [idx[y] for y in te_y]), names
