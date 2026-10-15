"""Fine-tune a pretrained transformer for 77-way intent classification with a plain PyTorch loop.

    uv run python -m intent.train --model distilbert-base-uncased --epochs 5

The best epoch is chosen on the validation split; the test split is evaluated once, at the end.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import load
from .metrics import report

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(tok, texts, labels, batch_size, shuffle, max_len):
    order = list(range(len(texts)))
    def collate(ix):
        enc = tok([texts[i] for i in ix], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[i] for i in ix])
        return enc
    return DataLoader(order, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, dev) -> list[int]:
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        out.extend(model(**b).logits.argmax(-1).tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    seed_everything(args.seed)
    train, val, test, names = load()
    dev = device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(names), id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)}
    ).to(dev)

    train_dl = batches(tok, train.texts, train.labels, args.batch_size, True, args.max_len)
    val_dl = batches(tok, val.texts, val.labels, 128, False, args.max_len)
    test_dl = batches(tok, test.texts, test.labels, 128, False, args.max_len)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best = {"acc": -1.0}
    ckpt = ROOT / "checkpoints" / args.model.replace("/", "_")
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for b in train_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
        val_acc = report(val.labels, predict(model, val_dl, dev), names)["accuracy"]
        history.append({"epoch": epoch, "train_loss": round(total / len(train_dl), 4), "val_accuracy": val_acc})
        print(f"epoch {epoch}  loss {total / len(train_dl):.4f}  val acc {val_acc:.4f}  ({time.time() - t0:.0f}s)")
        if val_acc > best["acc"]:
            best = {"acc": val_acc, "epoch": epoch}
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)

    # Final, single evaluation on the held-out test split with the best-on-validation checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(dev)
    result = {
        "model": args.model, "device": str(dev), "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "args": vars(args), "best_epoch": best["epoch"], "history": history,
        "train_seconds": round(time.time() - t0),
        "test": report(test.labels, predict(model, test_dl, dev), names),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{args.model.replace('/', '_')}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"test acc {result['test']['accuracy']:.4f}  macro-F1 {result['test']['macro_f1']:.4f}  best epoch {best['epoch']}")


if __name__ == "__main__":
    main()
