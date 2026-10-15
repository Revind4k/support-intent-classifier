# support-intent-classifier

Routes customer-support messages to one of 77 intents (BANKING77). It compares a strong classical baseline against two fine-tuned transformers, trained with a hand-written PyTorch loop.

## Results

Official BANKING77 test split: 3,080 messages, 77 intents. The best epoch is selected on a held-out validation split (10% of training, stratified), and the test split is evaluated once at the end.

| Model | Params | Test accuracy | Macro-F1 | Training time (Apple M4, MPS) |
|---|---|---|---|---|
| Majority class | - | 1.3% | - | - |
| TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression | - | 91.2% | 0.912 | about 1 min (CPU) |
| DistilBERT, fine-tuned 5 epochs | 67M | 92.4% | 0.924 | 10 min |
| **BERT-base, fine-tuned 6 epochs** | 110M | **93.1%** | **0.931** | 42 min |

BERT-base cuts the error rate from 8.8% to 6.9%: **22% fewer misrouted messages** than the baseline. The baseline is deliberately strong. Character n-grams catch most of the intents' surface vocabulary, so the transformers win on the remaining semantic cases.

The most common remaining mistakes are genuinely close intents:

| True intent | Predicted as |
|---|---|
| `top_up_reverted` | `top_up_failed` |
| `card_acceptance` | `country_support` |
| `why_verify_identity` | `verify_my_identity` (DistilBERT) |

Full metrics, per-epoch history and confusions are in `results/*.json`.

## Data checks

- **Exact duplicates.** 4 training texts are exact duplicates (none with conflicting labels).
- **Train/test overlap.** 7 of the 3,080 test messages also appear verbatim in the training data (0.2%). That's too small to move the results, but it is tested and documented rather than assumed away.

## How it works

- `data.py` loads the CSVs, builds a fixed stratified validation split (seed 13), and keeps the test split untouched.
- `baseline.py` builds a TF-IDF feature union and a logistic regression in scikit-learn.
- `train.py` is a plain PyTorch fine-tuning loop (no Trainer abstraction):
  - AdamW, linear warmup/decay, gradient clipping
  - dynamic padding per batch
  - best-on-validation checkpointing
  - runs on CUDA, Apple MPS, or CPU
- `predict.py` gives top-3 intents with probabilities for any message.

```bash
uv sync
uv run python -m intent.baseline
uv run python -m intent.train --model distilbert-base-uncased --epochs 5
uv run python -m intent.train --model bert-base-uncased --epochs 6 --lr 4e-5
uv run python -m intent.predict "my card still hasn't arrived" "why do you need to verify my identity?"
uv run pytest
```

## Data

BANKING77 by PolyAI (Casanueva et al., 2020, "Efficient Intent Detection with Dual Sentence Encoders"), redistributed in `data/` under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) from [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets).
