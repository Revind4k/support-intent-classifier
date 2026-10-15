from collections import Counter

from intent.data import load
from intent.metrics import report


def test_split_is_deterministic_stratified_and_disjoint():
    a_train, a_val, test, names = load()
    b_train, b_val, _, _ = load()
    assert a_val.texts == b_val.texts  # fixed seed: the same validation set every run
    assert len(names) == 77
    # BANKING77 itself repeats 4 training texts, so a few can land on both sides of the split.
    assert len(set(a_val.texts) & set(a_train.texts)) <= 4
    per_class = Counter(a_val.labels)
    assert len(per_class) == 77 and min(per_class.values()) >= 3  # every intent is represented
    assert len(test.texts) == 3080


def test_report_counts_confusions():
    names = ["a", "b", "c"]
    r = report([0, 1, 2, 2], [0, 2, 2, 1], names)
    assert r["accuracy"] == 0.5
    pairs = {(c["true"], c["predicted"]) for c in r["top_confusions"]}
    assert pairs == {("b", "c"), ("c", "b")}


def test_test_split_barely_overlaps_training_data():
    train, val, test, _ = load()
    norm = lambda t: " ".join(t.lower().split())
    seen = {norm(t) for t in train.texts + val.texts}
    overlap = sum(norm(t) in seen for t in test.texts)
    assert overlap <= 7  # 7 of 3,080 test messages also appear in training (0.2%); see README
