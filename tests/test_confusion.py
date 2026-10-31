import numpy as np

from intent.confusion import most_confused


def test_picks_pairs_by_errors_in_both_directions():
    cm = np.array([
        [9, 2, 0, 0],   # 0 -> 1 twice
        [3, 7, 0, 0],   # 1 -> 0 three times: pair (0, 1) has 5 errors
        [0, 0, 8, 4],   # 2 -> 3 four times, never back: pair (2, 3) has 4
        [0, 0, 0, 10],
    ])
    assert most_confused(cm, 2) == [0, 1]
    assert most_confused(cm, 4) == [0, 1, 2, 3]


def test_a_class_with_scattered_errors_does_not_crowd_out_a_real_pair():
    # Class 0 makes 6 errors, one to each of six other classes; classes 7 and 8 confuse each other 4 times.
    cm = np.eye(9, dtype=int) * 10
    cm[0, 1:7] = 1
    cm[7, 8] = cm[8, 7] = 2
    assert most_confused(cm, 2) == [7, 8]


def test_stops_when_there_are_no_more_errors():
    cm = np.diag([5, 5, 5])
    cm[0, 1] = 1
    assert most_confused(cm, 10) == [0, 1]
