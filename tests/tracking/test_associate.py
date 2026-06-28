import numpy as np

from ragnarok.tracking.associate import linear_assignment


def test_simple_two_by_two():
    # row0 prefers col0 (cost 0.1), row1 prefers col1 (cost 0.2);
    # off-diagonal entries are above thresh.
    cost = np.array([[0.1, 0.9], [0.9, 0.2]])
    matches, ur, uc = linear_assignment(cost, thresh=0.5)
    pairs = {tuple(m) for m in matches}
    assert pairs == {(0, 0), (1, 1)}
    assert ur == ()
    assert uc == ()


def test_above_thresh_unmatched():
    # only one viable pairing (0,0); the rest exceed thresh.
    cost = np.array([[0.1, 0.9], [0.9, 0.95]])
    matches, ur, uc = linear_assignment(cost, thresh=0.5)
    pairs = {tuple(m) for m in matches}
    assert pairs == {(0, 0)}
    assert ur == (1,)
    assert uc == (1,)


def test_empty_cost():
    cost = np.empty((0, 3))
    matches, ur, uc = linear_assignment(cost, thresh=0.5)
    assert matches.shape == (0, 2)
    assert ur == ()
    assert uc == (0, 1, 2)


def test_rectangular_more_dets():
    cost = np.array([[0.1, 0.9, 0.9]])
    matches, ur, uc = linear_assignment(cost, thresh=0.5)
    pairs = {tuple(m) for m in matches}
    assert pairs == {(0, 0)}
    assert ur == ()
    assert uc == (1, 2)
