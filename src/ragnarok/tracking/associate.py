"""Linear assignment via scipy (drop-in replacement for ``lap.lapjv``).

Mask costs above ``thresh`` so the solver avoids them, solve the rectangular
assignment problem, then prune any returned pair whose true cost exceeds
``thresh``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost, thresh):
    """Solve assignment, returning ``(matches Nx2, unmatched_rows, unmatched_cols)``.

    Parameters
    ----------
    cost : ndarray (R, C)
        Cost matrix; lower is better.
    thresh : float
        Maximum acceptable cost for a valid match.

    Returns
    -------
    matches : ndarray (N, 2) int
        Each row is ``(row_index, col_index)`` of an accepted pair.
    unmatched_rows : tuple[int, ...]
    unmatched_cols : tuple[int, ...]
    """
    if cost.size == 0:
        return (np.empty((0, 2), int),
                tuple(range(cost.shape[0])),
                tuple(range(cost.shape[1])))
    c = cost.copy()
    c[c > thresh] = thresh + 1e-5
    rows, cols = linear_sum_assignment(c)
    matches = []
    ur, uc = set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, k in zip(rows, cols):
        if cost[r, k] <= thresh:
            matches.append((r, k))
            ur.discard(r)
            uc.discard(k)
    return (np.array(matches).reshape(-1, 2), tuple(sorted(ur)), tuple(sorted(uc)))
