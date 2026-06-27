"""Mahalanobis 2-DOF gate applied on top of the IoU association cost.

Pure numpy. Gates detections whose squared Mahalanobis distance to a track's
projected (x, y) center distribution exceeds the chi-square 0.95 quantile.
"""
from __future__ import annotations

import numpy as np

# chi-square 0.95 quantile, 2 DOF (== KalmanFilter.chi2inv95[2]); gate on (x, y).
CHI2_GATE_2DOF = 5.9915


def mahalanobis_gate(cost, track_mean_xy, track_S_xy, det_xy,
                     thresh=CHI2_GATE_2DOF, gated=np.inf):
    """Set ``cost[t, d]`` to ``gated`` when det ``d`` is outside track ``t``'s gate.

    Parameters
    ----------
    cost : ndarray (T, D)
        Association cost matrix, modified in place and returned.
    track_mean_xy : ndarray (T, 2)
        Projected center (x, y) of each track's measurement distribution.
    track_S_xy : ndarray (T, 2, 2)
        Projected (x, y) innovation covariance of each track.
    det_xy : ndarray (D, 2)
        Detection centers (x, y).
    thresh : float
        Squared Mahalanobis gating threshold.
    gated : float
        Value written for gated entries (default ``np.inf``).
    """
    for t in range(cost.shape[0]):
        L = np.linalg.cholesky(track_S_xy[t])
        d = (det_xy - track_mean_xy[t]).T
        z = np.linalg.solve(L, d)
        d2 = np.einsum('ij,ij->j', z, z)
        cost[t, d2 > thresh] = gated
    return cost
