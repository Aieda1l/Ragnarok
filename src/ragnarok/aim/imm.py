"""Per-track IMM (CV + CA) motion filter for the aim target.

A 2-model Interacting Multiple Model estimator (filterpy) over a shared
6-state layout ``[x, vx, ax, y, vy, ay]``: a low-process-noise constant-velocity
model and a high-process-noise constant-acceleration model. The IMM blends them
by mode probability, so it tracks smooth motion *and* snaps to jukes.

Phase 3 runs in pixel space (identity ego-motion). ``lead(t)`` extrapolates the
fused estimate forward by ``pos + v*t + 0.5*a*t^2`` for predictive aim.
``IMMManager`` keeps one filter per ``track_id`` and prunes dead ids.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise

STATE = 6  # [x, vx, ax, y, vy, ay]
_H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=float)

# Process-noise spectral variance (px^2/s): CV tight, CA loose.
CV_VAR = 5.0
CA_VAR = 4000.0
R_PX = 4.0      # measurement noise (px std ~2 -> var 4)
P0 = 1000.0     # initial covariance


def _F(dt: float, accel: bool) -> np.ndarray:
    """6x6 transition. CV: accel decoupled and decaying; CA: full kinematics."""
    if accel:
        blk = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
    else:
        blk = np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    return block_diag(blk, blk)


def _Q(dt: float, var: float) -> np.ndarray:
    # dim=3 derivatives/axis, block_size=2 axes, contiguous per-axis blocks.
    return Q_discrete_white_noise(dim=3, dt=dt, var=var, block_size=2, order_by_dim=True)


def _make_kf(accel: bool, var: float, z0: tuple[float, float], dt: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=STATE, dim_z=2)
    kf.x = np.array([z0[0], 0.0, 0.0, z0[1], 0.0, 0.0], dtype=float)
    kf.H = _H.copy()
    kf.R = np.eye(2) * R_PX
    kf.P = np.eye(STATE) * P0
    kf.F = _F(dt, accel)
    kf.Q = _Q(dt, var)
    return kf


class TrackIMM:
    def __init__(self, x0: float, y0: float, dt: float = 1 / 144.0) -> None:
        cv = _make_kf(False, CV_VAR, (x0, y0), dt)
        ca = _make_kf(True, CA_VAR, (x0, y0), dt)
        M = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.imm = IMMEstimator([cv, ca], np.array([0.5, 0.5]), M)

    def update(self, x: float, y: float, dt: float) -> None:
        cv, ca = self.imm.filters
        cv.F, ca.F = _F(dt, False), _F(dt, True)
        cv.Q, ca.Q = _Q(dt, CV_VAR), _Q(dt, CA_VAR)
        self.imm.predict()
        self.imm.update(np.array([x, y], dtype=float))

    def position(self) -> tuple[float, float]:
        s = np.asarray(self.imm.x).ravel()
        return (float(s[0]), float(s[3]))

    def velocity(self) -> tuple[float, float]:
        s = np.asarray(self.imm.x).ravel()
        return (float(s[1]), float(s[4]))

    def lead(self, t_lead: float) -> tuple[float, float]:
        s = np.asarray(self.imm.x).ravel()
        px = s[0] + s[1] * t_lead + 0.5 * s[2] * t_lead * t_lead
        py = s[3] + s[4] * t_lead + 0.5 * s[5] * t_lead * t_lead
        return (float(px), float(py))

    @property
    def mode_probs(self) -> tuple[float, float]:
        mu = np.asarray(self.imm.mu).ravel()
        return (float(mu[0]), float(mu[1]))

    def inflate(self, factor: float = 100.0) -> None:
        """Inflate covariance + reset mode probs on re-acquire after a gap."""
        for f in self.imm.filters:
            f.P *= factor
        self.imm.mu = np.array([0.5, 0.5])


class IMMManager:
    """Per-track-id IMM book; create on first sight, prune dead ids each frame."""

    def __init__(self, default_dt: float = 1 / 144.0) -> None:
        self._dt = default_dt
        self._imms: dict[int, TrackIMM] = {}

    def update(self, track_id: int, x: float, y: float, dt: float) -> None:
        if track_id not in self._imms:
            self._imms[track_id] = TrackIMM(x, y, dt)
        else:
            self._imms[track_id].update(x, y, dt)

    def lead(self, track_id: int, t_lead: float) -> tuple[float, float]:
        return self._imms[track_id].lead(t_lead)

    def position(self, track_id: int) -> tuple[float, float]:
        return self._imms[track_id].position()

    def velocity(self, track_id: int) -> tuple[float, float]:
        return self._imms[track_id].velocity()

    def mode_probs(self, track_id: int) -> tuple[float, float]:
        return self._imms[track_id].mode_probs

    def prune(self, live_ids) -> None:
        for tid in [t for t in self._imms if t not in live_ids]:
            del self._imms[tid]

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._imms
