"""Ego-motion (camera/global motion) providers.

The tracker consumes a 2x3 affine warp per frame. Phase 2 uses identity (no
compensation); Phase 3/4 can drop in a feed-forward GMC provider that returns a
real estimate without changing the tracker.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ragnarok.aim.fov import focal_length_px


class EgoMotion(ABC):
    """Estimates a 2x3 affine warp mapping previous-frame coords to current."""

    @abstractmethod
    def estimate(self, frame) -> np.ndarray:
        """Return a (2, 3) float32 affine for ``frame``."""
        raise NotImplementedError


class IdentityEgoMotion(EgoMotion):
    """No-op provider: always returns the identity affine."""

    def estimate(self, frame) -> np.ndarray:
        return np.eye(2, 3, dtype=np.float32)


class CommandedMotionBuffer:
    """Timestamped ring buffer of commanded mouse-count deltas (spec §5.3).

    The controller pushes each frame's commanded (dx, dy) in mouse counts;
    FeedForwardGMC integrates them over the render-time window. A passthrough
    (physical-mouse) source can push into the same buffer in Phase 7.
    """

    def __init__(self, maxlen: int = 4096) -> None:
        self._buf: deque[tuple[int, float, float]] = deque(maxlen=maxlen)

    def push(self, t_ns: int, d_counts_x: float, d_counts_y: float) -> None:
        self._buf.append((t_ns, d_counts_x, d_counts_y))

    def integrate(self, t_lo_ns: int, t_hi_ns: int) -> tuple[float, float]:
        sx = 0.0
        sy = 0.0
        for t, dx, dy in self._buf:
            if t_lo_ns <= t <= t_hi_ns:
                sx += dx
                sy += dy
        return (sx, sy)


class FeedForwardGMC(EgoMotion):
    """Back-projects known camera motion into a 2x3 affine (spec §5.3).

    Instead of CV optical-flow GMC, integrate the commanded (and, later,
    passthrough) mouse counts over the tau_render-aligned window and convert to
    a pixel translation via the pinhole model. R = I for pure yaw/pitch.
    """

    def __init__(
        self,
        *,
        hfov_deg: float,
        screen_width_px: int,
        deg_per_count: float,
        tau_render_s: float = 0.0,
        frame_dt_s: float = 1.0 / 144.0,
        buffer: CommandedMotionBuffer | None = None,
    ) -> None:
        self._f = focal_length_px(hfov_deg, screen_width_px)
        self._deg_per_count = deg_per_count
        self._tau = tau_render_s
        self._frame_dt = frame_dt_s
        self.buffer = buffer if buffer is not None else CommandedMotionBuffer()

    def estimate(self, frame) -> np.ndarray:
        t_cap = getattr(frame, "t_capture_ns", None)
        if t_cap is None:
            return np.eye(2, 3, dtype=np.float32)
        hi = int(t_cap - self._tau * 1e9)
        lo = int(hi - self._frame_dt * 1e9)
        dcx, dcy = self.buffer.integrate(lo, hi)
        yaw = math.radians(dcx * self._deg_per_count)
        pitch = math.radians(dcy * self._deg_per_count)
        tx = -self._f * math.tan(yaw)
        ty = -self._f * math.tan(pitch)
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
