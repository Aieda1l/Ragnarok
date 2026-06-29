"""Resample jittery (t_ns, value) samples onto a uniform timeline.

perf_counter_ns sample times are uneven and the mouse driver quantizes motion to
integer px, so step-response crossings/settling must be read off a uniform grid.
Uses monotone PCHIP (spec §6.6: PCHIP, not natural cubic which overshoots).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def resample_uniform(t_ns, values, *, hz: float) -> tuple[np.ndarray, np.ndarray]:
    t_ns = np.asarray(t_ns, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if t_ns.size < 2:
        raise ValueError("resample_uniform needs at least 2 samples")
    t_s = (t_ns - t_ns[0]) / 1e9                  # seconds relative to first sample
    span = float(t_s[-1])
    n = max(2, int(round(span * hz)) + 1)
    grid = np.linspace(0.0, span, n)
    pchip = PchipInterpolator(t_s, values, extrapolate=False)
    y = pchip(grid)
    return grid, y
