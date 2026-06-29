"""Pure feed-forward-GMC calibration solvers (spec §5.3, §18).

estimate_tau_render: the render+display latency by which the on-screen response
trails a commanded motion, found by cross-correlating the commanded signal with
the measured global-motion (optical-flow) signal on a common uniform grid. The
live optical-flow capture that produces `measured` is a box-only smoke; this
function (the analysis) is pure and unit-tested.
"""
from __future__ import annotations

import numpy as np


def estimate_tau_render(commanded, measured, dt_s: float, *, max_lag_s: float = 0.1) -> float:
    c = np.asarray(commanded, dtype=float)
    m = np.asarray(measured, dtype=float)
    c = c - c.mean()
    m = m - m.mean()
    # Full cross-correlation; positive lag = measured trails commanded.
    corr = np.correlate(m, c, mode="full")
    lags = np.arange(-(len(c) - 1), len(m))          # sample lags aligned with corr
    max_lag = int(round(max_lag_s / dt_s))
    keep = (lags >= 0) & (lags <= max_lag)           # render latency is non-negative
    if not keep.any():
        return 0.0
    best = lags[keep][int(np.argmax(corr[keep]))]
    return float(best) * dt_s
