"""Closed-loop latency estimation (spec §11).

The aim feedback loop's round-trip latency (send → game render → display →
capture → detect) is the same physical delay that both the Smith-predictor
``aim.deadtime_ms`` and the GMC ``tracking.tau_render_s`` need. Measure it at a
flat wall: command a rich mouse motion and watch the wall's global optical flow
respond — the lag at peak cross-correlation IS the latency. No detection / target
motion is involved, so the signal is clean.

Pure/testable: ``estimate_lag`` cross-correlates the commanded view motion with
the (negated) observed scene flow over candidate lags. The capture is box-only —
see scripts/measure_latency.py.
"""
from __future__ import annotations

import numpy as np


def estimate_lag(commanded, observed, dt_s: float, max_lag_frames: int) -> float | None:
    """Round-trip latency (s) between commanded view motion and observed scene
    flow. ``observed`` is the phase-correlation scene shift (opposite the view),
    so it should match the commanded motion, negated and delayed. Returns None if
    there isn't enough signal to correlate."""
    c = np.asarray(commanded, dtype=float)
    o = -np.asarray(observed, dtype=float)          # scene moves opposite the view
    n = len(c)
    if n < 4 or len(o) != n or dt_s <= 0.0:
        return None
    c = c - c.mean()
    o = o - o.mean()
    best_lag, best_corr = None, 0.0
    for lag in range(0, min(max_lag_frames, n - 2) + 1):
        a, b = c[:n - lag], o[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            continue
        corr = float(np.dot(a, b) / denom)
        if corr > best_corr:                        # want positive correlation (aligned)
            best_corr, best_lag = corr, lag
    if best_lag is None:
        return None
    return best_lag * dt_s
