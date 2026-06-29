"""Pure step-response metrics (spec §11) over a UNIFORM (t, y) sample array.

Inputs are already resampled onto a uniform grid (see resample.py). All metrics
normalize the response to the commanded step via frac = (y - y0)/(y_final - y0)
so they are sign-agnostic. A degenerate step (|y_final - y0| < EPS) yields
None for time metrics and 0.0 overshoot.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _frac(y: np.ndarray, y0: float, y_final: float) -> np.ndarray | None:
    span = y_final - y0
    if abs(span) < _EPS:
        return None
    return (np.asarray(y, dtype=float) - y0) / span


def _first_cross_time(t: np.ndarray, frac: np.ndarray, level: float) -> float | None:
    """Linear-interpolated time at which frac first reaches `level`."""
    above = frac >= level
    if not above.any():
        return None
    i = int(np.argmax(above))            # first True
    if i == 0:
        return float(t[0])
    f0, f1 = frac[i - 1], frac[i]
    if f1 == f0:
        return float(t[i])
    w = (level - f0) / (f1 - f0)
    return float(t[i - 1] + w * (t[i] - t[i - 1]))


def rise_time(t, y, *, y0: float, y_final: float, lo: float = 0.1, hi: float = 0.9):
    t = np.asarray(t, dtype=float)
    frac = _frac(y, y0, y_final)
    if frac is None:
        return None
    t_lo = _first_cross_time(t, frac, lo)
    t_hi = _first_cross_time(t, frac, hi)
    if t_lo is None or t_hi is None:
        return None
    return t_hi - t_lo


def overshoot_pct(y, *, y0: float, y_final: float) -> float:
    frac = _frac(y, y0, y_final)
    if frac is None:
        return 0.0
    peak = float(np.max(frac))
    return max(0.0, (peak - 1.0) * 100.0)


def settling_time(t, y, *, y0: float, y_final: float, band: float = 0.02):
    t = np.asarray(t, dtype=float)
    frac = _frac(y, y0, y_final)
    if frac is None:
        return None
    outside = np.abs(frac - 1.0) > band
    if not outside.any():
        return 0.0
    last = int(np.max(np.flatnonzero(outside)))
    if last >= len(t) - 1:
        return None                       # still outside the band at the end
    return float(t[last + 1] - t[0])


def dead_time(t, y, *, y0: float, y_final: float, dead_frac: float = 0.05):
    t = np.asarray(t, dtype=float)
    frac = _frac(y, y0, y_final)
    if frac is None:
        return None
    tc = _first_cross_time(t, np.abs(frac), dead_frac)
    if tc is None:
        return None
    return tc - float(t[0])
