"""Relay-feedback (Åström-Hägglund) limit-cycle analysis + ZN seeding (spec §11).

ku_from_relay: Ku = 4d/(π·a) from relay amplitude d and limit-cycle amplitude a.
zn_seed: Ziegler-Nichols gains; default 'low_overshoot' (Pessen-style) for the
precision-aim loop. These are SEEDS for tuning, never final values (spec §11).
"""
from __future__ import annotations

import math

import numpy as np


def _peaks(y: np.ndarray) -> np.ndarray:
    """Indices of local maxima (strict interior)."""
    return np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1


def limit_cycle(t, y) -> tuple[float, float]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    half = len(y) // 2                       # analyze the steady tail
    tail_y = y[half:]
    tail_t = t[half:]
    amplitude = (float(np.max(tail_y)) - float(np.min(tail_y))) / 2.0
    pk = _peaks(tail_y)
    if pk.size >= 2:
        period = float(np.median(np.diff(tail_t[pk])))
    else:
        period = 0.0
    return amplitude, period


def ku_from_relay(d: float, a: float) -> float:
    # Describing-function ultimate gain for an ideal (zero-hysteresis) relay.
    # With relay hysteresis eps > 0 the correct form is 4d/(pi*sqrt(a**2 - eps**2));
    # callers using hysteresis must account for it (run_relay_tune defaults eps=0).
    return 4.0 * d / (math.pi * a)


def zn_seed(Ku: float, Tu: float, *, rule: str = "low_overshoot") -> dict[str, float]:
    if rule == "classic":
        return {"kp": 0.6 * Ku, "ki": 1.2 * Ku / Tu, "kd": 0.075 * Ku * Tu}
    if rule == "pi":
        return {"kp": 0.45 * Ku, "ki": 0.54 * Ku / Tu, "kd": 0.0}
    if rule == "low_overshoot":
        return {"kp": 0.2 * Ku, "ki": 0.4 * Ku / Tu, "kd": 0.066 * Ku * Tu}
    raise ValueError(f"unknown ZN rule {rule!r}")
