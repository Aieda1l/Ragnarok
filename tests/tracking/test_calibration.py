"""Tests for the pure GMC calibration solvers."""
from __future__ import annotations
import numpy as np
from ragnarok.tracking.calibration import estimate_tau_render


def test_recovers_known_lag():
    dt = 0.001
    n = 500
    rng = np.zeros(n)
    rng[100:140] = 1.0                       # a commanded pulse
    commanded = rng
    measured = np.zeros(n)
    measured[115:155] = 1.0                  # same pulse delayed by 15 samples = 15 ms
    lag = estimate_tau_render(commanded, measured, dt, max_lag_s=0.1)
    assert abs(lag - 0.015) < 1.5e-3


def test_zero_lag_when_aligned():
    dt = 0.001
    x = np.zeros(200)
    x[50:90] = 1.0
    lag = estimate_tau_render(x, x.copy(), dt)
    assert abs(lag) < 1e-9


def test_lag_capped_at_max():
    dt = 0.001
    n = 400
    commanded = np.zeros(n); commanded[10:30] = 1.0
    measured = np.zeros(n); measured[300:320] = 1.0   # 290 ms apart, beyond max
    lag = estimate_tau_render(commanded, measured, dt, max_lag_s=0.05)
    assert lag <= 0.05 + 1e-9
