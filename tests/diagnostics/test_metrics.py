"""Tests for pure step-response metrics against analytic responses."""
from __future__ import annotations
import math
import numpy as np
from ragnarok.diagnostics.metrics import rise_time, overshoot_pct, settling_time, dead_time


def _first_order(tau=0.1, dt=0.0005, t_end=2.0, K=1.0):
    t = np.arange(0.0, t_end, dt)
    y = K * (1.0 - np.exp(-t / tau))
    return t, y


def test_first_order_rise_time():
    t, y = _first_order(tau=0.1)
    r = rise_time(t, y, y0=0.0, y_final=1.0)
    assert abs(r - 0.1 * math.log(9.0)) < 5e-3   # tau*ln(9) ~= 0.2197 s


def test_first_order_no_overshoot():
    t, y = _first_order(tau=0.1)
    assert overshoot_pct(y, y0=0.0, y_final=1.0) < 0.5


def test_first_order_settling_2pct():
    t, y = _first_order(tau=0.1)
    s = settling_time(t, y, y0=0.0, y_final=1.0, band=0.02)
    assert abs(s - 0.1 * math.log(50.0)) < 1e-2   # tau*ln(50) ~= 0.3912 s


def test_second_order_overshoot_matches_zeta():
    # Underdamped 2nd-order step: overshoot% = 100*exp(-pi*z/sqrt(1-z^2))
    z, wn = 0.5, 30.0
    wd = wn * math.sqrt(1 - z * z)
    t = np.arange(0.0, 1.0, 0.0002)
    y = 1.0 - np.exp(-z * wn * t) * (np.cos(wd * t) + (z * wn / wd) * np.sin(wd * t))
    expected = 100.0 * math.exp(-math.pi * z / math.sqrt(1 - z * z))   # ~16.3 %
    assert abs(overshoot_pct(y, y0=0.0, y_final=1.0) - expected) < 1.5


def test_never_settles_returns_none():
    t = np.arange(0.0, 1.0, 0.001)
    y = 1.0 + 0.5 * np.sin(50.0 * t)        # oscillates forever outside the band
    assert settling_time(t, y, y0=0.0, y_final=1.0, band=0.02) is None


def test_zero_step_is_guarded():
    t = np.arange(0.0, 1.0, 0.001)
    y = np.zeros_like(t)
    assert rise_time(t, y, y0=0.0, y_final=0.0) is None
    assert overshoot_pct(y, y0=0.0, y_final=0.0) == 0.0
