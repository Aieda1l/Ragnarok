"""Tests for relay limit-cycle analysis + ZN seeding (pure)."""
from __future__ import annotations
import math
import numpy as np
from ragnarok.diagnostics.relay import limit_cycle, ku_from_relay, zn_seed


def test_limit_cycle_amplitude_and_period():
    T = 0.2
    t = np.arange(0.0, 2.0, 0.001)
    y = 1.0 + 3.0 * np.sin(2 * math.pi * t / T)   # amplitude 3, period 0.2
    a, Tu = limit_cycle(t, y)
    assert abs(a - 3.0) < 0.1
    assert abs(Tu - 0.2) < 0.01


def test_ku_from_relay_formula():
    assert abs(ku_from_relay(d=1.0, a=2.0) - (4.0 / (math.pi * 2.0))) < 1e-9


def test_zn_seed_classic():
    s = zn_seed(Ku=10.0, Tu=0.5, rule="classic")
    assert abs(s["kp"] - 6.0) < 1e-9            # 0.6*Ku
    assert abs(s["ki"] - 24.0) < 1e-9           # 1.2*Ku/Tu
    assert abs(s["kd"] - 0.375) < 1e-9          # 0.075*Ku*Tu


def test_zn_seed_low_overshoot_is_gentler_than_classic():
    classic = zn_seed(Ku=10.0, Tu=0.5, rule="classic")
    low = zn_seed(Ku=10.0, Tu=0.5, rule="low_overshoot")
    assert low["kp"] < classic["kp"]            # precision loop -> less aggressive
