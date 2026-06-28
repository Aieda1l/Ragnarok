"""Tests for VelocitySmoother EMA + clamp behaviour."""
from __future__ import annotations

import math
from ragnarok.aim.velocity import VelocitySmoother


def test_first_call_seeds_then_clamps():
    s = VelocitySmoother(alpha=0.5, max_px_s=100.0)
    vx, vy = s.smooth_clamp(50.0, 0.0)   # first call seeds from raw
    assert abs(vx - 50.0) < 1e-9 and vy == 0.0


def test_lowpass_damps_step():
    s = VelocitySmoother(alpha=0.5, max_px_s=1e9)
    s.smooth_clamp(0.0, 0.0)             # seed at 0
    vx, vy = s.smooth_clamp(100.0, 0.0)  # ema = 0 + 0.5*(100-0) = 50
    assert abs(vx - 50.0) < 1e-9


def test_magnitude_clamp():
    s = VelocitySmoother(alpha=1.0, max_px_s=100.0)
    vx, vy = s.smooth_clamp(300.0, 400.0)  # raw mag 500 -> clamp to 100
    assert abs(math.hypot(vx, vy) - 100.0) < 1e-6


def test_reset_reseeds():
    s = VelocitySmoother(alpha=0.5, max_px_s=1e9)
    s.smooth_clamp(0.0, 0.0)
    s.reset()
    vx, vy = s.smooth_clamp(80.0, 0.0)   # reseeds from raw after reset
    assert abs(vx - 80.0) < 1e-9
