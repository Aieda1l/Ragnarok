"""Tests for NullShaper and WindMouseShaper motion-shaping behaviour."""
from __future__ import annotations

import math
import random
from ragnarok.motion.shaper import NullShaper, WindMouseShaper


def test_null_shaper_is_identity():
    s = NullShaper()
    assert s.shape(3.0, -4.0) == (3.0, -4.0)


def test_windmouse_zero_delta_is_zero():
    s = WindMouseShaper(rng=random.Random(0))
    assert s.shape(0.0, 0.0) == (0.0, 0.0)


def test_windmouse_never_overshoots_single_step():
    s = WindMouseShaper(max_step=15.0, rng=random.Random(1))
    dx, dy = s.shape(5.0, 0.0)               # target only 5 px away this frame
    assert math.hypot(dx, dy) <= 5.0 + 1e-9


def test_windmouse_converges_to_fixed_target():
    # Repeatedly feed the remaining vector to a fixed destination; it should arrive.
    s = WindMouseShaper(gravity=9.0, wind=3.0, max_step=15.0, rng=random.Random(7))
    x, y = 0.0, 0.0
    dest = (300.0, 120.0)
    for _ in range(2000):
        dx, dy = s.shape(dest[0] - x, dest[1] - y)
        x += dx
        y += dy
        if math.hypot(dest[0] - x, dest[1] - y) < 1.0:
            break
    assert math.hypot(dest[0] - x, dest[1] - y) < 2.0


def test_windmouse_is_deterministic_with_seed():
    a = WindMouseShaper(rng=random.Random(42))
    b = WindMouseShaper(rng=random.Random(42))
    assert a.shape(100.0, 50.0) == b.shape(100.0, 50.0)


def test_windmouse_reset_clears_momentum():
    s = WindMouseShaper(rng=random.Random(3))
    s.shape(100.0, 0.0)
    s.reset()
    # after reset the internal velocity/wind are zero again; first step is small
    dx, dy = s.shape(1.0, 0.0)
    assert math.hypot(dx, dy) <= 1.0 + 1e-9
