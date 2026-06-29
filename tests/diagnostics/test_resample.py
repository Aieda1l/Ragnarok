"""Tests for the jitter->uniform resampler."""
from __future__ import annotations
import numpy as np
import pytest
from ragnarok.diagnostics.resample import resample_uniform


def test_linear_ramp_is_preserved():
    # Jittered sample times; value is an exact linear ramp -> PCHIP reproduces it.
    t_ns = [0, 1_300_000, 2_900_000, 4_100_000, 5_000_000]   # ns, uneven
    vals = [0.0, 1.3, 2.9, 4.1, 5.0]                          # value == t in ms
    t_s, y = resample_uniform(t_ns, vals, hz=2000.0)
    assert t_s[0] == 0.0
    assert abs(t_s[-1] - 0.005) < 1e-9
    # at every uniform grid point, y(ms) ~= t_s*1000
    assert np.allclose(y, t_s * 1000.0, atol=1e-6)


def test_grid_spacing_matches_hz():
    t_ns = [0, 10_000_000]            # 10 ms span
    vals = [0.0, 1.0]
    t_s, y = resample_uniform(t_ns, vals, hz=1000.0)
    assert abs((t_s[1] - t_s[0]) - 0.001) < 1e-9   # 1 kHz -> 1 ms spacing


def test_requires_two_samples():
    with pytest.raises(ValueError):
        resample_uniform([5], [1.0], hz=1000.0)
