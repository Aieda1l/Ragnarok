"""Tests for StepResponseResult, recorder, and compute_step_result."""
from __future__ import annotations
import numpy as np
from ragnarok.diagnostics.results import (
    StepResponseRecorder, compute_step_result, StepResponseResult,
)


class _Clock:
    def __init__(self):
        self.t = 0
    def __call__(self):
        return self.t


def test_recorder_stamps_and_returns_samples():
    clk = _Clock()
    rec = StepResponseRecorder(clock=clk)
    clk.t = 1_000_000; rec.record(0.0)
    clk.t = 2_000_000; rec.record(1.0)
    t_ns, vals = rec.samples()
    assert list(t_ns) == [1_000_000, 2_000_000]
    assert list(vals) == [0.0, 1.0]
    rec.reset()
    assert rec.samples()[0].size == 0


def test_compute_step_result_on_first_order():
    tau, dt = 0.1, 0.0005
    t = np.arange(0.0, 2.0, dt)
    y = 1.0 - np.exp(-t / tau)
    t_ns = (t * 1e9).astype(np.int64)
    res = compute_step_result(t_ns, y, y0=0.0, y_final=1.0, hz=2000.0)
    assert isinstance(res, StepResponseResult)
    assert abs(res.rise_s - 0.2197) < 1e-2
    assert res.overshoot_pct < 0.5
    assert res.settling_s is not None
