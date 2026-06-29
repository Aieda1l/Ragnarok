"""Tests for the step-response runner against a synthetic plant (no real IO)."""
from __future__ import annotations
from ragnarok.diagnostics.runner import StepResponseRunner
from ragnarok.diagnostics.plant import AimPlant


class _Clock:
    def __init__(self, dt_ns):
        self.t = 0
        self._dt = dt_ns
    def __call__(self):
        v = self.t
        self.t += self._dt          # each read advances time by one tick
        return v


def test_runner_characterizes_first_order_plant():
    # A first-order actuator-lag plant fed a single step is a first-order response.
    dt = 0.001
    plant = AimPlant(dt_s=dt, lag_tau_s=0.05)
    pos = {"v": 0.0}

    def move(dx, dy):
        # the runner injects the step once; the plant integrates it over time
        move.cmd = dx
    move.cmd = 0.0

    def sample():
        pos["v"] = plant.step(move.cmd)
        move.cmd = 0.0              # the step is a one-shot impulse of size step_px
        return pos["v"]

    clk = _Clock(int(dt * 1e9))
    runner = StepResponseRunner(move=move, sample=sample, clock=clk,
                                step_px=10.0, hz=1000.0, timeout_s=1.0)
    res = runner.run()
    assert res.y0 == 0.0
    assert res.y_final == 10.0
    assert res.rise_s is not None and res.rise_s > 0.0
    assert res.overshoot_pct < 1.0
