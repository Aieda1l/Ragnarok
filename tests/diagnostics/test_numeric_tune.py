"""Tests for the ITAE cost and the Nelder-Mead numeric tuner."""
from __future__ import annotations
import numpy as np
from ragnarok.diagnostics.cost import itae_cost
from ragnarok.diagnostics.numeric_tune import numeric_tune, PidSeeds
from ragnarok.diagnostics.plant import AimPlant


def test_itae_penalizes_slow_response():
    t = np.linspace(0, 1, 100)
    fast = np.ones_like(t)               # at setpoint immediately
    slow = np.linspace(0, 1, 100)        # ramps up slowly
    cmd = np.zeros_like(t)
    c_fast = itae_cost(t, fast, cmd, setpoint=1.0)
    c_slow = itae_cost(t, slow, cmd, setpoint=1.0)
    assert c_fast < c_slow


def test_numeric_tune_lowers_cost_vs_seed():
    def plant_factory():
        return AimPlant(dt_s=0.005, lag_tau_s=0.03, dead_time_s=0.01)

    seed = PidSeeds(kp=0.05, ki=0.0, kd=0.0)
    tuned = numeric_tune(plant_factory, seed=seed, setpoint=100.0,
                         n_steps=400, dt_s=0.005)
    assert isinstance(tuned, PidSeeds)

    from ragnarok.diagnostics.plant import simulate_closed_loop
    from ragnarok.aim.aimers import FeedbackAimer

    def score(s):
        a = FeedbackAimer(kp=s.kp, ki=s.ki, kd=s.kd, max_step_px=1e9, ema_alpha=1.0)
        t, m, u = simulate_closed_loop(lambda e, dt: a.step((0, 0), (e, 0), dt)[0],
                                       plant_factory(), setpoint=100.0,
                                       n_steps=400, dt_s=0.005)
        return itae_cost(t, m, u, setpoint=100.0)

    assert score(tuned) <= score(seed) + 1e-9    # tuning did not worsen the loop
    assert tuned.kp >= 0.0 and tuned.ki >= 0.0 and tuned.kd >= 0.0
