"""Tests for the aim plant integrator + closed-loop simulator."""
from __future__ import annotations
import numpy as np
from ragnarok.diagnostics.plant import AimPlant, simulate_closed_loop


def test_integrator_accumulates_commands():
    p = AimPlant(dt_s=0.01)             # pure integrator, gain 1
    assert p.step(5.0) == 5.0
    assert p.step(3.0) == 8.0


def test_dead_time_delays_response():
    p = AimPlant(dt_s=0.01, dead_time_s=0.02)   # 2-tick delay
    assert p.step(10.0) == 0.0          # tick 1: still delayed
    assert p.step(0.0) == 0.0           # tick 2: still delayed
    assert p.step(0.0) == 10.0          # tick 3: the first command lands


def test_p_controller_on_integrator_is_first_order():
    # error -> command = kp*error; integrator closed loop: m_{k+1}=m_k+kp*(sp-m_k)
    # -> geometric approach to sp with ratio (1-kp).
    p = AimPlant(dt_s=0.01)
    kp = 0.3
    t, m, u = simulate_closed_loop(lambda e, dt: kp * e, p, setpoint=1.0,
                                   n_steps=200, dt_s=0.01)
    assert abs(m[-1] - 1.0) < 1e-3          # converges to setpoint
    assert m[0] == 0.3                      # first step = kp*1.0
    assert np.all(np.diff(m) >= -1e-9)      # monotone (no overshoot for P on integrator)
