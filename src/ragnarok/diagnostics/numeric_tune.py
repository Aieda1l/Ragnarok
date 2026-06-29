"""Numeric PID tuning via Nelder-Mead over the synthetic plant (spec §11).

Deterministic: each evaluation builds a fresh FeedbackAimer (PID) + a fresh
plant and scores the closed loop with itae_cost. Returns SEEDS, not final gains.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ragnarok.aim.aimers import FeedbackAimer
from ragnarok.diagnostics.plant import simulate_closed_loop
from ragnarok.diagnostics.cost import itae_cost


@dataclass(frozen=True)
class PidSeeds:
    kp: float
    ki: float
    kd: float


def numeric_tune(plant_factory, *, seed: PidSeeds, setpoint: float, n_steps: int,
                 dt_s: float, max_step_px: float = 1e9, w_overshoot: float = 1.0,
                 w_effort: float = 0.0) -> PidSeeds:
    def cost(theta) -> float:
        kp, ki, kd = (max(0.0, float(v)) for v in theta)   # gains are non-negative
        aimer = FeedbackAimer(kp=kp, ki=ki, kd=kd, max_step_px=max_step_px, ema_alpha=1.0)
        t, m, u = simulate_closed_loop(
            lambda e, dt: aimer.step((0.0, 0.0), (e, 0.0), dt)[0],
            plant_factory(), setpoint=setpoint, n_steps=n_steps, dt_s=dt_s,
        )
        return itae_cost(t, m, u, setpoint=setpoint, w_overshoot=w_overshoot, w_effort=w_effort)

    x0 = np.array([seed.kp, seed.ki, seed.kd], dtype=float)
    res = minimize(cost, x0, method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})
    kp, ki, kd = (max(0.0, float(v)) for v in res.x)
    return PidSeeds(kp=kp, ki=ki, kd=kd)
