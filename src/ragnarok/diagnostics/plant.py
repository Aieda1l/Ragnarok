"""Deterministic aim-plant model + closed-loop simulator (CI-safe, no IO).

The aim 'plant' is fundamentally an INTEGRATOR: each commanded mouse delta moves
the crosshair, so position += gain*command. Optional first-order actuator lag and
pure dead-time make it a more realistic FOPDT-ish plant for tuner coverage.
This lets the controller/auto-tuners be characterized entirely off-box.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class AimPlant:
    def __init__(self, *, gain: float = 1.0, lag_tau_s: float = 0.0,
                 dead_time_s: float = 0.0, dt_s: float) -> None:
        self._gain = gain
        self._dt = dt_s
        self._lag_tau = lag_tau_s
        self._pos = 0.0
        self._lagged = 0.0                       # actuator-lag state
        delay = max(0, int(round(dead_time_s / dt_s)))
        self._delay = deque([0.0] * (delay + 1), maxlen=delay + 1) if delay else None

    @property
    def position(self) -> float:
        return self._pos

    def reset(self) -> None:
        self._pos = 0.0
        self._lagged = 0.0
        if self._delay is not None:
            self._delay = deque([0.0] * self._delay.maxlen, maxlen=self._delay.maxlen)

    def step(self, command: float) -> float:
        u = command
        if self._delay is not None:              # pure dead-time
            self._delay.append(u)
            u = self._delay[0]
        if self._lag_tau > 0.0:                  # first-order actuator lag
            a = self._dt / (self._lag_tau + self._dt)
            self._lagged += a * (u - self._lagged)
            u = self._lagged
        self._pos += self._gain * u              # integrator
        return self._pos


def simulate_closed_loop(controller_step, plant: AimPlant, *, setpoint: float,
                         n_steps: int, dt_s: float, y0: float = 0.0):
    plant.reset()
    measured = y0
    t_s = np.empty(n_steps)
    m_arr = np.empty(n_steps)
    u_arr = np.empty(n_steps)
    for k in range(n_steps):
        error = setpoint - measured
        command = controller_step(error, dt_s)
        measured = y0 + plant.step(command)
        t_s[k] = k * dt_s
        m_arr[k] = measured
        u_arr[k] = command
    return t_s, m_arr, u_arr
