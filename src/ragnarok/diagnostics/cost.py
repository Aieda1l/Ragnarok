"""Control-tuning cost: ITAE + overshoot + effort penalties (spec §11)."""
from __future__ import annotations

import numpy as np

from ragnarok.diagnostics.metrics import overshoot_pct


def itae_cost(t, measured, command, *, setpoint: float, y0: float = 0.0,
              w_overshoot: float = 1.0, w_effort: float = 0.0) -> float:
    t = np.asarray(t, dtype=float)
    measured = np.asarray(measured, dtype=float)
    command = np.asarray(command, dtype=float)
    dt = float(t[1] - t[0]) if t.size > 1 else 0.0
    itae = float(np.sum(t * np.abs(setpoint - measured) * dt))
    os = overshoot_pct(measured, y0=y0, y_final=setpoint)
    effort = float(np.sum(np.abs(command)))
    return itae + w_overshoot * os + w_effort * effort
