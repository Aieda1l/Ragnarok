"""Step-response result object, sample recorder, and the compute helper."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragnarok.core.clock import now_ns
from ragnarok.diagnostics.resample import resample_uniform
from ragnarok.diagnostics import metrics


@dataclass(frozen=True, eq=False)
class StepResponseResult:
    rise_s: float | None
    overshoot_pct: float
    settling_s: float | None
    dead_time_s: float | None
    t_s: np.ndarray
    y: np.ndarray
    y0: float
    y_final: float


class StepResponseRecorder:
    """Accumulates (clock(), value) samples during a step-response run."""

    def __init__(self, *, clock=now_ns) -> None:
        self._clock = clock
        self._t: list[int] = []
        self._v: list[float] = []

    def record(self, value: float) -> None:
        self._t.append(int(self._clock()))
        self._v.append(float(value))

    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self._t, dtype=np.int64), np.asarray(self._v, dtype=float))

    def reset(self) -> None:
        self._t.clear()
        self._v.clear()


def compute_step_result(
    t_ns, values, *, y0: float, y_final: float, hz: float,
    band: float = 0.02, rise_lo: float = 0.1, rise_hi: float = 0.9,
    dead_frac: float = 0.05,
) -> StepResponseResult:
    t_s, y = resample_uniform(t_ns, values, hz=hz)
    return StepResponseResult(
        rise_s=metrics.rise_time(t_s, y, y0=y0, y_final=y_final, lo=rise_lo, hi=rise_hi),
        overshoot_pct=metrics.overshoot_pct(y, y0=y0, y_final=y_final),
        settling_s=metrics.settling_time(t_s, y, y0=y0, y_final=y_final, band=band),
        dead_time_s=metrics.dead_time(t_s, y, y0=y0, y_final=y_final, dead_frac=dead_frac),
        t_s=t_s, y=y, y0=y0, y_final=y_final,
    )
