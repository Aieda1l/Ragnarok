"""Transport-agnostic step-response runner (spec §11 modes a/b).

Inject a known step via move(dx,dy), poll sample()->position until timeout, then
compute the step-response metrics. The move/sample/clock seams keep it CI-safe:
tests drive a synthetic AimPlant; live desktop (GetCursorPos) and in-game
(detector-as-sensor) modes pass real callables (box-only smokes).
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.diagnostics.results import StepResponseRecorder, compute_step_result


class StepResponseRunner:
    def __init__(self, *, move, sample, clock=now_ns, step_px: float, hz: float,
                 timeout_s: float, axis: str = "x", band: float = 0.02,
                 rise_lo: float = 0.1, rise_hi: float = 0.9, dead_frac: float = 0.05) -> None:
        self._move = move
        self._sample = sample
        self._clock = clock
        self._step = step_px
        self._hz = hz
        self._timeout_ns = int(timeout_s * 1e9)
        self._axis = axis
        self._band = band
        self._rise_lo = rise_lo
        self._rise_hi = rise_hi
        self._dead_frac = dead_frac

    def run(self):
        rec = StepResponseRecorder(clock=self._clock)
        y0 = float(self._sample())
        if self._axis == "y":
            self._move(0.0, self._step)
        else:
            self._move(self._step, 0.0)
        t_start = self._clock()
        while True:
            rec.record(self._sample())
            if self._clock() - t_start >= self._timeout_ns:
                break
        t_ns, vals = rec.samples()
        return compute_step_result(
            t_ns, vals, y0=y0, y_final=y0 + self._step, hz=self._hz,
            band=self._band, rise_lo=self._rise_lo, rise_hi=self._rise_hi,
            dead_frac=self._dead_frac,
        )
