"""Feed-forward velocity conditioning (spec §6.4 anti-runaway guards).

The Kff*v̂ feed-forward term amplifies any noise/residual ego-motion in the
velocity estimate. Two guards before it reaches the aimer:
  * low-pass (EMA) smoothing  -> damps high-frequency gain spikes
  * magnitude clamp           -> a bad estimate can't drive a runaway
"""
from __future__ import annotations

import math


class VelocitySmoother:
    def __init__(self, *, alpha: float = 0.5, max_px_s: float = 4000.0) -> None:
        self._alpha = alpha
        self._max = max_px_s
        self._vx = 0.0
        self._vy = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False

    def smooth_clamp(self, vx: float, vy: float) -> tuple[float, float]:
        if not self._initialized:
            self._vx, self._vy = vx, vy
            self._initialized = True
        else:
            a = self._alpha
            self._vx += a * (vx - self._vx)
            self._vy += a * (vy - self._vy)
        ox, oy = self._vx, self._vy
        mag = math.hypot(ox, oy)
        if mag > self._max and mag > 0.0:
            s = self._max / mag
            ox *= s
            oy *= s
        return (ox, oy)
