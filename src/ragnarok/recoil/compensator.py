"""Recoil compensation (spec §6.6).

A per-weapon cumulative spray pattern (px crosshair drift per shot). The
compensator emits the per-shot counter-move and advances one entry per shot,
resetting on fire-release. (The wall-learner that *learns* the pattern and the
fold-into-ego-motion path are later phases; a hand-authored table works now.)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoilPattern:
    """Cumulative (dx, dy) crosshair drift in px, indexed by shot number."""

    points: tuple[tuple[float, float], ...]


class RecoilCompensator:
    def __init__(self, pattern: RecoilPattern, *, scale: float = 1.0,
                 fire_rate_rps: float = 0.0) -> None:
        self._pts = pattern.points
        self._scale = scale
        self.fire_rate_rps = fire_rate_rps    # >0: advance per-shot while held (full-auto)
        self._idx = 0

    def on_fire(self) -> tuple[float, float]:
        i = self._idx
        if i >= len(self._pts):
            self._idx += 1
            return (0.0, 0.0)
        cx, cy = self._pts[i]
        if i == 0:
            px, py = 0.0, 0.0
        else:
            px, py = self._pts[i - 1]
        self._idx += 1
        return (-(cx - px) * self._scale, -(cy - py) * self._scale)

    def release(self) -> None:
        self._idx = 0
