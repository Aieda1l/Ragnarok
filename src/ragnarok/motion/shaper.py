"""Motion shaping (spec §6.3 Layer B): how the cursor travels to its target.

A MotionShaper transforms the aimer's per-frame pixel delta into a (possibly
reshaped) delta, adding human-like curvature/tremor without overshooting the
per-frame target. WindMouseShaper is a per-frame adaptation of ben.land's
WindMouse: it carries momentum + a random "wind" force between frames and is
pulled toward the current target by gravity. RNG is injected for determinism.
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)


class MotionShaper(ABC):
    @abstractmethod
    def shape(self, dx: float, dy: float) -> tuple[float, float]:
        """Return the reshaped (dx, dy) for this frame's commanded delta."""

    def reset(self) -> None:
        """Clear internal state (called on disengage / target switch)."""


class NullShaper(MotionShaper):
    """Pass-through shaper: raw deltas (no humanization)."""

    def shape(self, dx: float, dy: float) -> tuple[float, float]:
        return (dx, dy)


class WindMouseShaper(MotionShaper):
    """Per-frame WindMouse shaper: momentum + wind + gravity for human-like curves."""

    def __init__(
        self,
        *,
        gravity: float = 9.0,
        wind: float = 3.0,
        max_step: float = 15.0,
        target_area: float = 10.0,
        rng: random.Random | None = None,
    ) -> None:
        self._g = gravity
        self._w = wind
        self._max = max_step
        self._area = target_area
        self._rng = rng if rng is not None else random.Random()
        self._vx = 0.0
        self._vy = 0.0
        self._wx = 0.0
        self._wy = 0.0

    def reset(self) -> None:
        self._vx = self._vy = self._wx = self._wy = 0.0

    def shape(self, dx: float, dy: float) -> tuple[float, float]:
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return (0.0, 0.0)

        w = min(self._w, dist)
        if dist >= self._area:
            # random wind walk, scaled down each frame
            self._wx = self._wx / _SQRT3 + (2.0 * self._rng.random() - 1.0) * w / _SQRT5
            self._wy = self._wy / _SQRT3 + (2.0 * self._rng.random() - 1.0) * w / _SQRT5
        else:
            self._wx /= _SQRT3
            self._wy /= _SQRT3

        # momentum += wind + gravity toward the target
        self._vx += self._wx + self._g * dx / dist
        self._vy += self._wy + self._g * dy / dist

        vmag = math.hypot(self._vx, self._vy)
        if vmag > self._max:
            clip = self._max / 2.0 + self._rng.random() * self._max / 2.0
            self._vx = self._vx / vmag * clip
            self._vy = self._vy / vmag * clip

        # never overshoot the per-frame target
        step = math.hypot(self._vx, self._vy)
        if step > dist:
            self._vx = self._vx / step * dist
            self._vy = self._vy / step * dist

        return (self._vx, self._vy)
