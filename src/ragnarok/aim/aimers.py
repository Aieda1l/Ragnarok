"""ragnarok.aim.aimers — Aimer ABC, NullAimer, FlickAimer, FeedbackAimer.

Aimer contract:
    step(crosshair, target_point, dt) -> (dx_px, dy_px)
    reset() -> None  (called on key re-trigger / target change)

All aimers operate in pixel space and must never overshoot the target.

FlickAimer:
    Latches the target_point on the FIRST call after reset(), then glides
    toward it at a constant speed (px/s), clamped to the remaining distance
    so it never overshoots.  The latched point is NOT updated until reset().

FeedbackAimer:
    P-controller on the live (target_point - crosshair) error, EMA-smoothed,
    magnitude-clamped to max_step_px.  Phase-4 hook: kff param present but
    unused (feed-forward term = kff * target_vel * dt, where kff=0 in Phase 3).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class Aimer(ABC):
    """Abstract base for all aimers.

    Every concrete aimer must implement step(); reset() is optional (default
    is a no-op) for aimers with no internal state to clear.
    """

    @abstractmethod
    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """Return (dx, dy) pixel delta to apply this frame.

        Args:
            crosshair:    Current crosshair position in ROI pixel space.
            target_point: Desired aim point in ROI pixel space.
            dt:           Frame duration in seconds (clamped by caller).
            target_vel:   IMM velocity estimate (px/s) for feed-forward aimers;
                          aimers that don't use it must accept and ignore it.

        Returns:
            (dx, dy) — signed pixel deltas, never causing an overshoot.
        """

    def reset(self) -> None:
        """Clear internal state.  Called on key re-trigger / target change."""


# ---------------------------------------------------------------------------
# NullAimer
# ---------------------------------------------------------------------------

class NullAimer(Aimer):
    """No-op aimer — always returns (0, 0).  Safe CI/test default."""

    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        return (0.0, 0.0)


# ---------------------------------------------------------------------------
# FlickAimer
# ---------------------------------------------------------------------------

class FlickAimer(Aimer):
    """Latch-once, constant-speed aimer.

    On the first call after construction or reset(), the *target_point* is
    latched.  Subsequent calls ignore the target_point argument and continue
    gliding toward the latched point at *flick_speed_px_s* px/s.

    The step is clamped to the remaining distance so the crosshair never
    overshoots.  A new target is only acquired after an explicit reset().
    """

    def __init__(self, *, flick_speed_px_s: float) -> None:
        self._speed: float = flick_speed_px_s
        self._latched: tuple[float, float] | None = None

    def reset(self) -> None:
        self._latched = None

    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        # Latch target_point on first call after reset (or construction).
        if self._latched is None:
            self._latched = target_point

        tx, ty = self._latched
        ex = tx - crosshair[0]
        ey = ty - crosshair[1]
        d = math.hypot(ex, ey)

        if d <= 1e-9:
            return (0.0, 0.0)

        # Glide at constant speed; clamp to remaining distance (no overshoot).
        step_len = min(self._speed * dt, d)
        return (ex / d * step_len, ey / d * step_len)


# ---------------------------------------------------------------------------
# FeedbackAimer
# ---------------------------------------------------------------------------

class FeedbackAimer(Aimer):
    """2-DOF PID: u = Kp·ē + Ki·∫e + Kd·d(ē)/dt + Kff·v̂ (spec §6.3).

    ē is the EMA-filtered error; the derivative is taken on the FILTERED error
    (no derivative kick). Three-fold anti-windup (spec §6.3): conditional
    integration (only when |e| <= cond_integ_thresh_px), an integral-contribution
    clamp (±integral_clamp on Ki·∫e), and freeze-on-saturation (back out the
    integral increment when the magnitude clamp fires). Output is magnitude-
    clamped to min(max_step_px, remaining distance) — never overshoots.

    Defaults (ki=0, kd=0, integral_clamp=None, cond_integ_thresh_px=None)
    reproduce the original P-controller behaviour exactly.
    """

    def __init__(self, *, kp: float, max_step_px: float, ema_alpha: float = 1.0,
                 kff: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 integral_clamp: float | None = None,
                 cond_integ_thresh_px: float | None = None) -> None:
        self._kp = kp
        self._max = max_step_px
        self._alpha = ema_alpha
        self._kff = kff
        self._ki = ki
        self._kd = kd
        self._iclamp = integral_clamp
        self._cond = cond_integ_thresh_px
        self._fx = 0.0
        self._fy = 0.0
        self._ix = 0.0
        self._iy = 0.0
        self._prev_fx = 0.0
        self._prev_fy = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False
        self._ix = 0.0
        self._iy = 0.0

    def step(self, crosshair, target_point, dt, target_vel=(0.0, 0.0)):
        ex = target_point[0] - crosshair[0]
        ey = target_point[1] - crosshair[1]

        if not self._initialized:
            self._fx, self._fy = ex, ey
            self._prev_fx, self._prev_fy = ex, ey
            self._initialized = True
            dfx = dfy = 0.0
        else:
            a = self._alpha
            self._fx += a * (ex - self._fx)
            self._fy += a * (ey - self._fy)
            if dt > 0.0:
                dfx = (self._fx - self._prev_fx) / dt
                dfy = (self._fy - self._prev_fy) / dt
            else:
                dfx = dfy = 0.0
            self._prev_fx, self._prev_fy = self._fx, self._fy

        # Conditional integration (anti-windup #1).
        e_mag = math.hypot(ex, ey)
        integrate = self._cond is None or e_mag <= self._cond
        inc_x = ex * dt if integrate else 0.0
        inc_y = ey * dt if integrate else 0.0
        self._ix += inc_x
        self._iy += inc_y

        # Integral contribution, clamped (anti-windup #2).
        icx = self._ki * self._ix
        icy = self._ki * self._iy
        if self._iclamp is not None:
            icx = max(-self._iclamp, min(self._iclamp, icx))
            icy = max(-self._iclamp, min(self._iclamp, icy))

        dx = self._kp * self._fx + icx + self._kd * dfx + self._kff * target_vel[0] * dt
        dy = self._kp * self._fy + icy + self._kd * dfy + self._kff * target_vel[1] * dt

        # Magnitude clamp: never overshoot remaining distance OR max step.
        mag = math.hypot(dx, dy)
        limit = min(self._max, e_mag)
        if mag > limit and mag > 0.0:
            scale = limit / mag
            dx *= scale
            dy *= scale
            # Freeze-on-saturation (anti-windup #3): undo this step's integration.
            self._ix -= inc_x
            self._iy -= inc_y

        return (dx, dy)


# ---------------------------------------------------------------------------
# HybridAimer
# ---------------------------------------------------------------------------

class HybridAimer(Aimer):
    """Proportional approach far out, full flick when close.

    error magnitude > flick_dist_px : smooth P-controller (EMA error, clamped
                                      to max_step_px) — covers long travel.
    error magnitude <= flick_dist_px: snap the full remaining error (clamped to
                                      the remaining distance, so no overshoot) —
                                      crisp final settle for snipers / low ROF.

    Position-only: ignores ``target_vel``; ``flick_speed_px_s`` is reserved/unused.
    """

    def __init__(
        self,
        *,
        kp: float,
        max_step_px: float,
        flick_dist_px: float,
        flick_speed_px_s: float,
        ema_alpha: float = 1.0,
    ) -> None:
        self._kp = kp
        self._max = max_step_px
        self._flick_dist = flick_dist_px
        self._speed = flick_speed_px_s
        self._alpha = ema_alpha
        self._fx = 0.0
        self._fy = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False

    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        ex = target_point[0] - crosshair[0]
        ey = target_point[1] - crosshair[1]
        d = math.hypot(ex, ey)
        if d <= 1e-9:
            return (0.0, 0.0)

        if d <= self._flick_dist:
            # Close: snap the full remaining error (already <= flick_dist, no clamp needed).
            self._initialized = False  # next far-approach re-seeds the EMA
            return (ex, ey)

        # Far: smooth proportional approach.
        if not self._initialized:
            self._fx, self._fy = ex, ey
            self._initialized = True
        else:
            a = self._alpha
            self._fx += a * (ex - self._fx)
            self._fy += a * (ey - self._fy)
        dx = self._kp * self._fx
        dy = self._kp * self._fy
        mag = math.hypot(dx, dy)
        limit = min(self._max, d)          # never exceed remaining distance OR max step
        if mag > limit and mag > 0.0:
            s = limit / mag
            dx *= s
            dy *= s
        return (dx, dy)


# ---------------------------------------------------------------------------
# PredictiveAimer
# ---------------------------------------------------------------------------

class PredictiveAimer(Aimer):
    """Crisp predicted-point aimer with velocity feed-forward.

    The controller feeds the IMM lead point as target_point and v̂ as
    target_vel. This aimer commands the full positional error to that predicted
    point (no smoothing) plus kff * v̂ * dt, magnitude-clamped to max_step_px.
    Best for fast, confidently-tracked targets where prediction beats damping.
    """

    def __init__(self, *, max_step_px: float, kff: float = 1.0) -> None:
        self._max = max_step_px
        self._kff = kff

    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        ex = target_point[0] - crosshair[0]
        ey = target_point[1] - crosshair[1]
        dx = ex + self._kff * target_vel[0] * dt
        dy = ey + self._kff * target_vel[1] * dt
        mag = math.hypot(dx, dy)
        if mag > self._max and mag > 0.0:
            s = self._max / mag
            dx *= s
            dy *= s
        return (dx, dy)
