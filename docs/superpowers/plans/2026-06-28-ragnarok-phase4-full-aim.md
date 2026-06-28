# Ragnarok Phase 4 — Full Aim System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the aim system on top of the Phase 3 core — add the remaining aimers (Hybrid, Predictive), live feed-forward velocity with anti-runaway guards, adaptive predictive lead, WindMouse motion shaping, recoil compensation, a safety-gated trigger bot, and a feed-forward "active GMC" ego-motion provider.

**Architecture:** Every new piece is a small, pure, dependency-injected unit (matching Phases 1–3): aimers are stateless-ish control laws in `aim/`, motion shaping lives in a new `motion/` package, recoil/trigger get their own packages, latency gets an adaptive-lead estimator, and the ego-motion seam (`tracking/egomotion.py`) gains a back-projection GMC. The `AimController` orchestrates them; the worker passes the GMC affine to the tracker. All side effects (mouse move/button) go through the existing injectable `MouseDriver`, so the entire phase is unit-testable with `NullMouseDriver` + fakes and stays CI-safe (no GPU/display/cursor).

**Tech Stack:** Python 3.11+, numpy, scipy, filterpy (existing IMM), stdlib `random` (seeded) for WindMouse, pydantic/tomlkit config. No new third-party dependencies.

## Global Constraints

- **Self-owned offline single-player game** — closed environment; "humanization", recoil, and trigger are control-quality/input engineering, not evasion (spec §Scope).
- **CI-safe always:** no real GPU, display, mouse, or keyboard in unit tests. Win32 access stays lazily bound inside `connect()`; all side effects via injected fakes (`NullMouseDriver`, `FakeKeyProvider`, injected `clock`, seeded `random.Random`).
- **Pixel space this phase.** Aim filter/lead run in ROI pixel space with identity ego-motion *by default*. The feed-forward GMC (Task 10) computes a real affine from *commanded* deltas, but full **world-angular target filtering** stays deferred (see Scope Boundary) — it requires the GMC validated empirically + τ_render from Phase 5 + physical-mouse passthrough from Phase 7.
- **No overshoot / no runaway:** every aimer clamps to remaining distance or `max_step_px`; feed-forward velocity is low-pass smoothed AND magnitude-clamped (spec §6.4); the per-tick `SendInputMouseDriver` clamp stays in force.
- **Fail-safe:** aim and trigger act only while their key is held/toggled AND `cfg.*.enabled`; loss of target, occlusion, or disengage resets all stateful units. Trigger never fires on a coasted/predicted box (spec §6.7).
- **Frozen pydantic config** (`model_config = ConfigDict(frozen=True)`), TOML round-trip, backward-compatible defaults (existing `config.toml` must still load).
- **TDD, frequent commits, exact file paths.** Match the surrounding code's style (module docstrings, `from __future__ import annotations`, keyword-only constructors).

## Scope Boundary (explicit deferrals — keep the plan honest)

These are spec items intentionally NOT implemented in Phase 4, with the reason and the seam left for them:

- **World-angular target filtering (§6.4 root fix):** deferred. Needs the feed-forward GMC empirically validated (`deg_per_count`/sign/`τ_render`) — that calibration is Phase 5 (diagnostics). Phase 4 ships the *guards* that work without it: feed-forward `v̂` smoothing + velocity clamp. The `fov.py` deg↔px seam and the GMC provider are the forward-compatible hooks.
- **Physical-mouse passthrough deltas (§5.3):** deferred to Phase 7 (Arduino USB Host Shield). The GMC `CommandedMotionBuffer` accepts an optional passthrough source now (defaults to none) so Phase 7 only injects it.
- **Recoil wall-learner wizard + firing-from-held-mouse detection (§6.6):** deferred to Phase 5/8 (diagnostics + GUI). Phase 4 ships the `RecoilCompensator` data structure + per-shot advance, driven by trigger-bot shots; a hand-authored pattern table works now.
- **Trigger line-clear pixel re-check (§6.7):** the controller is frame-free, so the "teammate pixel on the crosshair-to-target line" check is an injected predicate (default `True`). Wiring the real pixel scan happens when the trigger gets frame access (small follow-up; documented in Task 9).
- **Recoil folded into the camera-orientation integrator (§6.4/§6.6):** deferred with world-angular filtering; Phase 4 applies the recoil counter-move as an additive output offset (still correct, just not yet part of the ego-motion model).

---

## File Structure

**New files:**
- `src/ragnarok/motion/__init__.py` — package marker.
- `src/ragnarok/motion/shaper.py` — `MotionShaper` ABC, `NullShaper`, `WindMouseShaper`.
- `src/ragnarok/aim/velocity.py` — `VelocitySmoother` (low-pass + magnitude clamp for `v̂`).
- `src/ragnarok/latency/adaptive_lead.py` — `AdaptiveLead` (frame age + EWMA actuation latency).
- `src/ragnarok/recoil/__init__.py` — package marker.
- `src/ragnarok/recoil/compensator.py` — `RecoilPattern`, `RecoilCompensator`.
- `src/ragnarok/trigger/__init__.py` — package marker.
- `src/ragnarok/trigger/bot.py` — `TriggerBot` (safety gates).
- `tests/motion/__init__.py`, `tests/motion/test_shaper.py`
- `tests/aim/test_velocity.py`, `tests/aim/test_hybrid_predictive.py`
- `tests/latency/test_adaptive_lead.py`
- `tests/recoil/__init__.py`, `tests/recoil/test_compensator.py`
- `tests/trigger/__init__.py`, `tests/trigger/test_bot.py`
- `tests/tracking/test_ffgmc.py`
- `tests/config/test_phase4_config.py`

**Modified files:**
- `src/ragnarok/aim/aimers.py` — unify `step()` signature with `target_vel`; add `HybridAimer`, `PredictiveAimer`.
- `src/ragnarok/aim/mouse.py` — implement `SendInputMouseDriver.set_button`.
- `src/ragnarok/tracking/egomotion.py` — add `CommandedMotionBuffer`, `FeedForwardGMC`.
- `src/ragnarok/config/schema.py` — extend `AimConfig`; add `MotionConfig`, `RecoilConfig`, `TriggerConfig`; nest in `AppConfig`.
- `src/ragnarok/aim/controller.py` — orchestrate shaper, feed-forward velocity, adaptive lead, recoil, trigger.
- `src/ragnarok/wiring.py` — `build_aimer`, `build_shaper`, `build_recoil`, `build_trigger` helpers.
- `src/ragnarok/app.py` — wire the new components into `_build_aim_controller`.
- `tests/aim/test_aimers.py` — extend for unified signature (if present); else covered by new test files.

---

## Task 1: Unify the Aimer.step signature with optional target_vel

**Files:**
- Modify: `src/ragnarok/aim/aimers.py`
- Test: `tests/aim/test_hybrid_predictive.py` (new; also exercises the unified signature)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Aimer.step(crosshair, target_point, dt, target_vel=(0.0, 0.0)) -> (dx, dy)` on the ABC and ALL concrete aimers (`NullAimer`, `FlickAimer`, `FeedbackAimer`). `FlickAimer`/`NullAimer` accept and ignore `target_vel`. `FeedbackAimer` already has it — make the keyword identical.

- [ ] **Step 1: Write the failing test**

```python
# tests/aim/test_hybrid_predictive.py
from ragnarok.aim.aimers import NullAimer, FlickAimer, FeedbackAimer


def test_all_aimers_accept_target_vel_kwarg():
    # Uniform signature so the controller can always pass velocity.
    assert NullAimer().step((0, 0), (10, 0), 0.01, target_vel=(5.0, 0.0)) == (0.0, 0.0)
    fl = FlickAimer(flick_speed_px_s=1000.0)
    dx, dy = fl.step((0, 0), (10, 0), 0.01, target_vel=(5.0, 0.0))
    assert dx > 0 and dy == 0
    fb = FeedbackAimer(kp=0.5, max_step_px=100.0, ema_alpha=1.0)
    dx, dy = fb.step((0, 0), (10, 0), 0.01, target_vel=(0.0, 0.0))
    assert dx > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/aim/test_hybrid_predictive.py::test_all_aimers_accept_target_vel_kwarg -v`
Expected: FAIL — `NullAimer.step()`/`FlickAimer.step()` got an unexpected keyword argument `target_vel`.

- [ ] **Step 3: Add `target_vel` to the ABC and the existing aimers**

In `src/ragnarok/aim/aimers.py`, update the ABC `step` signature and the two aimers that lack the kwarg:

```python
# Aimer.step (abstractmethod) — add the parameter to the signature + docstring:
    @abstractmethod
    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """Return (dx, dy) pixel delta to apply this frame.

        target_vel is the IMM velocity estimate (px/s) for feed-forward aimers;
        aimers that don't use it must accept and ignore it.
        """
```

```python
# NullAimer.step:
    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        return (0.0, 0.0)
```

```python
# FlickAimer.step — add target_vel (ignored); body unchanged otherwise:
    def step(
        self,
        crosshair: tuple[float, float],
        target_point: tuple[float, float],
        dt: float,
        target_vel: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        if self._latched is None:
            self._latched = target_point
        tx, ty = self._latched
        ex = tx - crosshair[0]
        ey = ty - crosshair[1]
        d = math.hypot(ex, ey)
        if d <= 1e-9:
            return (0.0, 0.0)
        step_len = min(self._speed * dt, d)
        return (ex / d * step_len, ey / d * step_len)
```

`FeedbackAimer.step` already declares `target_vel` — leave it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/aim/test_hybrid_predictive.py::test_all_aimers_accept_target_vel_kwarg -v`
Expected: PASS

- [ ] **Step 5: Run the existing aim suite to confirm no regression**

Run: `python -m pytest tests/aim -q`
Expected: PASS (Phase 3 aim tests unaffected — they call `step` with 3 positional args).

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/aim/aimers.py tests/aim/test_hybrid_predictive.py
git commit -m "refactor(aim): unify Aimer.step with optional target_vel"
```

---

## Task 2: HybridAimer (proportional approach, flick when close)

**Files:**
- Modify: `src/ragnarok/aim/aimers.py`
- Test: `tests/aim/test_hybrid_predictive.py`

**Interfaces:**
- Consumes: `Aimer` ABC (Task 1).
- Produces: `HybridAimer(*, kp, max_step_px, flick_dist_px, flick_speed_px_s, ema_alpha=1.0)`. Behaviour: when error magnitude `> flick_dist_px`, behave like a smooth P-controller (EMA-smoothed, clamped to `max_step_px`); when `<= flick_dist_px`, snap the full remaining error (clamped to remaining distance — no overshoot). Stateless across targets except the EMA seed; `reset()` clears the EMA seed.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/aim/test_hybrid_predictive.py
from ragnarok.aim.aimers import HybridAimer


def test_hybrid_far_is_proportional_not_full():
    a = HybridAimer(kp=0.3, max_step_px=100.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (200.0, 0.0), 0.01)   # error 200 >> flick_dist
    assert 0 < dx < 200.0          # proportional: a fraction of the error
    assert abs(dx - 0.3 * 200.0) < 1e-6


def test_hybrid_close_snaps_full_error():
    a = HybridAimer(kp=0.3, max_step_px=100.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (5.0, 0.0), 0.01)     # error 5 < flick_dist
    assert abs(dx - 5.0) < 1e-6 and abs(dy) < 1e-6    # full snap, no overshoot


def test_hybrid_never_overshoots_close():
    a = HybridAimer(kp=2.0, max_step_px=1000.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), 0.01)
    assert 0 < dx <= 10.0 + 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_hybrid_predictive.py -k hybrid -v`
Expected: FAIL — `cannot import name 'HybridAimer'`.

- [ ] **Step 3: Implement HybridAimer**

Append to `src/ragnarok/aim/aimers.py`:

```python
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
        if mag > self._max and mag > 0.0:
            s = self._max / mag
            dx *= s
            dy *= s
        return (dx, dy)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/aim/test_hybrid_predictive.py -k hybrid -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/aim/aimers.py tests/aim/test_hybrid_predictive.py
git commit -m "feat(aim): HybridAimer (proportional approach + close-range flick)"
```

---

## Task 3: PredictiveAimer (snap to lead point + velocity feed-forward)

**Files:**
- Modify: `src/ragnarok/aim/aimers.py`
- Test: `tests/aim/test_hybrid_predictive.py`

**Interfaces:**
- Consumes: `Aimer` ABC.
- Produces: `PredictiveAimer(*, max_step_px, kff=1.0)`. The controller already passes the IMM **lead** point as `target_point` and `target_vel` as `v̂`. PredictiveAimer commands the full positional error to the predicted point (crisp, no smoothing) plus a velocity feed-forward `kff * target_vel * dt`, magnitude-clamped to `max_step_px`. Distinct from `FeedbackAimer` (smoothed P) and `FlickAimer` (latch-once).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/aim/test_hybrid_predictive.py
from ragnarok.aim.aimers import PredictiveAimer


def test_predictive_snaps_full_error_when_no_velocity():
    a = PredictiveAimer(max_step_px=1000.0, kff=1.0)
    dx, dy = a.step((0.0, 0.0), (30.0, 0.0), 0.01, target_vel=(0.0, 0.0))
    assert abs(dx - 30.0) < 1e-6 and abs(dy) < 1e-6


def test_predictive_adds_velocity_feedforward():
    a = PredictiveAimer(max_step_px=1000.0, kff=1.0)
    # zero positional error, but target moving right at 500 px/s over dt=0.01 -> +5 px FF
    dx, dy = a.step((0.0, 0.0), (0.0, 0.0), 0.01, target_vel=(500.0, 0.0))
    assert abs(dx - 5.0) < 1e-6


def test_predictive_magnitude_clamped():
    a = PredictiveAimer(max_step_px=10.0, kff=1.0)
    dx, dy = a.step((0.0, 0.0), (100.0, 0.0), 0.01)
    assert abs(math.hypot(dx, dy) - 10.0) < 1e-6
```

Add `import math` at the top of the test file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_hybrid_predictive.py -k predictive -v`
Expected: FAIL — `cannot import name 'PredictiveAimer'`.

- [ ] **Step 3: Implement PredictiveAimer**

Append to `src/ragnarok/aim/aimers.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/aim/test_hybrid_predictive.py -k predictive -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/aim/aimers.py tests/aim/test_hybrid_predictive.py
git commit -m "feat(aim): PredictiveAimer (predicted-point snap + velocity feed-forward)"
```

---

## Task 4: VelocitySmoother (low-pass + magnitude clamp for feed-forward v̂)

**Files:**
- Create: `src/ragnarok/aim/velocity.py`
- Test: `tests/aim/test_velocity.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `VelocitySmoother(*, alpha=0.5, max_px_s=4000.0)` with `smooth_clamp(vx, vy) -> (vx, vy)` (EMA low-pass then magnitude clamp) and `reset()`. These are the spec §6.4 feed-forward-runaway guards (`v̂` smoothing + velocity saturation).

- [ ] **Step 1: Write the failing tests**

```python
# tests/aim/test_velocity.py
import math
from ragnarok.aim.velocity import VelocitySmoother


def test_first_call_seeds_then_clamps():
    s = VelocitySmoother(alpha=0.5, max_px_s=100.0)
    vx, vy = s.smooth_clamp(50.0, 0.0)   # first call seeds from raw
    assert abs(vx - 50.0) < 1e-9 and vy == 0.0


def test_lowpass_damps_step():
    s = VelocitySmoother(alpha=0.5, max_px_s=1e9)
    s.smooth_clamp(0.0, 0.0)             # seed at 0
    vx, vy = s.smooth_clamp(100.0, 0.0)  # ema = 0 + 0.5*(100-0) = 50
    assert abs(vx - 50.0) < 1e-9


def test_magnitude_clamp():
    s = VelocitySmoother(alpha=1.0, max_px_s=100.0)
    vx, vy = s.smooth_clamp(300.0, 400.0)  # raw mag 500 -> clamp to 100
    assert abs(math.hypot(vx, vy) - 100.0) < 1e-6


def test_reset_reseeds():
    s = VelocitySmoother(alpha=0.5, max_px_s=1e9)
    s.smooth_clamp(0.0, 0.0)
    s.reset()
    vx, vy = s.smooth_clamp(80.0, 0.0)   # reseeds from raw after reset
    assert abs(vx - 80.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_velocity.py -v`
Expected: FAIL — `No module named 'ragnarok.aim.velocity'`.

- [ ] **Step 3: Implement VelocitySmoother**

```python
# src/ragnarok/aim/velocity.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/aim/test_velocity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/aim/velocity.py tests/aim/test_velocity.py
git commit -m "feat(aim): VelocitySmoother (feed-forward low-pass + velocity clamp)"
```

---

## Task 5: WindMouse motion shaper

**Files:**
- Create: `src/ragnarok/motion/__init__.py`
- Create: `src/ragnarok/motion/shaper.py`
- Create: `tests/motion/__init__.py`
- Create: `tests/motion/test_shaper.py`

**Interfaces:**
- Consumes: nothing (stdlib `random`, `math`).
- Produces:
  - `MotionShaper` ABC: `shape(dx, dy) -> (dx, dy)`, `reset()`.
  - `NullShaper`: identity (`shape` returns input unchanged).
  - `WindMouseShaper(*, gravity=9.0, wind=3.0, max_step=15.0, target_area=10.0, rng=None)`: per-frame WindMouse-derived shaper carrying internal momentum + wind state; converges to the per-frame target without overshoot. `rng` is an injected `random.Random` for deterministic tests.

- [ ] **Step 1: Write the failing tests**

```python
# tests/motion/test_shaper.py
import math
import random
from ragnarok.motion.shaper import NullShaper, WindMouseShaper


def test_null_shaper_is_identity():
    s = NullShaper()
    assert s.shape(3.0, -4.0) == (3.0, -4.0)


def test_windmouse_zero_delta_is_zero():
    s = WindMouseShaper(rng=random.Random(0))
    assert s.shape(0.0, 0.0) == (0.0, 0.0)


def test_windmouse_never_overshoots_single_step():
    s = WindMouseShaper(max_step=15.0, rng=random.Random(1))
    dx, dy = s.shape(5.0, 0.0)               # target only 5 px away this frame
    assert math.hypot(dx, dy) <= 5.0 + 1e-9


def test_windmouse_converges_to_fixed_target():
    # Repeatedly feed the remaining vector to a fixed destination; it should arrive.
    s = WindMouseShaper(gravity=9.0, wind=3.0, max_step=15.0, rng=random.Random(7))
    x, y = 0.0, 0.0
    dest = (300.0, 120.0)
    for _ in range(2000):
        dx, dy = s.shape(dest[0] - x, dest[1] - y)
        x += dx
        y += dy
        if math.hypot(dest[0] - x, dest[1] - y) < 1.0:
            break
    assert math.hypot(dest[0] - x, dest[1] - y) < 2.0


def test_windmouse_is_deterministic_with_seed():
    a = WindMouseShaper(rng=random.Random(42))
    b = WindMouseShaper(rng=random.Random(42))
    assert a.shape(100.0, 50.0) == b.shape(100.0, 50.0)


def test_windmouse_reset_clears_momentum():
    s = WindMouseShaper(rng=random.Random(3))
    s.shape(100.0, 0.0)
    s.reset()
    # after reset the internal velocity/wind are zero again; first step is small
    dx, dy = s.shape(1.0, 0.0)
    assert math.hypot(dx, dy) <= 1.0 + 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/motion/test_shaper.py -v`
Expected: FAIL — `No module named 'ragnarok.motion'`.

- [ ] **Step 3: Implement the package**

```python
# src/ragnarok/motion/__init__.py
```

```python
# src/ragnarok/motion/shaper.py
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
```

```python
# tests/motion/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/motion/test_shaper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/motion tests/motion
git commit -m "feat(motion): WindMouse + Null motion shapers (seeded, no overshoot)"
```

---

## Task 6: Adaptive lead estimator

**Files:**
- Create: `src/ragnarok/latency/adaptive_lead.py`
- Test: `tests/latency/test_adaptive_lead.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `AdaptiveLead(*, alpha=0.1, base_latency_s=0.0)` with:
  - `observe_actuation(latency_s)` — EWMA-update the actuation/transport latency estimate.
  - `lead_seconds(t_capture_ns, now_ns) -> float` — `max(0, (now - t_capture)/1e9) + ewma_latency`. (spec §6.5)
  - `latency_s` property (current EWMA).

- [ ] **Step 1: Write the failing tests**

```python
# tests/latency/test_adaptive_lead.py
from ragnarok.latency.adaptive_lead import AdaptiveLead


def test_lead_includes_frame_age_plus_base():
    al = AdaptiveLead(alpha=0.1, base_latency_s=0.005)
    lead = al.lead_seconds(t_capture_ns=1_000_000_000, now_ns=1_008_000_000)
    assert abs(lead - (0.008 + 0.005)) < 1e-9   # 8 ms age + 5 ms base


def test_frame_age_never_negative():
    al = AdaptiveLead(base_latency_s=0.0)
    lead = al.lead_seconds(t_capture_ns=2_000, now_ns=1_000)  # clock skew
    assert lead >= 0.0


def test_observe_actuation_ewma():
    al = AdaptiveLead(alpha=0.5, base_latency_s=0.0)
    al.observe_actuation(0.010)   # 0 + 0.5*(0.010-0) = 0.005
    assert abs(al.latency_s - 0.005) < 1e-9
    al.observe_actuation(0.010)   # 0.005 + 0.5*(0.010-0.005) = 0.0075
    assert abs(al.latency_s - 0.0075) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/latency/test_adaptive_lead.py -v`
Expected: FAIL — `No module named 'ragnarok.latency.adaptive_lead'`.

- [ ] **Step 3: Implement AdaptiveLead**

```python
# src/ragnarok/latency/adaptive_lead.py
"""Adaptive predictive lead (spec §6.5).

t_lead = true per-frame age (now - t_capture) + EWMA(actuation + transport
latency), recomputed each frame so prediction self-corrects under scheduling
jitter instead of using a fixed constant.
"""
from __future__ import annotations


class AdaptiveLead:
    def __init__(self, *, alpha: float = 0.1, base_latency_s: float = 0.0) -> None:
        self._alpha = alpha
        self._lat = base_latency_s

    @property
    def latency_s(self) -> float:
        return self._lat

    def observe_actuation(self, latency_s: float) -> None:
        self._lat += self._alpha * (latency_s - self._lat)

    def lead_seconds(self, t_capture_ns: int, now_ns: int) -> float:
        age = (now_ns - t_capture_ns) / 1e9
        if age < 0.0:
            age = 0.0
        return age + self._lat
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/latency/test_adaptive_lead.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/latency/adaptive_lead.py tests/latency/test_adaptive_lead.py
git commit -m "feat(latency): AdaptiveLead estimator (frame age + EWMA actuation)"
```

---

## Task 7: Recoil compensator

**Files:**
- Create: `src/ragnarok/recoil/__init__.py`
- Create: `src/ragnarok/recoil/compensator.py`
- Create: `tests/recoil/__init__.py`
- Create: `tests/recoil/test_compensator.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RecoilPattern(points: tuple[tuple[float, float], ...])` — frozen dataclass; `points` are **cumulative** crosshair drift (px) caused by recoil, indexed by shot number (shot 0 = `points[0]`).
  - `RecoilCompensator(pattern, *, scale=1.0)`:
    - `on_fire() -> (dx, dy)` — call once per shot; returns the per-shot **counter** delta (px) = `-(points[i] - points[i-1]) * scale` (with `points[-1]` treated as origin for shot 0), then advances. Past the end → `(0.0, 0.0)`.
    - `release()` — reset shot index to 0 (fire button released).

- [ ] **Step 1: Write the failing tests**

```python
# tests/recoil/test_compensator.py
from ragnarok.recoil.compensator import RecoilPattern, RecoilCompensator


def test_first_shot_counter_is_first_point_negated():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0), (0.0, 18.0)))
    rc = RecoilCompensator(pat, scale=1.0)
    assert rc.on_fire() == (0.0, 0.0)     # cumulative (0,0) -> no counter on shot 0


def test_subsequent_shots_counter_increments():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0), (0.0, 18.0)))
    rc = RecoilCompensator(pat)
    rc.on_fire()                          # shot 0
    assert rc.on_fire() == (0.0, -10.0)   # counter the +10 rise
    assert rc.on_fire() == (0.0, -8.0)    # counter the +8 rise


def test_scale_applied():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0)))
    rc = RecoilCompensator(pat, scale=0.5)
    rc.on_fire()
    assert rc.on_fire() == (0.0, -5.0)


def test_past_end_returns_zero():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0)))
    rc = RecoilCompensator(pat)
    rc.on_fire(); rc.on_fire()
    assert rc.on_fire() == (0.0, 0.0)


def test_release_resets_index():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0), (0.0, 18.0)))
    rc = RecoilCompensator(pat)
    rc.on_fire(); rc.on_fire()
    rc.release()
    assert rc.on_fire() == (0.0, 0.0)     # back to shot 0


def test_empty_pattern_is_safe():
    rc = RecoilCompensator(RecoilPattern(points=()))
    assert rc.on_fire() == (0.0, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/recoil/test_compensator.py -v`
Expected: FAIL — `No module named 'ragnarok.recoil'`.

- [ ] **Step 3: Implement the package**

```python
# src/ragnarok/recoil/__init__.py
```

```python
# src/ragnarok/recoil/compensator.py
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
    def __init__(self, pattern: RecoilPattern, *, scale: float = 1.0) -> None:
        self._pts = pattern.points
        self._scale = scale
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
```

```python
# tests/recoil/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/recoil/test_compensator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/recoil tests/recoil
git commit -m "feat(recoil): per-shot RecoilCompensator over a cumulative pattern table"
```

---

## Task 8: SendInput mouse button support

**Files:**
- Modify: `src/ragnarok/aim/mouse.py`
- Test: `tests/aim/test_mouse_button.py` (new)

**Interfaces:**
- Consumes: existing `MouseButton`, `_make_real_send` (the `(dx, dy, flags) -> int` callable), and the `MOUSEEVENTF_*DOWN/UP` constants already declared in the module.
- Produces: `SendInputMouseDriver.set_button(button, down)` emitting a 0-move event with the correct `MOUSEEVENTF_*DOWN`/`*UP` flag for `LEFT`/`RIGHT`/`MIDDLE`. (`NullMouseDriver.set_button` already records `(button, down)` — unchanged.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/aim/test_mouse_button.py
from ragnarok.aim.mouse import (
    SendInputMouseDriver, MouseButton,
    MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP,
)


def _recording_driver():
    calls = []
    drv = SendInputMouseDriver(send=lambda dx, dy, flags: calls.append((dx, dy, flags)) or 1)
    drv.connect()
    return drv, calls


def test_left_down_emits_leftdown_flag():
    drv, calls = _recording_driver()
    drv.set_button(MouseButton.LEFT, True)
    assert calls == [(0, 0, MOUSEEVENTF_LEFTDOWN)]


def test_left_up_emits_leftup_flag():
    drv, calls = _recording_driver()
    drv.set_button(MouseButton.LEFT, False)
    assert calls == [(0, 0, MOUSEEVENTF_LEFTUP)]


def test_right_button_flags():
    drv, calls = _recording_driver()
    drv.set_button(MouseButton.RIGHT, True)
    drv.set_button(MouseButton.RIGHT, False)
    assert calls == [(0, 0, MOUSEEVENTF_RIGHTDOWN), (0, 0, MOUSEEVENTF_RIGHTUP)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_mouse_button.py -v`
Expected: FAIL — `NotImplementedError: buttons deferred to Phase 4 trigger bot`.

- [ ] **Step 3: Implement set_button**

In `src/ragnarok/aim/mouse.py`, first add the middle-button constants next to the existing ones (after `MOUSEEVENTF_RIGHTUP`):

```python
MOUSEEVENTF_MIDDLEDOWN: int = 0x0020
MOUSEEVENTF_MIDDLEUP: int = 0x0040
```

Then replace the stubbed `SendInputMouseDriver.set_button` body:

```python
    _BUTTON_FLAGS = {
        MouseButton.LEFT: (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        MouseButton.RIGHT: (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        MouseButton.MIDDLE: (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    }

    def set_button(self, button: MouseButton, down: bool) -> None:
        down_flag, up_flag = self._BUTTON_FLAGS[button]
        self._send(0, 0, down_flag if down else up_flag)
```

(Note: `_BUTTON_FLAGS` is a class attribute; place it inside the `SendInputMouseDriver` class body, above `set_button`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/aim/test_mouse_button.py -v`
Expected: PASS

- [ ] **Step 5: Run the full aim suite (no regression on move/accumulator)**

Run: `python -m pytest tests/aim -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/aim/mouse.py tests/aim/test_mouse_button.py
git commit -m "feat(aim): SendInput mouse button press/release (left/right/middle)"
```

---

## Task 9: Trigger bot with safety gates

**Files:**
- Create: `src/ragnarok/trigger/__init__.py`
- Create: `src/ragnarok/trigger/bot.py`
- Create: `tests/trigger/__init__.py`
- Create: `tests/trigger/test_bot.py`

**Interfaces:**
- Consumes: `MouseButton`, a `MouseDriver` (`set_button`), `now_ns` clock, `Track` (uses `.xyxy`).
- Produces: `TriggerBot(*, mouse, activation_delay_s, button=MouseButton.LEFT, clock=now_ns)` with:
  - `update(*, track, crosshair, occluded, enemy_confirmed, line_clear, active) -> bool` — returns `True` on the frame a NEW press is issued (a "shot", which drives recoil advance). Press only when ALL gates hold: `active` AND `enemy_confirmed` AND `not occluded` AND `line_clear` AND crosshair inside `track.xyxy`, continuously for `activation_delay_s`. When any gate drops, release the button (if pressed) and clear the eligibility timer.
  - `release()` — force release + clear timer (disengage).
- Gate rationale (spec §6.7): never fires on a coasted/predicted box (`occluded`), requires enemy confirmation (selector already restricts to ENEMY; controller passes `True`), `line_clear` is the teammate-pixel safety check (injected predicate result; default `True` until frame-based scan is wired).

- [ ] **Step 1: Write the failing tests**

```python
# tests/trigger/test_bot.py
from ragnarok.core.types import Track, Team
from ragnarok.aim.mouse import NullMouseDriver, MouseButton
from ragnarok.trigger.bot import TriggerBot


def _track():
    return Track(track_id=1, xyxy=(100.0, 100.0, 200.0, 300.0),
                 confidence=0.9, class_id=0, team=Team.ENEMY)


class _Clock:
    def __init__(self):
        self.t = 0
    def __call__(self):
        return self.t


def _bot(delay_s=0.1):
    clk = _Clock()
    mouse = NullMouseDriver()
    mouse.connect()
    bot = TriggerBot(mouse=mouse, activation_delay_s=delay_s, clock=clk)
    return bot, mouse, clk


def _gates(**over):
    g = dict(track=_track(), crosshair=(150.0, 150.0), occluded=False,
             enemy_confirmed=True, line_clear=True, active=True)
    g.update(over)
    return g


def test_fires_after_activation_delay():
    bot, mouse, clk = _bot(delay_s=0.1)
    assert bot.update(**_gates()) is False        # t=0: eligibility starts
    clk.t = 50_000_000                            # 50 ms < 100 ms
    assert bot.update(**_gates()) is False
    clk.t = 120_000_000                           # 120 ms >= 100 ms
    assert bot.update(**_gates()) is True         # NEW press -> shot
    assert (MouseButton.LEFT, True) in mouse.buttons


def test_no_fire_when_inactive():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(active=False)) is False
    assert mouse.buttons == []


def test_no_fire_when_occluded():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(occluded=True)) is False
    assert mouse.buttons == []


def test_no_fire_when_crosshair_outside_hitbox():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(crosshair=(10.0, 10.0))) is False


def test_no_fire_when_line_blocked():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(line_clear=False)) is False


def test_releases_when_gate_drops():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates()) is True          # press
    bot.update(**_gates(active=False))             # gate drops -> release
    assert (MouseButton.LEFT, False) in mouse.buttons


def test_single_press_held_not_repeated():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates()) is True           # first press
    clk.t = 200_000_000
    assert bot.update(**_gates()) is False          # still held, no new shot
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/trigger/test_bot.py -v`
Expected: FAIL — `No module named 'ragnarok.trigger'`.

- [ ] **Step 3: Implement the package**

```python
# src/ragnarok/trigger/__init__.py
```

```python
# src/ragnarok/trigger/bot.py
"""Trigger bot with safety gates (spec §6.7).

Fires ONLY when every gate holds continuously for activation_delay_s:
  active (trigger key) AND enemy_confirmed AND not occluded (never a coasted
  box) AND line_clear (no teammate pixel on the path) AND crosshair inside the
  hitbox. Any gate dropping releases the button immediately. update() returns
  True on the frame a NEW press is issued, so the caller advances recoil.
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.aim.mouse import MouseButton


class TriggerBot:
    def __init__(
        self,
        *,
        mouse,
        activation_delay_s: float,
        button: MouseButton = MouseButton.LEFT,
        clock=now_ns,
    ) -> None:
        self._mouse = mouse
        self._delay = activation_delay_s
        self._button = button
        self._clock = clock
        self._pressed = False
        self._eligible_since: int | None = None

    @staticmethod
    def _inside(track, crosshair) -> bool:
        x1, y1, x2, y2 = track.xyxy
        cx, cy = crosshair
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def update(
        self,
        *,
        track,
        crosshair,
        occluded: bool,
        enemy_confirmed: bool,
        line_clear: bool,
        active: bool,
    ) -> bool:
        ready = (
            active
            and enemy_confirmed
            and not occluded
            and line_clear
            and track is not None
            and self._inside(track, crosshair)
        )
        if not ready:
            self._eligible_since = None
            self._release_if_pressed()
            return False

        now = self._clock()
        if self._eligible_since is None:
            self._eligible_since = now
        elapsed = (now - self._eligible_since) / 1e9
        if elapsed >= self._delay and not self._pressed:
            self._mouse.set_button(self._button, True)
            self._pressed = True
            return True
        return False

    def release(self) -> None:
        self._eligible_since = None
        self._release_if_pressed()

    def _release_if_pressed(self) -> None:
        if self._pressed:
            self._mouse.set_button(self._button, False)
            self._pressed = False
```

```python
# tests/trigger/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/trigger/test_bot.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/trigger tests/trigger
git commit -m "feat(trigger): safety-gated trigger bot (delay, occlusion, line-clear, hitbox)"
```

---

## Task 10: Feed-forward active GMC (back-projected commanded motion)

**Files:**
- Modify: `src/ragnarok/tracking/egomotion.py`
- Test: `tests/tracking/test_ffgmc.py` (new)

**Interfaces:**
- Consumes: `focal_length_px` from `ragnarok.aim.fov`; `EgoMotion` ABC (existing).
- Produces:
  - `CommandedMotionBuffer(maxlen=4096)` — ring buffer of `(t_ns, d_counts_x, d_counts_y)` with `push(t_ns, dcx, dcy)` and `integrate(t_lo_ns, t_hi_ns) -> (sum_dcx, sum_dcy)` (sum of deltas with timestamps in `[t_lo, t_hi]`).
  - `FeedForwardGMC(*, hfov_deg, screen_width_px, deg_per_count, tau_render_s=0.0, frame_dt_s=1/144, buffer=None)` implementing `EgoMotion.estimate(frame)`:
    - reads the per-frame capture time via `frame.t_capture_ns` (the worker passes the `Frame`),
    - integrates commanded counts over the render-time window `[t_capture - tau_render - frame_dt, t_capture - tau_render]`,
    - converts to yaw/pitch degrees (`d_counts * deg_per_count`) and back-projects to a pixel translation `t_x = -f_px * tan(yaw)`, `t_y = -f_px * tan(pitch)` (R = I for pure yaw/pitch),
    - returns the `(2, 3)` float32 affine `[[1,0,t_x],[0,1,t_y]]`.
  - `estimate(None)` (or a frame without `t_capture_ns`) returns identity — safe default.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_ffgmc.py
import math
import numpy as np
from ragnarok.core.types import Frame
from ragnarok.tracking.egomotion import CommandedMotionBuffer, FeedForwardGMC
from ragnarok.aim.fov import focal_length_px


def test_buffer_integrates_window():
    b = CommandedMotionBuffer()
    b.push(100, 1.0, 2.0)
    b.push(200, 3.0, 4.0)
    b.push(300, 5.0, 6.0)
    assert b.integrate(150, 250) == (3.0, 4.0)          # only t=200 in window
    assert b.integrate(100, 300) == (9.0, 12.0)         # all three


def test_identity_when_no_commanded_motion():
    g = FeedForwardGMC(hfov_deg=90.0, screen_width_px=1920, deg_per_count=0.02)
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=1_000_000_000, region=(0, 0, 4, 4))
    aff = g.estimate(frame)
    assert np.allclose(aff, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))


def test_none_frame_is_identity():
    g = FeedForwardGMC(hfov_deg=90.0, screen_width_px=1920, deg_per_count=0.02)
    assert np.allclose(g.estimate(None), np.eye(2, 3, dtype=np.float32))


def test_rightward_pan_translates_world_left():
    # Commanded +X counts => camera yaws right => static world shifts left on screen
    # => translation t_x should be negative.
    buf = CommandedMotionBuffer()
    g = FeedForwardGMC(hfov_deg=90.0, screen_width_px=1920, deg_per_count=0.02,
                       tau_render_s=0.0, frame_dt_s=0.01, buffer=buf)
    t_cap = 1_000_000_000
    buf.push(t_cap - 5_000_000, 100.0, 0.0)             # inside [t_cap-0.01, t_cap]
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=t_cap, region=(0, 0, 4, 4))
    aff = g.estimate(frame)
    yaw = math.radians(100.0 * 0.02)
    expected_tx = -focal_length_px(90.0, 1920) * math.tan(yaw)
    assert abs(aff[0, 2] - expected_tx) < 1e-3
    assert abs(aff[1, 2]) < 1e-6
    assert aff[0, 2] < 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_ffgmc.py -v`
Expected: FAIL — `cannot import name 'CommandedMotionBuffer'`.

- [ ] **Step 3: Implement CommandedMotionBuffer + FeedForwardGMC**

Append to `src/ragnarok/tracking/egomotion.py` (keep existing `EgoMotion`/`IdentityEgoMotion`):

```python
import math
from collections import deque

from ragnarok.aim.fov import focal_length_px


class CommandedMotionBuffer:
    """Timestamped ring buffer of commanded mouse-count deltas (spec §5.3).

    The controller pushes each frame's commanded (dx, dy) in mouse counts;
    FeedForwardGMC integrates them over the render-time window. A passthrough
    (physical-mouse) source can push into the same buffer in Phase 7.
    """

    def __init__(self, maxlen: int = 4096) -> None:
        self._buf: deque[tuple[int, float, float]] = deque(maxlen=maxlen)

    def push(self, t_ns: int, d_counts_x: float, d_counts_y: float) -> None:
        self._buf.append((t_ns, d_counts_x, d_counts_y))

    def integrate(self, t_lo_ns: int, t_hi_ns: int) -> tuple[float, float]:
        sx = 0.0
        sy = 0.0
        for t, dx, dy in self._buf:
            if t_lo_ns <= t <= t_hi_ns:
                sx += dx
                sy += dy
        return (sx, sy)


class FeedForwardGMC(EgoMotion):
    """Back-projects known camera motion into a 2x3 affine (spec §5.3).

    Instead of CV optical-flow GMC, integrate the commanded (and, later,
    passthrough) mouse counts over the tau_render-aligned window and convert to
    a pixel translation via the pinhole model. R = I for pure yaw/pitch.
    """

    def __init__(
        self,
        *,
        hfov_deg: float,
        screen_width_px: int,
        deg_per_count: float,
        tau_render_s: float = 0.0,
        frame_dt_s: float = 1.0 / 144.0,
        buffer: CommandedMotionBuffer | None = None,
    ) -> None:
        self._f = focal_length_px(hfov_deg, screen_width_px)
        self._deg_per_count = deg_per_count
        self._tau = tau_render_s
        self._frame_dt = frame_dt_s
        self.buffer = buffer if buffer is not None else CommandedMotionBuffer()

    def estimate(self, frame) -> np.ndarray:
        t_cap = getattr(frame, "t_capture_ns", None)
        if t_cap is None:
            return np.eye(2, 3, dtype=np.float32)
        hi = int(t_cap - self._tau * 1e9)
        lo = int(hi - self._frame_dt * 1e9)
        dcx, dcy = self.buffer.integrate(lo, hi)
        yaw = math.radians(dcx * self._deg_per_count)
        pitch = math.radians(dcy * self._deg_per_count)
        tx = -self._f * math.tan(yaw)
        ty = -self._f * math.tan(pitch)
        return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_ffgmc.py -v`
Expected: PASS

- [ ] **Step 5: Run the tracking suite (no regression)**

Run: `python -m pytest tests/tracking -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/tracking/egomotion.py tests/tracking/test_ffgmc.py
git commit -m "feat(egomotion): feed-forward active GMC from back-projected commanded motion"
```

---

## Task 11: Phase 4 config schema

**Files:**
- Modify: `src/ragnarok/config/schema.py`
- Test: `tests/config/test_phase4_config.py` (new)

**Interfaces:**
- Consumes: existing `AppConfig`, `AimConfig`.
- Produces:
  - `AimConfig` gains: `aimer` Literal extended to `"flick" | "feedback" | "hybrid" | "predictive"`; `kff: float = 0.0`; `vel_clamp_px_s: float = 4000.0`; `vel_smooth_alpha: float = 0.5`; `hybrid_flick_dist_px: float = 20.0`; `adaptive_lead: bool = True`; `lead_alpha: float = 0.1`.
  - New `MotionConfig`: `shaper: Literal["none", "windmouse"] = "none"`, `gravity=9.0`, `wind=3.0`, `max_step=15.0`, `target_area=10.0`.
  - New `RecoilConfig`: `enabled: bool = False`, `scale: float = 1.0`, `pattern: tuple[tuple[float, float], ...] = ()`.
  - New `TriggerConfig`: `enabled: bool = False`, `trigger_key: str = "VK_LBUTTON"`, `activation_delay_ms: float = 80.0`, `require_line_clear: bool = True`, `button: Literal["left", "right", "middle"] = "left"`.
  - `AppConfig` nests `motion: MotionConfig`, `recoil: RecoilConfig`, `trigger: TriggerConfig` (all default-constructed, backward-compatible).

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_phase4_config.py
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import (
    AimConfig, MotionConfig, RecoilConfig, TriggerConfig, AppConfig,
)


def test_aim_new_defaults():
    a = AimConfig()
    assert a.kff == 0.0
    assert a.vel_clamp_px_s == 4000.0
    assert a.vel_smooth_alpha == 0.5
    assert a.hybrid_flick_dist_px == 20.0
    assert a.adaptive_lead is True
    assert a.lead_alpha == 0.1


def test_aimer_accepts_hybrid_and_predictive():
    assert AimConfig(aimer="hybrid").aimer == "hybrid"
    assert AimConfig(aimer="predictive").aimer == "predictive"


def test_aimer_rejects_unknown():
    with pytest.raises(ValidationError):
        AimConfig(aimer="magic")


def test_motion_defaults():
    m = MotionConfig()
    assert m.shaper == "none"
    assert m.gravity == 9.0 and m.wind == 3.0
    assert m.max_step == 15.0 and m.target_area == 10.0


def test_recoil_defaults_and_pattern():
    r = RecoilConfig(pattern=((0.0, 0.0), (0.0, 10.0)))
    assert r.enabled is False and r.scale == 1.0
    assert r.pattern == ((0.0, 0.0), (0.0, 10.0))


def test_trigger_defaults():
    t = TriggerConfig()
    assert t.enabled is False
    assert t.trigger_key == "VK_LBUTTON"
    assert t.activation_delay_ms == 80.0
    assert t.require_line_clear is True
    assert t.button == "left"


def test_appconfig_nests_phase4_sections():
    app = AppConfig()
    assert isinstance(app.motion, MotionConfig)
    assert isinstance(app.recoil, RecoilConfig)
    assert isinstance(app.trigger, TriggerConfig)


def test_backward_compat_without_phase4_sections():
    app = AppConfig(detection={"backend": "rfdetr_torch", "model": "nano"})
    assert app.motion.shaper == "none"
    assert app.trigger.enabled is False


def test_frozen():
    with pytest.raises(Exception):
        MotionConfig().shaper = "windmouse"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/config/test_phase4_config.py -v`
Expected: FAIL — `cannot import name 'MotionConfig'`.

- [ ] **Step 3: Extend AimConfig and add the new config models**

In `src/ragnarok/config/schema.py`, change the `aimer` field and add the new `AimConfig` fields (place them after `lead_ms`):

```python
    aimer: Literal["flick", "feedback", "hybrid", "predictive"] = "feedback"
```

```python
    # --- Phase 4 additions ---
    kff: float = Field(default=0.0, ge=0.0, le=4.0)               # feed-forward velocity gain
    vel_clamp_px_s: float = Field(default=4000.0, gt=0.0)         # v̂ saturation
    vel_smooth_alpha: float = Field(default=0.5, gt=0.0, le=1.0)  # v̂ low-pass
    hybrid_flick_dist_px: float = Field(default=20.0, gt=0.0)     # HybridAimer threshold
    adaptive_lead: bool = True                                    # §6.5 adaptive vs fixed lead_ms
    lead_alpha: float = Field(default=0.1, gt=0.0, le=1.0)        # adaptive-lead EWMA
```

Add the three new models (before `AppConfig`):

```python
class MotionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    shaper: Literal["none", "windmouse"] = "none"
    gravity: float = Field(default=9.0, ge=0.0)
    wind: float = Field(default=3.0, ge=0.0)
    max_step: float = Field(default=15.0, gt=0.0)
    target_area: float = Field(default=10.0, gt=0.0)


class RecoilConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    scale: float = Field(default=1.0, ge=0.0)
    pattern: tuple[tuple[float, float], ...] = ()


class TriggerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enabled: bool = False
    trigger_key: str = "VK_LBUTTON"
    activation_delay_ms: float = Field(default=80.0, ge=0.0, le=2000.0)
    require_line_clear: bool = True
    button: Literal["left", "right", "middle"] = "left"
```

Extend `AppConfig`:

```python
class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    capture: CaptureConfig = CaptureConfig()
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    classification: ClassificationConfig = ClassificationConfig()
    aim: AimConfig = AimConfig()
    motion: MotionConfig = MotionConfig()
    recoil: RecoilConfig = RecoilConfig()
    trigger: TriggerConfig = TriggerConfig()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/config/test_phase4_config.py -v`
Expected: PASS

- [ ] **Step 5: Run the full config suite + TOML round-trip (no regression)**

Run: `python -m pytest tests/config -q`
Expected: PASS (the `save_config`/`load_config` round-trip handles the new nested tuples — pydantic serializes tuples to TOML arrays).

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/config/schema.py tests/config/test_phase4_config.py
git commit -m "feat(config): Phase 4 aim/motion/recoil/trigger config"
```

---

## Task 12: Controller integration (shaper, feed-forward, adaptive lead, recoil, trigger)

**Files:**
- Modify: `src/ragnarok/aim/controller.py`
- Test: `tests/aim/test_controller_phase4.py` (new)

**Interfaces:**
- Consumes: all Phase 4 units (`VelocitySmoother`, `MotionShaper`, `AdaptiveLead`, `RecoilCompensator`, `TriggerBot`, `CommandedMotionBuffer`) plus the Phase 3 collaborators.
- Produces: `AimController.__init__` gains optional keyword collaborators (all default `None`, preserving the Phase 3 constructor and tests):
  - `shaper=None` (defaults to a `NullShaper` internally),
  - `vel_smoother=None` (defaults to a pass-through that returns IMM velocity unchanged only if `kff>0`; if `None` and feed-forward unused, velocity stays `(0,0)`),
  - `adaptive_lead=None`,
  - `recoil=None`,
  - `trigger=None`,
  - `trigger_active=None` (callable like `is_aim_active`; defaults to "never"),
  - `line_clear=None` (callable `() -> bool`, defaults to `lambda: True`),
  - `commanded_buffer=None` (a `CommandedMotionBuffer` to push commanded counts into).
  - The controller reads `cfg.kff` (feed-forward gain) when present; falls back to `0.0`.
- Behaviour additions (in `update`), all gated by the collaborators being present:
  1. velocity feed-forward: `v̂` from IMM, conditioned by `vel_smoother`, passed as `target_vel` to `aimer.step` (only when `cfg.kff > 0`).
  2. adaptive lead: `t_lead = adaptive_lead.lead_seconds(t_capture_ns, clock())` when present and `cfg.adaptive_lead`, else `cfg.lead_ms/1000`.
  3. motion shaping: `aimer` delta → `shaper.shape(...)` → counts.
  4. trigger: `trigger.update(...)`; on a `True` shot, advance `recoil` and add its counter to the pixel delta BEFORE the counts conversion.
  5. push commanded counts to `commanded_buffer` (for the GMC).
  6. `_disengage` and target-switch reset `shaper`, `vel_smoother`, `trigger.release()`, `recoil.release()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/aim/test_controller_phase4.py
from ragnarok.core.types import Track, Tracks, Team
from ragnarok.config.schema import AimConfig
from ragnarok.aim.controller import AimController
from ragnarok.aim.select import TargetSelector
from ragnarok.aim.imm import IMMManager
from ragnarok.aim.aimers import FeedbackAimer
from ragnarok.aim.velocity import VelocitySmoother
from ragnarok.aim.mouse import NullMouseDriver, MouseButton
from ragnarok.motion.shaper import NullShaper
from ragnarok.recoil.compensator import RecoilPattern, RecoilCompensator
from ragnarok.trigger.bot import TriggerBot
from ragnarok.tracking.egomotion import CommandedMotionBuffer


def _enemy(tid=1, xyxy=(250.0, 180.0, 290.0, 300.0)):   # to the RIGHT of the crosshair
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=0, team=Team.ENEMY)


def _selector():
    return TargetSelector(fov_px=400.0, retain_fov_px=500.0, dwell_ms=0.0,
                          switch_margin=0.0, clock=lambda: 0)


def test_commanded_counts_pushed_to_buffer():
    cfg = AimConfig(enabled=True)
    buf = CommandedMotionBuffer()
    mouse = NullMouseDriver(); mouse.connect()
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: True, roi_size=384,
        shaper=NullShaper(), commanded_buffer=buf, clock=lambda: 1,
    )
    ac.update(Tracks((_enemy(),)), 0)
    # one (t, dcx, dcy) entry pushed this frame
    assert len(buf._buf) == 1


def test_trigger_fires_and_advances_recoil():
    cfg = AimConfig(enabled=True)
    mouse = NullMouseDriver(); mouse.connect()
    # crosshair is ROI centre (192,192); place an enemy whose box covers it.
    enemy = _enemy(xyxy=(150.0, 150.0, 240.0, 260.0))
    rc = RecoilCompensator(RecoilPattern(points=((0.0, 0.0), (0.0, 10.0))))
    trig = TriggerBot(mouse=mouse, activation_delay_s=0.0, clock=lambda: 0)
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: True, roi_size=384,
        shaper=NullShaper(), recoil=rc, trigger=trig,
        trigger_active=lambda: True, clock=lambda: 0,
    )
    ac.update(Tracks((enemy,)), 0)
    assert (MouseButton.LEFT, True) in mouse.buttons   # trigger fired


def test_disengage_releases_trigger():
    cfg = AimConfig(enabled=True)
    mouse = NullMouseDriver(); mouse.connect()
    enemy = _enemy(xyxy=(150.0, 150.0, 240.0, 260.0))
    trig = TriggerBot(mouse=mouse, activation_delay_s=0.0, clock=lambda: 0)
    active = {"v": True}
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: active["v"], roi_size=384,
        shaper=NullShaper(), trigger=trig, trigger_active=lambda: True,
        clock=lambda: 0,
    )
    ac.update(Tracks((enemy,)), 0)
    active["v"] = False
    ac.update(Tracks((enemy,)), 8_000_000)
    assert (MouseButton.LEFT, False) in mouse.buttons   # released on disengage


def test_phase3_constructor_still_works():
    # Backward compat: no Phase 4 collaborators -> behaves like Phase 3.
    cfg = AimConfig(enabled=True)
    mouse = NullMouseDriver(); mouse.connect()
    ac = AimController(
        cfg, selector=_selector(), imm_manager=IMMManager(),
        aimer=FeedbackAimer(kp=0.5, max_step_px=500.0, ema_alpha=1.0),
        mouse=mouse, is_aim_active=lambda: True, roi_size=384,
    )
    for i in range(5):
        ac.update(Tracks((_enemy(),)), i * 8_000_000)
    assert sum(m[0] for m in mouse.moves) > 0           # still aims
    assert mouse.buttons == []                          # no trigger wired
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/aim/test_controller_phase4.py -v`
Expected: FAIL — `AimController.__init__()` got an unexpected keyword argument `shaper`.

- [ ] **Step 3: Rewrite the controller**

Replace `src/ragnarok/aim/controller.py` with:

```python
"""AimController — ties FOV/selection → IMM lead → aimer → shaper → mouse,
with feed-forward velocity, adaptive lead, recoil, and a safety-gated trigger.

All collaborators are injected; the Phase 4 ones are optional (default None) so
the Phase 3 constructor and tests keep working. Side effects (mouse move/button)
happen only while is_aim_active()/cfg.enabled (aim) and trigger_active() (fire).
Targets are ENEMY-only (the selector enforces this).

Pixel space this phase (identity ego-motion by default). Commanded counts are
pushed to a CommandedMotionBuffer so a FeedForwardGMC can back-project them.
"""
from __future__ import annotations

from ragnarok.core.clock import now_ns
from ragnarok.core.types import Tracks
from ragnarok.aim.fov import aim_point
from ragnarok.motion.shaper import NullShaper


class AimController:
    def __init__(
        self,
        cfg,
        *,
        selector,
        imm_manager,
        aimer,
        mouse,
        is_aim_active,
        roi_size: int,
        clock=now_ns,
        shaper=None,
        vel_smoother=None,
        adaptive_lead=None,
        recoil=None,
        trigger=None,
        trigger_active=None,
        line_clear=None,
        commanded_buffer=None,
    ) -> None:
        self._cfg = cfg
        self._sel = selector
        self._imm = imm_manager
        self._aimer = aimer
        self._mouse = mouse
        self._active = is_aim_active
        self._cx = roi_size / 2.0
        self._cy = roi_size / 2.0
        self._deg_per_px = cfg.hfov_deg / float(cfg.screen_width_px)
        self._clock = clock
        self._shaper = shaper if shaper is not None else NullShaper()
        self._vel = vel_smoother
        self._lead = adaptive_lead
        self._recoil = recoil
        self._trigger = trigger
        self._trigger_active = trigger_active if trigger_active is not None else (lambda: False)
        self._line_clear = line_clear if line_clear is not None else (lambda: True)
        self._cmd_buf = commanded_buffer
        self._kff = float(getattr(cfg, "kff", 0.0))
        self._adaptive = bool(getattr(cfg, "adaptive_lead", False))
        self._last_ns: int | None = None
        self._cur_target: int | None = None
        self.target_id: int | None = None

    def update(self, tracks: Tracks, t_capture_ns: int) -> None:
        self._imm.prune({t.track_id for t in tracks})

        if not (self._cfg.enabled and self._active()):
            self._disengage()
            return

        tid = self._sel.select(tracks, self._cx, self._cy)
        self.target_id = tid
        if tid is None:
            self._reset_stateful()
            self._cur_target = None
            self._last_ns = t_capture_ns
            return

        if tid != self._cur_target:
            self._reset_stateful()
            self._cur_target = tid

        track = next((t for t in tracks if t.track_id == tid), None)
        if track is None:
            return

        ax, ay = aim_point(track, self._cfg.head_frac, self._cfg.aim_point)
        dt = self._dt(t_capture_ns)
        self._imm.update(tid, ax, ay, dt)

        # predictive lead
        if self._lead is not None and self._adaptive:
            t_lead = self._lead.lead_seconds(t_capture_ns, self._clock())
        else:
            t_lead = self._cfg.lead_ms / 1000.0
        lead_pt = self._imm.lead(tid, t_lead)

        # feed-forward velocity (smoothed + clamped) — only if used
        tvx, tvy = 0.0, 0.0
        if self._kff > 0.0:
            vx, vy = self._imm.velocity(tid)
            if self._vel is not None:
                vx, vy = self._vel.smooth_clamp(vx, vy)
            tvx, tvy = vx, vy

        dpx, dpy = self._aimer.step((self._cx, self._cy), lead_pt, dt, target_vel=(tvx, tvy))
        sx, sy = self._shaper.shape(dpx, dpy)

        # trigger + recoil
        if self._trigger is not None:
            fired = self._trigger.update(
                track=track,
                crosshair=(self._cx, self._cy),
                occluded=track.time_since_update > 0,
                enemy_confirmed=True,           # selector restricts to ENEMY
                line_clear=self._line_clear(),
                active=self._trigger_active(),
            )
            if fired and self._recoil is not None:
                rx, ry = self._recoil.on_fire()
                sx += rx
                sy += ry

        k = self._deg_per_px / self._cfg.sensitivity   # px → mouse counts
        cdx, cdy = sx * k, sy * k
        self._mouse.move_relative(cdx, cdy)
        if self._cmd_buf is not None:
            self._cmd_buf.push(self._clock(), cdx, cdy)

    def _dt(self, t_ns: int) -> float:
        if self._last_ns is None:
            self._last_ns = t_ns
            return 1.0 / 120.0
        dt = (t_ns - self._last_ns) / 1e9
        self._last_ns = t_ns
        return max(1e-3, min(0.1, dt))

    def _reset_stateful(self) -> None:
        self._aimer.reset()
        self._shaper.reset()
        if self._vel is not None:
            self._vel.reset()
        if self._trigger is not None:
            self._trigger.release()
        if self._recoil is not None:
            self._recoil.release()

    def _disengage(self) -> None:
        self._reset_stateful()
        self._sel.reset()
        self._last_ns = None
        self._cur_target = None
        self.target_id = None
```

- [ ] **Step 4: Run the new + Phase 3 controller tests**

Run: `python -m pytest tests/aim/test_controller_phase4.py tests/aim/test_controller.py -v`
Expected: PASS (both files).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/aim/controller.py tests/aim/test_controller_phase4.py
git commit -m "feat(aim): controller wires shaper, feed-forward, adaptive lead, recoil, trigger"
```

---

## Task 13: Wiring + app integration + full-suite green

**Files:**
- Modify: `src/ragnarok/wiring.py`
- Modify: `src/ragnarok/app.py`
- Test: `tests/test_wiring.py` (extend)

**Interfaces:**
- Consumes: `AppConfig` (with Phase 4 sections), all Phase 4 units.
- Produces:
  - `build_aimer(cfg) -> Aimer` — selects `flick`/`feedback`/`hybrid`/`predictive` from `cfg.aim`, passing the relevant params (`kp`, `max_step_px`, `ema_alpha`, `kff`, `flick_speed_px_s`, `hybrid_flick_dist_px`).
  - `build_shaper(cfg) -> MotionShaper` — `NullShaper` or `WindMouseShaper` from `cfg.motion`.
  - `build_recoil(cfg) -> RecoilCompensator | None` — `None` if `not cfg.recoil.enabled` or empty pattern.
  - `app._build_aim_controller` wires `build_aimer`, `build_shaper`, `build_recoil`, a `VelocitySmoother`, an `AdaptiveLead`, a `TriggerBot` (+ `AsyncKeyStateProvider` for `cfg.trigger.trigger_key`), and a `CommandedMotionBuffer` into the `AimController`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_wiring.py
from ragnarok.wiring import build_aimer, build_shaper, build_recoil
from ragnarok.aim.aimers import FlickAimer, FeedbackAimer, HybridAimer, PredictiveAimer
from ragnarok.motion.shaper import NullShaper, WindMouseShaper


def test_build_aimer_variants():
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "flick"})), FlickAimer)
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "feedback"})), FeedbackAimer)
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "hybrid"})), HybridAimer)
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "predictive"})), PredictiveAimer)


def test_build_shaper_variants():
    assert isinstance(build_shaper(AppConfig()), NullShaper)              # default "none"
    assert isinstance(build_shaper(AppConfig(motion={"shaper": "windmouse"})), WindMouseShaper)


def test_build_recoil_disabled_is_none():
    assert build_recoil(AppConfig()) is None                              # disabled by default


def test_build_recoil_enabled():
    cfg = AppConfig(recoil={"enabled": True, "pattern": ((0.0, 0.0), (0.0, 10.0))})
    rc = build_recoil(cfg)
    assert rc is not None
    rc.on_fire()
    assert rc.on_fire() == (0.0, -10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_wiring.py -k "aimer or shaper or recoil" -v`
Expected: FAIL — `cannot import name 'build_aimer'`.

- [ ] **Step 3: Add the builders to wiring.py**

Append to `src/ragnarok/wiring.py`:

```python
def build_aimer(cfg: AppConfig):
    a = cfg.aim
    from ragnarok.aim.aimers import (
        FlickAimer, FeedbackAimer, HybridAimer, PredictiveAimer,
    )
    if a.aimer == "flick":
        return FlickAimer(flick_speed_px_s=a.flick_speed_px_s)
    if a.aimer == "hybrid":
        return HybridAimer(
            kp=a.kp, max_step_px=a.max_step_px,
            flick_dist_px=a.hybrid_flick_dist_px,
            flick_speed_px_s=a.flick_speed_px_s, ema_alpha=a.ema_alpha,
        )
    if a.aimer == "predictive":
        return PredictiveAimer(max_step_px=a.max_step_px, kff=a.kff)
    return FeedbackAimer(
        kp=a.kp, max_step_px=a.max_step_px, ema_alpha=a.ema_alpha, kff=a.kff,
    )


def build_shaper(cfg: AppConfig):
    m = cfg.motion
    from ragnarok.motion.shaper import NullShaper, WindMouseShaper
    if m.shaper == "windmouse":
        return WindMouseShaper(
            gravity=m.gravity, wind=m.wind,
            max_step=m.max_step, target_area=m.target_area,
        )
    return NullShaper()


def build_recoil(cfg: AppConfig):
    r = cfg.recoil
    if not r.enabled or not r.pattern:
        return None
    from ragnarok.recoil.compensator import RecoilPattern, RecoilCompensator
    return RecoilCompensator(RecoilPattern(points=r.pattern), scale=r.scale)
```

- [ ] **Step 4: Run the wiring tests**

Run: `python -m pytest tests/test_wiring.py -v`
Expected: PASS

- [ ] **Step 5: Wire app.py**

In `src/ragnarok/app.py`, rewrite `_build_aim_controller` to use the builders and the Phase 4 collaborators. Replace the existing aimer-construction block + the `return` with:

```python
def _build_aim_controller(cfg):
    """Build the AimController from cfg (Windows-only deps imported lazily).

    Wires the Phase 4 collaborators: selected aimer, motion shaper, velocity
    smoother, adaptive lead, recoil compensator, and a safety-gated trigger bot
    (with its own key provider), plus a commanded-motion buffer for the GMC.
    """
    from ragnarok.aim.keys import AsyncKeyStateProvider, make_aim_active
    from ragnarok.aim.mouse import SendInputMouseDriver, MouseButton
    from ragnarok.aim.fov import fov_deg_to_radius_px
    from ragnarok.aim.select import TargetSelector
    from ragnarok.aim.imm import IMMManager
    from ragnarok.aim.velocity import VelocitySmoother
    from ragnarok.aim.controller import AimController
    from ragnarok.latency.adaptive_lead import AdaptiveLead
    from ragnarok.trigger.bot import TriggerBot
    from ragnarok.tracking.egomotion import CommandedMotionBuffer
    from ragnarok.wiring import build_aimer, build_shaper, build_recoil

    a = cfg.aim
    fov_px = fov_deg_to_radius_px(a.aim_fov_deg, a.hfov_deg, a.screen_width_px)
    retain_px = fov_deg_to_radius_px(a.retain_fov_deg, a.hfov_deg, a.screen_width_px)
    selector = TargetSelector(fov_px=fov_px, retain_fov_px=retain_px,
                              dwell_ms=a.dwell_ms, switch_margin=a.switch_margin,
                              head_frac=a.head_frac)
    mouse = SendInputMouseDriver()
    mouse.connect()
    is_active = make_aim_active(AsyncKeyStateProvider(a.aim_key), toggle=a.toggle)

    trigger = None
    trigger_active = None
    if cfg.trigger.enabled:
        btn = {"left": MouseButton.LEFT, "right": MouseButton.RIGHT,
               "middle": MouseButton.MIDDLE}[cfg.trigger.button]
        trigger = TriggerBot(mouse=mouse,
                             activation_delay_s=cfg.trigger.activation_delay_ms / 1000.0,
                             button=btn)
        trigger_active = make_aim_active(
            AsyncKeyStateProvider(cfg.trigger.trigger_key), toggle=False)

    return AimController(
        a, selector=selector, imm_manager=IMMManager(),
        aimer=build_aimer(cfg), mouse=mouse, is_aim_active=is_active,
        roi_size=cfg.capture.roi_size,
        shaper=build_shaper(cfg),
        vel_smoother=VelocitySmoother(alpha=a.vel_smooth_alpha, max_px_s=a.vel_clamp_px_s),
        adaptive_lead=AdaptiveLead(alpha=a.lead_alpha, base_latency_s=a.lead_ms / 1000.0),
        recoil=build_recoil(cfg),
        trigger=trigger, trigger_active=trigger_active,
        commanded_buffer=CommandedMotionBuffer(),
    )
```

- [ ] **Step 6: Verify app imports + builds under offscreen Qt**

Run:
```bash
QT_QPA_PLATFORM=offscreen python -c "import ragnarok.app as a; from ragnarok.config.schema import AppConfig; print(type(a._build_aim_controller(AppConfig(aim={'enabled':True,'aimer':'hybrid'}, trigger={'enabled':False}))).__name__)"
```
Expected: prints `AimController` (no exception). (`enabled` aim builds a real `SendInputMouseDriver`; `connect()` binds nothing until a real move — safe on CI as long as no move is issued.)

- [ ] **Step 7: Run the FULL suite**

Run: `python -m pytest -q`
Expected: PASS (all prior + all Phase 4 tests).

- [ ] **Step 8: Commit**

```bash
git add src/ragnarok/wiring.py src/ragnarok/app.py tests/test_wiring.py
git commit -m "feat(app): wire Phase 4 aimers/shaper/recoil/trigger/adaptive-lead into the controller"
```

---

## Phase 4 completion checklist

- [ ] All aimers present: Flick, Feedback, **Hybrid**, **Predictive** (Tasks 1–3).
- [ ] Feed-forward `Kff·v̂` live with **v̂ smoothing + velocity clamp** anti-runaway guards (Tasks 4, 12).
- [ ] **WindMouse** motion shaping (+ Null) (Task 5, 12).
- [ ] **Adaptive predictive lead** (Task 6, 12).
- [ ] **Recoil** compensator over a pattern table, advanced by trigger shots (Tasks 7, 12).
- [ ] **Trigger bot** with all safety gates (Tasks 8, 9, 12).
- [ ] **Feed-forward active GMC** from back-projected commanded motion (Task 10).
- [ ] Config + wiring + app integration; full suite green (Tasks 11, 13).
- [ ] Scope-Boundary deferrals documented (world-angular target filter, passthrough deltas, wall-learner, line-clear pixel scan, recoil-into-ego-motion).

After merge: update project memory (Phase 4 complete; list deferrals as the Phase 4b/5 follow-ups) and consider the worker integration of `FeedForwardGMC` (worker passing `gmc.estimate(frame)` as `ego_affine` instead of `IDENTITY_AFFINE`) — small, but it couples the controller's `commanded_buffer` to the worker's tracker call, so it belongs with the empirical GMC calibration in Phase 5.
```
